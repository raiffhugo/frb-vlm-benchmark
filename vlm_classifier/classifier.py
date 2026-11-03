from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vlm_classifier.models import BaseVLM
from vlm_classifier.parser import (
    TASK_MULTICLASS,
    map_label_for_task,
    normalize_task,
    parse_model_response,
)
from vlm_classifier.prompt import prompt_for_task


LOGGER = logging.getLogger(__name__)

FATAL_MODEL_ERROR_MARKERS = (
    "cuda error: device-side assert triggered",
    "cudaerrorassert",
)

TRANSIENT_MODEL_ERROR_MARKERS = (
    "cuda",
    "out of memory",
    "oom",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection",
    "ioerror",
    "i/o",
    "oserror",
    "runtimeerror",
)
TRANSIENT_MODEL_EXCEPTIONS = (TimeoutError, OSError, RuntimeError)

# Pipeline log: versioned, but written by the runs themselves. It is left out of the
# dirty-tree check, otherwise one run would dirty the provenance of the parallel ones.
RUNTIME_LOG = "logs/simulation.log"

# Files that determine the model's answer to an image. An interrupted run is resumed under
# another commit only if they are identical in both; manifest, prompt, model (revision,
# versions, generation) and configuration are compared through the provenance of each prediction.
INFERENCE_PATHS = (
    "vlm_classifier/models.py",
    "vlm_classifier/parser.py",
    "vlm_classifier/prompt.py",
    "pyproject.toml",
    "uv.lock",
)
_RESUME_SECTIONS = ("manifest", "prompt", "model", "classifier")
# each prediction goes to disk when written; the fsync bounds the loss in a power failure
FSYNC_EVERY = 20


@dataclass(frozen=True)
class ClassifierConfig:
    manifest_path: Path
    output_path: Path
    max_retries: int = 2
    retry_delay: float = 0.5
    task: str = TASK_MULTICLASS
    decision_threshold: float = 0.5
    # Alternative prompt (revision v4: lean, lean+reason, E6, E7). The file text is
    # used byte for byte; the provenance records its hash and source.
    prompt_path: Path | None = None

    def validate(self) -> None:
        normalize_task(self.task)
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0.")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be between 0 and 1.")


class VLMClassifier:
    def __init__(
        self,
        *,
        config: ClassifierConfig,
        model: BaseVLM,
        project_root: Path,
        git_provenance: dict[str, Any] | None = None,
    ) -> None:
        config.validate()
        self.config = ClassifierConfig(
            manifest_path=config.manifest_path,
            output_path=config.output_path,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            task=normalize_task(config.task),
            decision_threshold=config.decision_threshold,
            prompt_path=config.prompt_path,
        )
        self.model = model
        self.project_root = project_root.resolve()
        self.manifest_path = self._resolve(config.manifest_path)
        self.output_path = self._resolve(config.output_path)
        if self.manifest_path == self.output_path:
            raise ValueError(
                "manifest_path and output_path must point to different files."
            )
        if config.prompt_path is None:
            self.prompt = prompt_for_task(self.config.task)
            self.prompt_source = f"builtin:{self.config.task}"
        else:
            prompt_path = self._resolve(config.prompt_path)
            self.prompt = prompt_path.read_text(encoding="utf-8")
            if not self.prompt.strip():
                raise ValueError(f"Empty prompt: {prompt_path}")
            self.prompt_source = f"file:{self._display_path(prompt_path)}"
        self._manifest_sha256: str | None = None
        # Captured before any log reaches the disk: logs/simulation.log is versioned.
        self._git_provenance = dict(
            git_provenance
            if git_provenance is not None
            else read_git_provenance(self.project_root)
        )

    def run(self, *, overwrite: bool = False, resume: bool = False) -> list[dict[str, Any]]:
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive.")
        image_records = self._load_manifest()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self._reset_outputs()
        done: list[dict[str, Any]] = []
        if self.output_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Predictions file already exists: {self.output_path}. "
                    "Use --overwrite or --resume."
                )
            done = self._resumable_prefix(image_records)
            self._rewrite(done)  # drops a final line truncated by the interruption
            LOGGER.info("Resuming %s after %d predictions.", self.output_path, len(done))
        marker = (
            {
                "resumed_at_record": len(done),
                "previous_commit": done[-1]["provenance"]["git"].get("commit"),
            }
            if done
            else None
        )

        predictions: list[dict[str, Any]] = list(done)
        with self.output_path.open("a" if done else "w", encoding="utf-8") as handle:
            pending = image_records[len(done) :]
            progress = tqdm(
                pending,
                desc="Classifying images",
                unit="image",
                initial=len(done),
                total=len(image_records),
            )
            for count, image_record in enumerate(progress, start=1):
                prediction = self._classify_one(image_record)
                if marker is not None:
                    prediction["provenance"]["resume"] = marker
                predictions.append(prediction)
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                handle.flush()
                if count % FSYNC_EVERY == 0:
                    os.fsync(handle.fileno())
                if prediction["error"] and _is_fatal_model_error(str(prediction["error"])):
                    raise RuntimeError(
                        "Fatal CUDA error during classification. The CUDA context of "
                        "the process may have become invalid; restart the command. "
                        "Also try --attn-implementation none or "
                        "--cache-implementation none if the error persists."
                    )

        LOGGER.info("Predictions written to %s", self.output_path)
        return predictions

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Image manifest not found: {self.manifest_path}")

        # Read once: the hash recorded in the provenance is that of the content used.
        content = self.manifest_path.read_bytes()
        self._manifest_sha256 = hashlib.sha256(content).hexdigest()
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            content.decode("utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid line in {self.manifest_path}:{line_number}"
                ) from exc
            self._validate_manifest_record(record, line_number)
            records.append(record)

        if not records:
            raise ValueError(f"No images found in {self.manifest_path}")
        return records

    def _validate_manifest_record(self, record: dict[str, Any], line_number: int) -> None:
        required = {"sample_id", "image_path", "true_label", "simulation_parameters"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Line {line_number} missing required keys: {', '.join(missing)}"
            )

    def _classify_one(self, image_record: dict[str, Any]) -> dict[str, Any]:
        image_path = self._resolve(Path(image_record["image_path"]))
        raw_response: str | None = None
        parsed_response: dict[str, Any] | None = None
        error: str | None = None
        content_warning: str | None = None
        inference_seconds: float | None = None

        if not image_path.exists():
            error = f"Image not found: {image_path}"
            return self._prediction_record(
                image_record=image_record,
                image_sha256=None,
                raw_response=None,
                parsed_response=None,
                error=error,
                content_warning=None,
                inference_seconds=None,
            )

        image_sha256 = _sha256_path(image_path)
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                attempt_started = time.perf_counter()
                raw_response = self.model.classify(
                    image_path=image_path,
                    prompt=self.prompt,
                    sample_id=str(image_record["sample_id"]),
                )
                inference_seconds = time.perf_counter() - attempt_started
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                is_transient = _is_transient_model_error(exc)
                if is_transient:
                    LOGGER.warning(
                        "Transient failure classifying %s on attempt %d/%d: %s",
                        image_record["sample_id"],
                        attempt,
                        attempts,
                        error,
                    )
                else:
                    LOGGER.warning(
                        "Non-transient failure classifying %s; not retrying: %s",
                        image_record["sample_id"],
                        error,
                    )
                if _is_fatal_model_error(error):
                    LOGGER.error(
                        "Fatal CUDA error detected. Retries within the same process "
                        "do not recover the CUDA context; stopping attempts for %s.",
                        image_record["sample_id"],
                    )
                    break
                if not is_transient:
                    break
                if attempt < attempts and self.config.retry_delay > 0:
                    time.sleep(self.config.retry_delay)
                continue

            error = None
            try:
                parsed_response = parse_model_response(
                    raw_response,
                    task=self.config.task,
                    binary_threshold=self.config.decision_threshold,
                )
                content_warning = parsed_response.get("content_warning")
            except Exception as exc:
                content_warning = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Invalid content response for %s; not resampling: %s",
                    image_record["sample_id"],
                    content_warning,
                )
            break

        return self._prediction_record(
            image_record=image_record,
            image_sha256=image_sha256,
            raw_response=raw_response,
            parsed_response=parsed_response,
            error=error,
            content_warning=content_warning,
            inference_seconds=inference_seconds,
        )

    def _prediction_record(
        self,
        *,
        image_record: dict[str, Any],
        image_sha256: str | None,
        raw_response: str | None,
        parsed_response: dict[str, Any] | None,
        error: str | None,
        content_warning: str | None,
        inference_seconds: float | None = None,
    ) -> dict[str, Any]:
        predicted_label = None
        confidence = None
        source_true_label = image_record["true_label"]
        true_label = map_label_for_task(source_true_label, task=self.config.task)
        if parsed_response is not None:
            predicted_label = parsed_response["label"]
            confidence = parsed_response["confidence"]

        provenance = {
            "manifest": {
                "path": self._display_path(self.manifest_path),
                "sha256": self._manifest_sha256,
            },
            "images": (
                [{"image_path": image_record["image_path"], "sha256": image_sha256}]
                if image_sha256 is not None
                else []
            ),
            "prompt": {
                "sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "source": self.prompt_source,
            },
            "model": self.model.provenance(),
            "classifier": {
                "task": self.config.task,
                "decision_threshold": self.config.decision_threshold,
                "max_retries": self.config.max_retries,
            },
            "git": self._git_provenance,
        }
        return {
            "sample_id": image_record["sample_id"],
            "source_sample_id": image_record.get("source_sample_id"),
            "file_group_id": image_record.get("file_group_id"),
            "window_index": image_record.get("window_index"),
            "window_count": image_record.get("window_count"),
            "window_start_seconds": image_record.get("window_start_seconds"),
            "window_stop_seconds": image_record.get("window_stop_seconds"),
            "image_path": image_record["image_path"],
            "true_label": true_label,
            "source_true_label": source_true_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "raw_model_response": raw_response,
            "parsed_response": parsed_response,
            "error": error,
            "content_warning": content_warning,
            "inference_seconds": inference_seconds,
            "provenance": provenance,
        }

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def _resumable_prefix(self, image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The predictions already saved, if they are the start of this same run."""
        records: list[dict[str, Any]] = []
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # only the last line can be truncated
        expected = [record["sample_id"] for record in image_records[: len(records)]]
        if [record.get("sample_id") for record in records] != expected:
            raise ValueError(
                f"{self.output_path}: the saved predictions are not the start of the manifest."
            )
        current = {
            "manifest": {"path": self._display_path(self.manifest_path), "sha256": self._manifest_sha256},
            "prompt": {
                "sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "source": self.prompt_source,
            },
            "model": self.model.provenance(),
            "classifier": {
                "task": self.config.task,
                "decision_threshold": self.config.decision_threshold,
                "max_retries": self.config.max_retries,
            },
        }
        checked: set[str] = set()
        for record in records:
            provenance = record.get("provenance") or {}
            for section in _RESUME_SECTIONS:
                if provenance.get(section) != current[section]:
                    raise ValueError(
                        f"{self.output_path}: section {section!r} differs from the current run; "
                        "cannot resume."
                    )
            git = provenance.get("git") or {}
            if git != self._git_provenance and git.get("commit") not in checked:
                self._check_same_inference_code(git)
                checked.add(git["commit"])
        return records

    def _check_same_inference_code(self, previous: dict[str, Any]) -> None:
        current = self._git_provenance
        if (
            not previous.get("commit")
            or not current.get("commit")
            or previous.get("tracked_worktree_dirty") is not False
            or current.get("tracked_worktree_dirty") is not False
        ):
            raise ValueError("Resuming under another commit requires both commits to be clean and known.")
        diff = subprocess.run(
            ["git", "diff", "--quiet", previous["commit"], current["commit"], "--", *INFERENCE_PATHS],
            cwd=self.project_root,
            capture_output=True,
        )
        if diff.returncode != 0:
            raise ValueError(
                f"The inference code changed between {previous['commit'][:7]} and "
                f"{current['commit'][:7]} (or git failed); cannot resume."
            )

    def _rewrite(self, records: list[dict[str, Any]]) -> None:
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.output_path)

    def _reset_outputs(self) -> None:
        if self.output_path.exists():
            self.output_path.unlink()
            LOGGER.info("Old artifact removed: %s", self.output_path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_git_provenance(project_root: Path) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "commit": None,
        "tracked_worktree_dirty": None,
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                ".",
                f":(exclude){RUNTIME_LOG}",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return provenance
    provenance["commit"] = commit.stdout.strip() or None
    provenance["tracked_worktree_dirty"] = bool(status.stdout.strip())
    return provenance


def _is_transient_model_error(exc: Exception) -> bool:
    if isinstance(exc, TRANSIENT_MODEL_EXCEPTIONS):
        return True
    normalized = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in normalized for marker in TRANSIENT_MODEL_ERROR_MARKERS)


def _is_fatal_model_error(error: str) -> bool:
    normalized = error.lower()
    return any(marker in normalized for marker in FATAL_MODEL_ERROR_MARKERS)
