from __future__ import annotations

import json
import logging
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


@dataclass(frozen=True)
class ClassifierConfig:
    manifest_path: Path
    output_path: Path
    max_retries: int = 2
    retry_delay: float = 0.5
    task: str = TASK_MULTICLASS
    decision_threshold: float = 0.5

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
    ) -> None:
        config.validate()
        self.config = ClassifierConfig(
            manifest_path=config.manifest_path,
            output_path=config.output_path,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            task=normalize_task(config.task),
            decision_threshold=config.decision_threshold,
        )
        self.model = model
        self.project_root = project_root.resolve()
        self.manifest_path = self._resolve(config.manifest_path)
        self.output_path = self._resolve(config.output_path)

    def run(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        image_records = self._load_manifest()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self._reset_outputs()
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Predictions file already exists: {self.output_path}. Use --overwrite."
            )

        predictions: list[dict[str, Any]] = []
        with self.output_path.open("w", encoding="utf-8") as handle:
            for image_record in tqdm(image_records, desc="Classifying images", unit="image"):
                prediction = self._classify_one(image_record)
                predictions.append(prediction)
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")
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

        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
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
                raw_response=None,
                parsed_response=None,
                error=error,
                content_warning=None,
                inference_seconds=None,
            )

        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                attempt_started = time.perf_counter()
                raw_response = self.model.classify(
                    image_path=image_path,
                    prompt=prompt_for_task(self.config.task),
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

        return {
            "sample_id": image_record["sample_id"],
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
        }

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def _reset_outputs(self) -> None:
        if self.output_path.exists():
            self.output_path.unlink()
            LOGGER.info("Old artifact removed: %s", self.output_path)


def _is_transient_model_error(exc: Exception) -> bool:
    if isinstance(exc, TRANSIENT_MODEL_EXCEPTIONS):
        return True
    normalized = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in normalized for marker in TRANSIENT_MODEL_ERROR_MARKERS)


def _is_fatal_model_error(error: str) -> bool:
    normalized = error.lower()
    return any(marker in normalized for marker in FATAL_MODEL_ERROR_MARKERS)
