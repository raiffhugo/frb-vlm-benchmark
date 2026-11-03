from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_classifier.parser import TASK_FRB_BINARY, normalize_label


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowAggregationConfig:
    predictions_path: Path
    output_path: Path
    threshold: float = 0.5
    allow_partial: bool = False

    def validate(self) -> None:
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold deve ser finito e estar entre 0 e 1.")
        if not isinstance(self.allow_partial, bool):
            raise ValueError("allow_partial deve ser booleano.")


class WindowPredictionAggregator:
    def __init__(
        self,
        *,
        config: WindowAggregationConfig,
        project_root: Path,
    ) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.predictions_path = self._resolve(config.predictions_path)
        self.output_path = self._resolve(config.output_path)
        if self.predictions_path.resolve() == self.output_path.resolve():
            raise ValueError("predictions_path e output_path devem ser diferentes.")
        self._predictions_sha256: str | None = None

    def aggregate(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Arquivo agregado ja existe: {self.output_path}. Use --overwrite."
            )

        sha256_before = _sha256_path(self.predictions_path)
        records = self._load_predictions()
        sha256_after = _sha256_path(self.predictions_path)
        if sha256_after != sha256_before:
            raise RuntimeError(
                "O arquivo de predicoes mudou durante a leitura; execute novamente."
            )
        self._predictions_sha256 = sha256_before
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["file_group_id"], []).append(record)

        aggregated = [
            self._aggregate_group(file_group_id, grouped[file_group_id])
            for file_group_id in sorted(grouped)
        ]
        self._write_atomic(aggregated)
        LOGGER.info(
            "Predicoes de %d janelas agregadas em %d arquivos: %s",
            len(records),
            len(aggregated),
            self.output_path,
        )
        return aggregated

    def _load_predictions(self) -> list[dict[str, Any]]:
        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Arquivo de predicoes nao encontrado: {self.predictions_path}"
            )

        records: list[dict[str, Any]] = []
        expected_provenance: str | None = None
        expected_provenance_line: int | None = None
        with self.predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Linha invalida em {self.predictions_path}:{line_number}"
                    ) from exc
                self._validate_record(record, line_number)
                provenance = _canonical_run_provenance(record)
                if expected_provenance is None:
                    expected_provenance = provenance
                    expected_provenance_line = line_number
                elif provenance != expected_provenance:
                    raise ValueError(
                        f"Linha {line_number} possui proveniencia de execucao "
                        f"inconsistente com a linha {expected_provenance_line}."
                    )
                records.append(record)

        if not records:
            raise ValueError(f"Nenhuma predicao encontrada em {self.predictions_path}")
        _check_git_chain(records)
        return records

    @staticmethod
    def _validate_record(record: Any, line_number: int) -> None:
        if not isinstance(record, dict):
            raise ValueError(f"Linha {line_number} deve conter um objeto JSON.")
        required = {
            "file_group_id",
            "sample_id",
            "true_label",
            "window_index",
            "window_count",
            "provenance",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Linha {line_number} sem chaves obrigatorias: {', '.join(missing)}"
            )
        file_group_id = record.get("file_group_id")
        sample_id = record.get("sample_id")
        if not isinstance(file_group_id, str) or not file_group_id.strip():
            raise ValueError(
                f"Linha {line_number}: file_group_id deve ser uma string nao vazia."
            )
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"Linha {line_number}: sample_id deve ser uma string nao vazia."
            )
        window_index = record.get("window_index")
        if (
            isinstance(window_index, bool)
            or not isinstance(window_index, int)
            or window_index < 0
        ):
            raise ValueError(
                f"Linha {line_number}: window_index deve ser inteiro >= 0."
            )
        window_count = record.get("window_count")
        if (
            isinstance(window_count, bool)
            or not isinstance(window_count, int)
            or window_count <= 0
        ):
            raise ValueError(
                f"Linha {line_number}: window_count deve ser inteiro > 0."
            )
        if window_index >= window_count:
            raise ValueError(
                f"Linha {line_number}: window_index {window_index} deve ser menor "
                f"que window_count {window_count}."
            )
        _validate_run_provenance(record.get("provenance"), line_number)
        try:
            normalize_label(record.get("true_label"), task=TASK_FRB_BINARY)
        except ValueError as exc:
            raise ValueError(
                f"Linha {line_number} com true_label invalido: "
                f"{record.get('true_label')!r}."
            ) from exc

    def _aggregate_group(
        self,
        file_group_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ordered = sorted(records, key=_window_order_key)
        expected_window_count = _validate_group_windows(file_group_id, ordered)
        labels = {
            normalize_label(record["true_label"], task=TASK_FRB_BINARY)
            for record in ordered
        }
        if len(labels) != 1:
            raise ValueError(
                f"Grupo {file_group_id!r} possui true_label inconsistente: "
                + ", ".join(sorted(labels))
                + "."
            )
        true_label = next(iter(labels))

        valid: list[tuple[dict[str, Any], float]] = []
        invalid: list[dict[str, str]] = []
        for record in ordered:
            probability, reason = window_probability(record)
            if reason is not None:
                invalid.append(
                    {
                        "sample_id": record["sample_id"],
                        "reason": reason,
                    }
                )
                continue
            assert probability is not None
            valid.append((record, probability))

        metadata: dict[str, Any] = {
            "method": "max",
            "threshold": self.config.threshold,
            "total_window_count": len(ordered),
            "expected_window_count": expected_window_count,
            "valid_window_count": len(valid),
            "invalid_window_count": len(invalid),
            "winning_sample_id": None,
            "winning_window_index": None,
            "winning_window_start_seconds": None,
            "window_probabilities": [
                probability for _, probability in valid
            ],
            "window_predictions": [
                {
                    "sample_id": record["sample_id"],
                    "frb_probability": probability,
                }
                for record, probability in valid
            ],
            "invalid_windows": invalid,
        }

        base = {
            "sample_id": file_group_id,
            "source_sample_id": ordered[0].get("source_sample_id"),
            "file_group_id": file_group_id,
            "image_path": ordered[0].get("image_path"),
            "true_label": true_label,
            "source_true_label": ordered[0].get("source_true_label", true_label),
            "predicted_label": None,
            "confidence": None,
            "raw_model_response": None,
            "parsed_response": None,
            "error": None,
            "content_warning": None,
            "window_aggregation": metadata,
            "provenance": {
                "window_predictions": {
                    "path": self._display_path(self.predictions_path),
                    "sha256": self._predictions_sha256,
                },
                "aggregation": {
                    "method": "max",
                    "threshold": self.config.threshold,
                    "allow_partial": self.config.allow_partial,
                },
                "source_run": _run_provenance(ordered[0]),
            },
        }
        if invalid and not self.config.allow_partial:
            base["error"] = "partial_window_failure"
            return base
        if not valid:
            base["error"] = "no_valid_window_prediction"
            return base

        max_probability = max(probability for _, probability in valid)
        winning_record = min(
            (
                record
                for record, probability in valid
                if probability == max_probability
            ),
            key=_window_order_key,
        )
        predicted_label = (
            "FRB" if max_probability >= self.config.threshold else "NON_FRB"
        )
        window_predictions = [
            {
                "sample_id": record["sample_id"],
                "frb_probability": probability,
            }
            for record, probability in valid
        ]
        metadata.update(
            {
                "winning_sample_id": winning_record["sample_id"],
                "winning_window_index": winning_record.get("window_index"),
                "winning_window_start_seconds": winning_record.get(
                    "window_start_seconds"
                ),
                "window_probabilities": [
                    item["frb_probability"] for item in window_predictions
                ],
                "window_predictions": window_predictions,
            }
        )
        confidence = (
            max_probability
            if predicted_label == "FRB"
            else 1.0 - max_probability
        )
        parsed_response = {
            "label": predicted_label,
            "confidence": confidence,
            "frb_probability": max_probability,
            "decision_threshold": self.config.threshold,
            "aggregation": "max",
            "total_window_count": len(ordered),
            "valid_window_count": len(valid),
            "invalid_window_count": len(invalid),
            "winning_sample_id": winning_record["sample_id"],
            "window_probabilities": metadata["window_probabilities"],
        }
        base.update(
            {
                "image_path": winning_record.get("image_path"),
                "predicted_label": predicted_label,
                "confidence": confidence,
                "raw_model_response": winning_record.get("raw_model_response"),
                "parsed_response": parsed_response,
            }
        )
        return base

    def _write_atomic(self, records: list[dict[str, Any]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.output_path.parent,
            prefix=f".{self.output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

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


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_probability(
    record: dict[str, Any],
) -> tuple[float | None, str | None]:
    if record.get("error"):
        return None, f"window_error: {record['error']}"

    parsed_response = record.get("parsed_response")
    if not isinstance(parsed_response, dict):
        return None, "invalid_parsed_response"

    value = parsed_response.get("frb_probability")
    if isinstance(value, bool):
        return None, "invalid_frb_probability"
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None, "invalid_frb_probability"
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None, "invalid_frb_probability"
    return probability, None


def _window_order_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("sample_id") or ""),
        json.dumps(record, sort_keys=True, separators=(",", ":")),
    )


def _validate_group_windows(
    file_group_id: str,
    records: list[dict[str, Any]],
) -> int:
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Grupo {file_group_id!r} possui sample_id duplicado.")

    window_indices = [int(record["window_index"]) for record in records]
    if len(set(window_indices)) != len(window_indices):
        raise ValueError(f"Grupo {file_group_id!r} possui window_index duplicado.")

    source_sample_ids = {
        json.dumps(
            record.get("source_sample_id"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    }
    if len(source_sample_ids) != 1:
        raise ValueError(
            f"Grupo {file_group_id!r} possui source_sample_id inconsistente."
        )

    source_true_labels = {
        json.dumps(
            record.get("source_true_label"),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    }
    if len(source_true_labels) != 1:
        raise ValueError(
            f"Grupo {file_group_id!r} possui source_true_label inconsistente."
        )

    window_counts = {int(record["window_count"]) for record in records}
    if len(window_counts) != 1:
        raise ValueError(
            f"Grupo {file_group_id!r} possui window_count inconsistente: "
            + ", ".join(str(value) for value in sorted(window_counts))
            + "."
        )
    expected_count = next(iter(window_counts))
    expected_indices = set(range(expected_count))
    actual_indices = set(window_indices)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"Grupo {file_group_id!r} possui cobertura incompleta de janelas: "
            f"ausentes={missing}, inesperadas={unexpected}."
        )
    return expected_count


def _validate_run_provenance(value: Any, line_number: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Linha {line_number}: provenance deve ser um objeto.")

    required_sections = ("manifest", "prompt", "model", "classifier", "git")
    for section in required_sections:
        if not isinstance(value.get(section), dict):
            raise ValueError(
                f"Linha {line_number}: provenance.{section} deve ser um objeto."
            )

    classifier = value["classifier"]
    if classifier.get("task") != TASK_FRB_BINARY:
        raise ValueError(
            f"Linha {line_number}: provenance.classifier.task deve ser "
            f"{TASK_FRB_BINARY!r}."
        )

    manifest = value["manifest"]
    _validate_sha256(manifest.get("sha256"), line_number, "manifest.sha256")

    prompt = value["prompt"]
    _validate_sha256(prompt.get("sha256"), line_number, "prompt.sha256")

    model = value["model"]
    for field in ("model_id", "model_revision"):
        if not isinstance(model.get(field), str) or not model[field].strip():
            raise ValueError(
                f"Linha {line_number}: provenance.model.{field} deve ser uma "
                "string nao vazia."
            )
    if not isinstance(model.get("generation"), dict):
        raise ValueError(
            f"Linha {line_number}: provenance.model.generation deve ser um objeto."
        )

    git = value["git"]
    if not isinstance(git.get("commit"), str) or not git["commit"].strip():
        raise ValueError(
            f"Linha {line_number}: provenance.git.commit deve ser uma string nao vazia."
        )
    if not isinstance(git.get("tracked_worktree_dirty"), bool):
        raise ValueError(
            f"Linha {line_number}: provenance.git.tracked_worktree_dirty deve ser "
            "booleano."
        )


def _validate_sha256(value: Any, line_number: int, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(
            f"Linha {line_number}: provenance.{field} deve ser um SHA-256 hexadecimal."
        )


def _run_provenance(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record["provenance"]
    return {
        section: provenance[section]
        for section in ("manifest", "prompt", "model", "classifier", "git")
    }


def _canonical_run_provenance(record: dict[str, Any]) -> str:
    """As secoes que definem a rodada, menos o commit: uma rodada retomada (classify
    --resume) pode mudar de commit, o que _check_git_chain confere a parte."""
    run = _run_provenance(record)
    run.pop("git")
    return json.dumps(run, sort_keys=True, separators=(",", ":"))


def _check_git_chain(records: list[dict[str, Any]]) -> None:
    """O commit so muda onde a predicao registra a retomada a partir do commit anterior."""
    current = records[0]["provenance"]["git"]
    for position, record in enumerate(records[1:], start=2):
        git = record["provenance"]["git"]
        if git == current:
            continue
        resume = record["provenance"].get("resume") or {}
        if resume.get("previous_commit") != current.get("commit"):
            raise ValueError(
                f"Predicao {position} possui proveniencia de execucao inconsistente: "
                "o commit muda sem uma retomada registrada."
            )
        current = git


__all__ = ["WindowAggregationConfig", "WindowPredictionAggregator"]
