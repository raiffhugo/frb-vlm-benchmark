from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vlm_classifier.parser import TASK_FRB_BINARY, map_label_for_task


LOGGER = logging.getLogger(__name__)
AGGREGATION_METHODS = ("max", "mean", "median")


@dataclass(frozen=True)
class ExternalPredictionImportConfig:
    benchmark_manifest: Path
    external_predictions: Path
    output_path: Path
    threshold: float = 0.5
    aggregation: str = "max"

    def validate(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if self.aggregation not in AGGREGATION_METHODS:
            raise ValueError(
                "aggregation must be one of: "
                + ", ".join(AGGREGATION_METHODS)
                + "."
            )


class ExternalPredictionImporter:
    def __init__(
        self,
        *,
        config: ExternalPredictionImportConfig,
        project_root: Path,
    ) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.benchmark_manifest = self._resolve(config.benchmark_manifest)
        self.external_predictions = self._resolve(config.external_predictions)
        self.output_path = self._resolve(config.output_path)

    def import_predictions(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        manifest_records = self._load_benchmark_manifest()
        external_records = self._load_external_predictions()
        grouped = self._group_external_records(external_records)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.output_path.exists():
            self.output_path.unlink()
            LOGGER.info("Old artifact removed: %s", self.output_path)
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Predictions file already exists: {self.output_path}. Use --overwrite."
            )

        predictions: list[dict[str, Any]] = []
        with self.output_path.open("w", encoding="utf-8") as handle:
            for manifest_record in tqdm(
                manifest_records,
                desc="Importing external predictions",
                unit="sample",
            ):
                prediction = self._prediction_from_manifest(manifest_record, grouped)
                predictions.append(prediction)
                handle.write(json.dumps(prediction, sort_keys=True) + "\n")

        LOGGER.info("External predictions imported to %s", self.output_path)
        return predictions

    def _load_benchmark_manifest(self) -> list[dict[str, Any]]:
        if not self.benchmark_manifest.exists():
            raise FileNotFoundError(
                f"Benchmark manifest not found: {self.benchmark_manifest}"
            )

        records: list[dict[str, Any]] = []
        with self.benchmark_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.benchmark_manifest}:{line_number}"
                    ) from exc
                self._validate_manifest_record(record, line_number)
                records.append(record)

        if not records:
            raise ValueError(f"Benchmark manifest is empty: {self.benchmark_manifest}")
        return records

    def _validate_manifest_record(self, record: dict[str, Any], line_number: int) -> None:
        required = {"sample_id", "true_label"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Line {line_number} is missing required keys: {', '.join(missing)}"
            )

    def _load_external_predictions(self) -> list[dict[str, Any]]:
        if not self.external_predictions.exists():
            raise FileNotFoundError(
                f"External predictions not found: {self.external_predictions}"
            )

        text = self.external_predictions.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"External predictions file is empty: {self.external_predictions}")

        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("External JSON file must contain a list.")
            records = parsed
        else:
            records = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.external_predictions}:{line_number}"
                    ) from exc

        clean_records: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"External prediction {index} is not a JSON object.")
            if "id" not in record or "prob" not in record:
                raise ValueError(f"External prediction {index} has no id/prob.")
            clean_records.append(record)

        return clean_records

    def _group_external_records(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            sample_id = normalize_external_sample_id(record["id"])
            grouped.setdefault(sample_id, []).append(record)
        return grouped

    def _prediction_from_manifest(
        self,
        manifest_record: dict[str, Any],
        grouped: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        sample_id = normalize_external_sample_id(manifest_record["sample_id"])
        source_true_label = manifest_record["true_label"]
        true_label = map_label_for_task(source_true_label, task=TASK_FRB_BINARY)
        raw_records = grouped.get(sample_id, [])

        if not raw_records:
            return {
                "sample_id": sample_id,
                "image_path": None,
                "true_label": true_label,
                "source_true_label": source_true_label,
                "predicted_label": None,
                "confidence": None,
                "raw_model_response": [],
                "parsed_response": None,
                "error": "missing_external_prediction",
            }

        probabilities = [_parse_probability(record.get("prob")) for record in raw_records]
        frb_probability = aggregate_probabilities(
            probabilities,
            method=self.config.aggregation,
        )
        predicted_label = "FRB" if frb_probability >= self.config.threshold else "NON_FRB"
        original_labels = [_parse_original_label(record.get("label")) for record in raw_records]
        disagreement_count = sum(
            1
            for probability, original_label in zip(probabilities, original_labels)
            if original_label is not None
            and original_label != (probability >= self.config.threshold)
        )
        parsed_response = {
            "label": predicted_label,
            "confidence": frb_probability,
            "frb_probability": frb_probability,
            "threshold": self.config.threshold,
            "aggregation": self.config.aggregation,
            "external_detection_count": len(raw_records),
            "n_candidates": len(raw_records),
            "candidate_probabilities": probabilities,
            "original_labels": original_labels,
            "original_threshold_disagreement_count": disagreement_count,
            "original_threshold_disagreement": disagreement_count > 0,
        }
        return {
            "sample_id": sample_id,
            "image_path": None,
            "true_label": true_label,
            "source_true_label": source_true_label,
            "predicted_label": predicted_label,
            "confidence": frb_probability,
            "raw_model_response": raw_records,
            "parsed_response": parsed_response,
            "error": None,
        }

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path


def normalize_external_sample_id(value: Any) -> str:
    name = Path(str(value)).name
    if name.lower().endswith(".fits"):
        name = name[:-5]
    return name


def _parse_probability(value: Any) -> float:
    probability = float(value)
    return min(1.0, max(0.0, probability))


def aggregate_probabilities(probabilities: list[float], *, method: str = "max") -> float:
    if not probabilities:
        raise ValueError("There are no probabilities to aggregate.")
    if method == "max":
        return max(probabilities)
    if method == "mean":
        return float(statistics.fmean(probabilities))
    if method == "median":
        return float(statistics.median(probabilities))
    raise ValueError(
        "Invalid aggregation method: "
        f"{method!r}. Use one of: {', '.join(AGGREGATION_METHODS)}."
    )


def _parse_original_label(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    normalized = str(value).strip().upper()
    if normalized in {"1", "TRUE", "FRB", "POSITIVE", "CANDIDATE"}:
        return True
    if normalized in {"0", "FALSE", "NON_FRB", "NON-FRB", "NOISE", "RFI", "NEGATIVE"}:
        return False
    return None
