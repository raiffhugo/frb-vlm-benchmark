from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BinaryBenchmarkSubsetConfig:
    source_labels: Path
    source_image_manifest: Path
    output_dir: Path
    n_frb: int = 1000
    per_rfi_subtype: int = 100
    n_noise: int = 500
    seed: int = 42
    source_labels_csv: Path | None = None


class BinaryBenchmarkSubsetSelector:
    """Derives binary benchmark manifests from the multiclass dataset."""

    def __init__(
        self, *, config: BinaryBenchmarkSubsetConfig, project_root: Path
    ) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.source_labels = self._resolve(config.source_labels)
        self.source_image_manifest = self._resolve(config.source_image_manifest)
        self.source_labels_csv = (
            self._resolve(config.source_labels_csv)
            if config.source_labels_csv is not None
            else None
        )
        self.output_dir = self._resolve(config.output_dir)
        self.metadata_dir = self.output_dir / "metadata"
        self.labels_path = self.metadata_dir / "labels.jsonl"
        self.labels_csv_path = self.metadata_dir / "labels.csv"
        self.manifest_path = self.metadata_dir / "image_manifest.jsonl"

    def select(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        label_lines = self._load_label_lines()
        manifest_by_source = self._load_manifest_by_source()
        chosen = self._choose(label_lines)
        self._check_outputs(overwrite=overwrite)

        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        composition: Counter[str] = Counter()
        with self.labels_path.open("w", encoding="utf-8") as labels_out, \
                self.manifest_path.open("w", encoding="utf-8") as manifest_out:
            index = 0
            for raw_line, record in label_lines:
                source_id = record["sample_id"]
                if source_id not in chosen:
                    continue
                manifest_record = manifest_by_source.get(source_id)
                if manifest_record is None:
                    raise ValueError(
                        f"Sample {source_id} has no image in the source manifest; "
                        "run the plot stage on the source dataset before selecting."
                    )
                labels_out.write(raw_line)
                derived = dict(manifest_record)
                derived["source_image_sample_id"] = manifest_record["sample_id"]
                derived["sample_id"] = f"sample_{index:06d}"
                manifest_out.write(json.dumps(derived, sort_keys=True) + "\n")
                records.append(derived)
                composition[self._composition_key(record)] += 1
                index += 1

        self._write_labels_csv(chosen)

        LOGGER.info(
            "Binary benchmark derived in %s: %d samples (seed=%d).",
            self.output_dir,
            len(records),
            self.config.seed,
        )
        for key in sorted(composition):
            LOGGER.info("  %s: %d", key, composition[key])
        return records

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_label_lines(self) -> list[tuple[str, dict[str, Any]]]:
        if not self.source_labels.exists():
            raise FileNotFoundError(
                f"Source labels.jsonl not found: {self.source_labels}"
            )
        lines: list[tuple[str, dict[str, Any]]] = []
        with self.source_labels.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.source_labels}:{line_number}"
                    ) from exc
                lines.append((line, record))
        if not lines:
            raise ValueError(f"Source labels.jsonl is empty: {self.source_labels}")
        return lines

    def _load_manifest_by_source(self) -> dict[str, dict[str, Any]]:
        if not self.source_image_manifest.exists():
            raise FileNotFoundError(
                "Source image manifest not found: "
                f"{self.source_image_manifest}"
            )
        by_source: dict[str, dict[str, Any]] = {}
        with self.source_image_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.source_image_manifest}:{line_number}"
                    ) from exc
                source_id = record.get("source_sample_id")
                if not source_id:
                    raise ValueError(
                        f"Line {line_number} of the source manifest has no "
                        "source_sample_id."
                    )
                by_source[source_id] = record
        return by_source

    # ------------------------------------------------------------------ #
    # Deterministic selection
    # ------------------------------------------------------------------ #
    def _choose(self, label_lines: list[tuple[str, dict[str, Any]]]) -> set[str]:
        pools: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        for _, record in label_lines:
            label = record.get("label")
            if label == "RFI":
                subtype = (record.get("parameters") or {}).get("rfi_category")
                pools[("RFI", subtype)].append(record["sample_id"])
            else:
                pools[(label, None)].append(record["sample_id"])

        rng = random.Random(self.config.seed)
        chosen: list[str] = []

        rfi_subtypes = sorted(
            key[1] for key in pools if key[0] == "RFI" and key[1] is not None
        )
        for subtype in rfi_subtypes:
            pool = sorted(pools[("RFI", subtype)])
            if len(pool) < self.config.per_rfi_subtype:
                raise ValueError(
                    f"RFI subtype {subtype} has {len(pool)} samples; "
                    f"cannot select {self.config.per_rfi_subtype}."
                )
            chosen += rng.sample(pool, self.config.per_rfi_subtype)

        noise_pool = sorted(pools[("NOISE", None)])
        if len(noise_pool) < self.config.n_noise:
            raise ValueError(
                f"NOISE has {len(noise_pool)} samples; cannot select "
                f"{self.config.n_noise}."
            )
        chosen += rng.sample(noise_pool, self.config.n_noise)

        frb_pool = sorted(pools[("FRB", None)])
        if len(frb_pool) < self.config.n_frb:
            raise ValueError(
                f"FRB has {len(frb_pool)} samples; cannot select "
                f"{self.config.n_frb}."
            )
        chosen += rng.sample(frb_pool, self.config.n_frb)

        return set(chosen)

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def _check_outputs(self, *, overwrite: bool) -> None:
        outputs = [self.labels_path, self.labels_csv_path, self.manifest_path]
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Outputs already exist in {self.metadata_dir}. Use --overwrite."
            )
        for path in existing:
            path.unlink()

    def _write_labels_csv(self, chosen: set[str]) -> None:
        if self.source_labels_csv is None or not self.source_labels_csv.exists():
            LOGGER.info("Source labels.csv missing; derived csv not generated.")
            return
        with self.source_labels_csv.open("r", encoding="utf-8", newline="") as src, \
                self.labels_csv_path.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
                raise ValueError(
                    f"Source labels.csv has no sample_id column: "
                    f"{self.source_labels_csv}"
                )
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["sample_id"] in chosen:
                    writer.writerow(row)

    @staticmethod
    def _composition_key(record: dict[str, Any]) -> str:
        label = record.get("label")
        if label == "RFI":
            subtype = (record.get("parameters") or {}).get("rfi_category")
            return f"RFI/{subtype}"
        return str(label)

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path
