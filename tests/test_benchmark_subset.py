from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulate_dataset.benchmark_subset import (
    BinaryBenchmarkSubsetConfig,
    BinaryBenchmarkSubsetSelector,
)


def _label_record(sample_id: str, label: str, subtype: str | None = None) -> dict:
    parameters: dict = {"flux_density": 1.0}
    if subtype is not None:
        parameters["rfi_category"] = subtype
    return {
        "sample_id": sample_id,
        "label": label,
        "parameters": parameters,
        "fits_path": f"dataset/fits/{label.lower()}/{sample_id}.fits",
        "seed": 123,
    }


class BinaryBenchmarkSubsetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        metadata = self.root / "dataset" / "metadata"
        metadata.mkdir(parents=True)

        self.labels = []
        for i in range(4):
            self.labels.append(_label_record(f"frb_{i:05d}", "FRB"))
        for i in range(4):
            subtype = "narrowband" if i < 2 else "wifi"
            self.labels.append(_label_record(f"rfi_{i:05d}", "RFI", subtype))
        for i in range(4):
            self.labels.append(_label_record(f"noise_{i:05d}", "NOISE"))

        with (metadata / "labels.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.labels:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        with (metadata / "labels.csv").open("w", encoding="utf-8") as handle:
            handle.write("sample_id,label\n")
            for record in self.labels:
                handle.write(f"{record['sample_id']},{record['label']}\n")

        with (metadata / "image_manifest.jsonl").open("w", encoding="utf-8") as handle:
            for index, record in enumerate(self.labels):
                row = {
                    "sample_id": f"sample_{index:06d}",
                    "source_sample_id": record["sample_id"],
                    "true_label": record["label"],
                    "image_path": f"dataset/images/samples/sample_{index:06d}.png",
                    "fits_path": record["fits_path"],
                    "simulation_parameters": record["parameters"],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _selector(self) -> BinaryBenchmarkSubsetSelector:
        return BinaryBenchmarkSubsetSelector(
            config=BinaryBenchmarkSubsetConfig(
                source_labels=Path("dataset/metadata/labels.jsonl"),
                source_image_manifest=Path("dataset/metadata/image_manifest.jsonl"),
                source_labels_csv=Path("dataset/metadata/labels.csv"),
                output_dir=Path("dataset_binary"),
                n_frb=4,
                per_rfi_subtype=2,
                n_noise=2,
                seed=7,
            ),
            project_root=self.root,
        )

    def test_selects_balanced_subset_with_aligned_ids(self) -> None:
        records = self._selector().select(overwrite=False)

        self.assertEqual(len(records), 10)
        derived_labels = [
            json.loads(line)
            for line in (self.root / "dataset_binary/metadata/labels.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        by_label: dict[str, int] = {}
        for record in derived_labels:
            by_label[record["label"]] = by_label.get(record["label"], 0) + 1
        self.assertEqual(by_label, {"FRB": 4, "RFI": 4, "NOISE": 2})

        subtypes = [
            record["parameters"]["rfi_category"]
            for record in derived_labels
            if record["label"] == "RFI"
        ]
        self.assertEqual(
            sorted(subtypes), ["narrowband", "narrowband", "wifi", "wifi"]
        )

        source_order = [record["sample_id"] for record in self.labels]
        derived_order = [record["sample_id"] for record in derived_labels]
        self.assertEqual(
            derived_order,
            [sid for sid in source_order if sid in set(derived_order)],
        )

        for index, record in enumerate(records):
            self.assertEqual(record["sample_id"], f"sample_{index:06d}")
            self.assertEqual(
                record["source_sample_id"], derived_labels[index]["sample_id"]
            )
            self.assertIn("source_image_sample_id", record)
            self.assertEqual(record["true_label"], derived_labels[index]["label"])

        csv_lines = (
            (self.root / "dataset_binary/metadata/labels.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertEqual(len(csv_lines), 1 + len(derived_labels))

    def test_is_deterministic_and_respects_overwrite(self) -> None:
        selector = self._selector()
        selector.select(overwrite=False)
        first = (self.root / "dataset_binary/metadata/image_manifest.jsonl").read_bytes()

        with self.assertRaises(FileExistsError):
            selector.select(overwrite=False)

        selector.select(overwrite=True)
        second = (self.root / "dataset_binary/metadata/image_manifest.jsonl").read_bytes()
        self.assertEqual(first, second)

    def test_rejects_insufficient_pool(self) -> None:
        selector = BinaryBenchmarkSubsetSelector(
            config=BinaryBenchmarkSubsetConfig(
                source_labels=Path("dataset/metadata/labels.jsonl"),
                source_image_manifest=Path("dataset/metadata/image_manifest.jsonl"),
                output_dir=Path("dataset_binary"),
                n_frb=4,
                per_rfi_subtype=3,
                n_noise=2,
                seed=7,
            ),
            project_root=self.root,
        )
        with self.assertRaises(ValueError):
            selector.select(overwrite=False)


if __name__ == "__main__":
    unittest.main()
