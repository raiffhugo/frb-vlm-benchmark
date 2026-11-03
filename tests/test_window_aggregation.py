from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluator import EvaluationConfig, PredictionEvaluator
from evaluation.window_aggregator import (
    WindowAggregationConfig,
    WindowPredictionAggregator,
)
from run_pipeline import build_parser


def _run_provenance() -> dict[str, object]:
    return {
        "manifest": {
            "path": "dataset/metadata/image_manifest.jsonl",
            "sha256": "a" * 64,
        },
        "prompt": {
            "sha256": "b" * 64,
        },
        "model": {
            "model_class": "vlm_classifier.models.Gemma4VLM",
            "model_id": "google/gemma-4-E2B-it",
            "model_revision": "c" * 40,
            "generation": {
                "max_new_tokens": 1024,
                "do_sample": False,
                "seed": None,
            },
        },
        "classifier": {
            "task": "frb-binary",
            "decision_threshold": 0.5,
            "max_retries": 2,
        },
        "git": {
            "commit": "d" * 40,
            "tracked_worktree_dirty": False,
        },
    }


def _prediction(
    file_group_id: str,
    sample_id: str,
    true_label: str,
    probability: float | None,
    *,
    error: str | None = None,
    window_count: int = 1,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    window_index = int(sample_id.rsplit("_", 1)[-1])
    return {
        "sample_id": sample_id,
        "source_sample_id": f"source_{file_group_id}",
        "file_group_id": file_group_id,
        "window_index": window_index,
        "window_count": window_count,
        "window_start_seconds": float(window_index),
        "image_path": f"dataset/{sample_id}.png",
        "true_label": true_label,
        "source_true_label": true_label,
        "predicted_label": None,
        "confidence": None,
        "raw_model_response": "{}",
        "parsed_response": (
            None
            if probability is None
            else {"label": "FRB", "frb_probability": probability}
        ),
        "error": error,
        "provenance": provenance if provenance is not None else _run_provenance(),
    }


class WindowPredictionAggregatorTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _aggregator(
        root: Path,
        *,
        threshold: float = 0.5,
        allow_partial: bool = False,
    ) -> WindowPredictionAggregator:
        return WindowPredictionAggregator(
            config=WindowAggregationConfig(
                predictions_path=Path("predictions.jsonl"),
                output_path=Path("aggregated.jsonl"),
                threshold=threshold,
                allow_partial=allow_partial,
            ),
            project_root=root,
        )

    def test_max_aggregation_is_deterministic_and_evaluator_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _prediction("group_b", "b_0", "RFI", 0.4),
                _prediction("group_a", "a_2", "FRB", 0.9, window_count=4),
                _prediction("group_a", "a_0", "FRB", 0.2, window_count=4),
                _prediction("group_a", "a_1", "FRB", 0.9, window_count=4),
                _prediction("group_a", "a_3", "FRB", 0.3, window_count=4),
            ]
            self._write_jsonl(root / "predictions.jsonl", rows)

            aggregated = self._aggregator(root).aggregate()

            self.assertEqual(
                [record["file_group_id"] for record in aggregated],
                ["group_a", "group_b"],
            )
            group_a = aggregated[0]
            self.assertEqual(group_a["sample_id"], "group_a")
            self.assertEqual(group_a["predicted_label"], "FRB")
            self.assertEqual(group_a["confidence"], 0.9)
            self.assertEqual(
                group_a["parsed_response"]["frb_probability"], 0.9
            )
            metadata = group_a["window_aggregation"]
            self.assertEqual(metadata["total_window_count"], 4)
            self.assertEqual(metadata["expected_window_count"], 4)
            self.assertEqual(metadata["valid_window_count"], 4)
            self.assertEqual(metadata["invalid_window_count"], 0)
            self.assertEqual(metadata["winning_sample_id"], "a_1")
            self.assertEqual(
                metadata["window_probabilities"], [0.2, 0.9, 0.9, 0.3]
            )
            self.assertEqual(aggregated[1]["true_label"], "NON_FRB")
            self.assertEqual(aggregated[1]["predicted_label"], "NON_FRB")
            self.assertEqual(aggregated[1]["confidence"], 0.6)
            self.assertEqual(
                aggregated[0]["provenance"]["window_predictions"]["sha256"],
                __import__("hashlib").sha256(
                    (root / "predictions.jsonl").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                aggregated[0]["provenance"]["source_run"],
                _run_provenance(),
            )

            evaluator = PredictionEvaluator(
                config=EvaluationConfig(
                    predictions_path=Path("aggregated.jsonl"),
                    output_dir=Path("evaluation"),
                    task="frb-binary",
                ),
                project_root=root,
            )
            evaluator.evaluate()
            metrics = json.loads(
                (root / "evaluation" / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["evaluated_count"], 2)
            self.assertEqual(metrics["accuracy"], 1.0)

    def test_inconsistent_true_labels_reject_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _prediction("same_file", "w_0", "FRB", 0.8, window_count=2),
                _prediction(
                    "same_file", "w_1", "NON_FRB", 0.1, window_count=2
                ),
            ]
            rows[1]["source_true_label"] = "FRB"
            self._write_jsonl(root / "predictions.jsonl", rows)

            with self.assertRaisesRegex(ValueError, "true_label inconsistente"):
                self._aggregator(root).aggregate()

            self.assertFalse((root / "aggregated.jsonl").exists())

    def test_strict_group_with_invalid_windows_emits_partial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _prediction("bad_file", "w_0", "NOISE", 0.8, window_count=2),
                _prediction("bad_file", "w_1", "NOISE", None, window_count=2),
            ]
            self._write_jsonl(root / "predictions.jsonl", rows)

            aggregated = self._aggregator(root).aggregate()

            self.assertEqual(len(aggregated), 1)
            record = aggregated[0]
            self.assertEqual(record["true_label"], "NON_FRB")
            self.assertIsNone(record["predicted_label"])
            self.assertIsNone(record["parsed_response"])
            self.assertEqual(record["error"], "partial_window_failure")
            metadata = record["window_aggregation"]
            self.assertEqual(metadata["valid_window_count"], 1)
            self.assertEqual(metadata["invalid_window_count"], 1)
            self.assertIsNone(metadata["winning_sample_id"])
            self.assertEqual(metadata["window_probabilities"], [0.8])

    def test_allow_partial_aggregates_only_present_valid_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                _prediction("group", "w_0", "FRB", 0.8, window_count=2),
                _prediction("group", "w_1", "FRB", None, window_count=2),
            ]
            self._write_jsonl(root / "predictions.jsonl", rows)

            aggregated = self._aggregator(root, allow_partial=True).aggregate()

            record = aggregated[0]
            self.assertIsNone(record["error"])
            self.assertEqual(record["predicted_label"], "FRB")
            self.assertEqual(record["confidence"], 0.8)
            metadata = record["window_aggregation"]
            self.assertEqual(metadata["valid_window_count"], 1)
            self.assertEqual(metadata["invalid_window_count"], 1)
            self.assertEqual(metadata["window_probabilities"], [0.8])
            self.assertTrue(
                record["provenance"]["aggregation"]["allow_partial"]
            )

    def test_confidence_follows_predicted_class_at_custom_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_jsonl(
                root / "predictions.jsonl",
                [_prediction("group", "w_0", "RFI", 0.7)],
            )

            aggregated = self._aggregator(root, threshold=0.8).aggregate()

            self.assertEqual(aggregated[0]["predicted_label"], "NON_FRB")
            self.assertAlmostEqual(aggregated[0]["confidence"], 0.3)

    def test_rejects_structural_window_contract_violations(self) -> None:
        missing = [
            _prediction("group", "w_0", "FRB", 0.8, window_count=2),
        ]
        duplicate_sample = [
            _prediction("group", "w_0", "FRB", 0.8, window_count=2),
            _prediction("group", "w_0", "FRB", 0.7, window_count=2),
        ]
        duplicate_index = [
            _prediction("group", "w_0", "FRB", 0.8, window_count=2),
            _prediction("group", "other_1", "FRB", 0.7, window_count=2),
        ]
        duplicate_index[1]["window_index"] = 0
        inconsistent_count = [
            _prediction("group", "w_0", "FRB", 0.8, window_count=2),
            _prediction("group", "w_1", "FRB", 0.7, window_count=3),
        ]
        inconsistent_source = [
            _prediction("group", "w_0", "RFI", 0.2, window_count=2),
            _prediction("group", "w_1", "RFI", 0.1, window_count=2),
        ]
        inconsistent_source[1]["source_sample_id"] = "another_source"
        inconsistent_source_label = [
            _prediction("group", "w_0", "RFI", 0.2, window_count=2),
            _prediction("group", "w_1", "RFI", 0.1, window_count=2),
        ]
        inconsistent_source_label[1]["source_true_label"] = "NOISE"
        cases = [
            ("missing", missing, "cobertura incompleta"),
            ("duplicate_sample", duplicate_sample, "sample_id duplicado"),
            ("duplicate_index", duplicate_index, "window_index duplicado"),
            ("count", inconsistent_count, "window_count inconsistente"),
            ("source", inconsistent_source, "source_sample_id inconsistente"),
            (
                "source_label",
                inconsistent_source_label,
                "source_true_label inconsistente",
            ),
        ]

        for name, rows, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self._write_jsonl(root / "predictions.jsonl", rows)
                with self.assertRaisesRegex(ValueError, message):
                    self._aggregator(root, allow_partial=True).aggregate()
                self.assertFalse((root / "aggregated.jsonl").exists())

    def test_rejects_invalid_window_fields_per_record(self) -> None:
        cases = [
            ("index", "window_index", True, "window_index"),
            ("count", "window_count", 0, "window_count"),
        ]
        for name, field, value, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                row = _prediction("group", "w_0", "FRB", 0.8)
                row[field] = value
                self._write_jsonl(root / "predictions.jsonl", [row])
                with self.assertRaisesRegex(ValueError, message):
                    self._aggregator(root).aggregate()

    def test_rejects_global_provenance_mismatch_across_disjoint_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other_provenance = copy.deepcopy(_run_provenance())
            other_model = other_provenance["model"]
            assert isinstance(other_model, dict)
            other_model["model_id"] = "google/gemma-4-E4B-it"
            rows = [
                _prediction("group_a", "a_0", "FRB", 0.8),
                _prediction(
                    "group_b",
                    "b_0",
                    "RFI",
                    0.2,
                    provenance=other_provenance,
                ),
            ]
            self._write_jsonl(root / "predictions.jsonl", rows)

            with self.assertRaisesRegex(ValueError, "proveniencia de execucao"):
                self._aggregator(root).aggregate()

    def test_commit_may_change_only_where_a_resume_is_recorded(self) -> None:
        first = _run_provenance()
        later = copy.deepcopy(first)
        later["git"] = {"commit": "e" * 40, "tracked_worktree_dirty": False}
        resumed = copy.deepcopy(later)
        resumed["resume"] = {"resumed_at_record": 1, "previous_commit": "d" * 40}
        for provenance, accepted in ((resumed, True), (later, False)):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                rows = [
                    _prediction("group", "w_0", "FRB", 0.8, window_count=2, provenance=first),
                    _prediction("group", "w_1", "FRB", 0.2, window_count=2, provenance=provenance),
                ]
                self._write_jsonl(root / "predictions.jsonl", rows)

                if accepted:
                    self.assertEqual(len(self._aggregator(root).aggregate()), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "retomada registrada"):
                        self._aggregator(root).aggregate()

    def test_rejects_missing_official_run_provenance(self) -> None:
        cases = [
            ("revision", "model", "model_revision", "model_revision"),
            ("commit", "git", "commit", "git.commit"),
            ("task", "classifier", "task", "classifier.task"),
        ]
        for name, section, field, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                provenance = copy.deepcopy(_run_provenance())
                section_data = provenance[section]
                assert isinstance(section_data, dict)
                section_data[field] = "multiclass" if field == "task" else ""
                row = _prediction(
                    "group", "w_0", "FRB", 0.8, provenance=provenance
                )
                self._write_jsonl(root / "predictions.jsonl", [row])

                with self.assertRaisesRegex(ValueError, message):
                    self._aggregator(root).aggregate()

    def test_rejects_same_input_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "devem ser diferentes"):
                WindowPredictionAggregator(
                    config=WindowAggregationConfig(
                        predictions_path=Path("same.jsonl"),
                        output_path=Path("same.jsonl"),
                    ),
                    project_root=root,
                )

    def test_overwrite_is_explicit_and_atomic_temp_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_jsonl(
                root / "predictions.jsonl",
                [_prediction("group", "w_0", "FRB", 0.8)],
            )
            output = root / "aggregated.jsonl"
            output.write_text("old\n", encoding="utf-8")
            aggregator = self._aggregator(root)

            with self.assertRaises(FileExistsError):
                aggregator.aggregate()
            self.assertEqual(output.read_text(encoding="utf-8"), "old\n")

            records = aggregator.aggregate(overwrite=True)

            self.assertEqual(records[0]["predicted_label"], "FRB")
            written = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(written, records)
            self.assertEqual(
                list(root.glob(".aggregated.jsonl.*.tmp")),
                [],
            )

    def test_cli_exposes_aggregate_windows(self) -> None:
        args = build_parser().parse_args(
            [
                "aggregate-windows",
                "--predictions",
                "windows.jsonl",
                "--output",
                "files.jsonl",
                "--threshold",
                "0.7",
                "--allow-partial",
                "--overwrite",
            ]
        )

        self.assertEqual(args.predictions, Path("windows.jsonl"))
        self.assertEqual(args.output, Path("files.jsonl"))
        self.assertEqual(args.threshold, 0.7)
        self.assertTrue(args.allow_partial)
        self.assertTrue(args.overwrite)


if __name__ == "__main__":
    unittest.main()
