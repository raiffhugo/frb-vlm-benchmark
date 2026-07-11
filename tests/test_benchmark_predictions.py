from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_predictions.compare import (
    ModelComparisonConfig,
    ModelComparator,
    bootstrap_metric_ci,
    expected_calibration_error,
    fpr_by_source_true_label,
)
from benchmark_predictions.importer import (
    ExternalPredictionImportConfig,
    ExternalPredictionImporter,
    aggregate_probabilities,
)


class BenchmarkPredictionTests(unittest.TestCase):
    def test_importer_aggregates_duplicate_external_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmark_manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "sample_000000",
                                "true_label": "FRB",
                                "exported_fits_path": "sample_000000.fits",
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "sample_000001",
                                "true_label": "NOISE",
                                "exported_fits_path": "sample_000001.fits",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            external = root / "manifest.json"
            external.write_text(
                json.dumps(
                    [
                        {"id": "sample_000000.fits", "prob": 0.2, "label": 0},
                        {"id": "sample_000000.fits", "prob": 0.9, "label": 1},
                        {"id": "sample_000000.fits", "prob": 0.4, "label": 1},
                        {"id": "sample_000001.fits", "prob": 0.1, "label": 0},
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "predictions.jsonl"
            importer = ExternalPredictionImporter(
                config=ExternalPredictionImportConfig(
                    benchmark_manifest=manifest,
                    external_predictions=external,
                    output_path=output,
                    threshold=0.5,
                ),
                project_root=root,
            )

            records = importer.import_predictions(overwrite=True)

            self.assertEqual(records[0]["predicted_label"], "FRB")
            self.assertEqual(records[0]["confidence"], 0.9)
            self.assertEqual(
                records[0]["parsed_response"]["external_detection_count"], 3
            )
            self.assertEqual(records[0]["parsed_response"]["aggregation"], "max")
            self.assertEqual(
                records[0]["parsed_response"]["original_threshold_disagreement_count"],
                1,
            )
            self.assertEqual(records[1]["true_label"], "NON_FRB")
            self.assertEqual(records[1]["predicted_label"], "NON_FRB")
            self.assertTrue(output.exists())

    def test_importer_supports_mean_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "benchmark_manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_000000",
                        "true_label": "FRB",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            external = root / "manifest.json"
            external.write_text(
                json.dumps(
                    [
                        {"id": "sample_000000.fits", "prob": 0.2, "label": 0},
                        {"id": "sample_000000.fits", "prob": 0.8, "label": 1},
                    ]
                ),
                encoding="utf-8",
            )
            importer = ExternalPredictionImporter(
                config=ExternalPredictionImportConfig(
                    benchmark_manifest=manifest,
                    external_predictions=external,
                    output_path=root / "predictions.jsonl",
                    threshold=0.5,
                    aggregation="mean",
                ),
                project_root=root,
            )

            records = importer.import_predictions(overwrite=True)

            self.assertAlmostEqual(records[0]["parsed_response"]["frb_probability"], 0.5)
            self.assertEqual(records[0]["predicted_label"], "FRB")

    def test_aggregate_probabilities(self) -> None:
        self.assertAlmostEqual(
            aggregate_probabilities([0.1, 0.6, 0.9], method="median"),
            0.6,
        )

    def test_model_comparator_writes_summary_and_disagreements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vlm_metrics = root / "vlm_metrics.json"
            external_metrics = root / "external_metrics.json"
            metric_payload = {
                "task": "frb-binary",
                "classes": ["FRB", "NON_FRB"],
                "accuracy": 0.5,
                "macro_f1": 0.5,
                "weighted_f1": 0.5,
                "per_class": {
                    "FRB": {"precision": 0.5, "recall": 1.0, "f1_score": 0.6667},
                    "NON_FRB": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                    },
                },
            }
            vlm_metrics.write_text(json.dumps(metric_payload), encoding="utf-8")
            external_metrics.write_text(
                json.dumps({**metric_payload, "accuracy": 1.0}),
                encoding="utf-8",
            )
            vlm_predictions = root / "vlm_predictions.jsonl"
            external_predictions = root / "external_predictions.jsonl"
            vlm_predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "sample_000000",
                                "true_label": "FRB",
                                "predicted_label": "FRB",
                                "confidence": 0.8,
                                "error": None,
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "sample_000001",
                                "true_label": "NON_FRB",
                                "predicted_label": "FRB",
                                "confidence": 0.7,
                                "error": None,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            external_predictions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "sample_id": "sample_000000",
                                "true_label": "FRB",
                                "predicted_label": "FRB",
                                "confidence": 0.9,
                                "error": None,
                            }
                        ),
                        json.dumps(
                            {
                                "sample_id": "sample_000001",
                                "true_label": "NON_FRB",
                                "predicted_label": "NON_FRB",
                                "confidence": 0.1,
                                "error": None,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            comparator = ModelComparator(
                config=ModelComparisonConfig(
                    vlm_metrics=vlm_metrics,
                    external_metrics=external_metrics,
                    vlm_predictions=vlm_predictions,
                    external_predictions=external_predictions,
                    output_dir=root / "comparison",
                ),
                project_root=root,
            )

            outputs = comparator.compare(overwrite=True)

            self.assertTrue(outputs["comparison_json"].exists())
            self.assertTrue(outputs["comparison_report_pdf"].exists())
            summary = json.loads(outputs["comparison_json"].read_text(encoding="utf-8"))
            self.assertEqual(summary["sample_comparison"]["common_count"], 2)
            self.assertEqual(
                summary["sample_comparison"]["external_only_correct_count"], 1
            )

    def test_model_comparator_writes_binary_score_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metric_payload = {
                "task": "frb-binary",
                "classes": ["FRB", "NON_FRB"],
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "per_class": {
                    "FRB": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},
                    "NON_FRB": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},
                },
            }
            vlm_metrics = root / "vlm_metrics.json"
            external_metrics = root / "external_metrics.json"
            vlm_metrics.write_text(json.dumps(metric_payload), encoding="utf-8")
            external_metrics.write_text(json.dumps(metric_payload), encoding="utf-8")
            vlm_predictions = root / "vlm_predictions.jsonl"
            external_predictions = root / "external_predictions.jsonl"

            def record(
                sample_id: str,
                true_label: str,
                source_true_label: str,
                predicted_label: str,
                probability: float,
            ) -> str:
                return json.dumps(
                    {
                        "sample_id": sample_id,
                        "true_label": true_label,
                        "source_true_label": source_true_label,
                        "predicted_label": predicted_label,
                        "confidence": max(probability, 1 - probability),
                        "parsed_response": {
                            "label": predicted_label,
                            "frb_probability": probability,
                            "confidence": max(probability, 1 - probability),
                        },
                        "raw_model_response": {},
                        "error": None,
                    }
                )

            vlm_predictions.write_text(
                "\n".join(
                    [
                        record("sample_000000", "FRB", "FRB", "FRB", 0.9),
                        record("sample_000001", "NON_FRB", "RFI", "NON_FRB", 0.2),
                        record("sample_000002", "NON_FRB", "NOISE", "FRB", 0.6),
                        record("sample_000003", "FRB", "FRB", "FRB", 0.8),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            external_predictions.write_text(
                "\n".join(
                    [
                        record("sample_000000", "FRB", "FRB", "FRB", 0.95),
                        record("sample_000001", "NON_FRB", "RFI", "NON_FRB", 0.1),
                        record("sample_000002", "NON_FRB", "NOISE", "NON_FRB", 0.3),
                        record("sample_000003", "FRB", "FRB", "FRB", 0.7),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            comparator = ModelComparator(
                config=ModelComparisonConfig(
                    vlm_metrics=vlm_metrics,
                    external_metrics=external_metrics,
                    vlm_predictions=vlm_predictions,
                    external_predictions=external_predictions,
                    output_dir=root / "comparison",
                    bootstrap_iterations=25,
                    bootstrap_seed=7,
                ),
                project_root=root,
            )

            outputs = comparator.compare(overwrite=True)

            summary = json.loads(outputs["comparison_json"].read_text(encoding="utf-8"))
            binary = summary["binary_score_analysis"]
            self.assertTrue(binary["available"])
            self.assertEqual(binary["paired_count"], 4)
            self.assertTrue(outputs["paired_scores"].exists())
            self.assertTrue(outputs["statistical_tests"].exists())
            self.assertTrue(outputs["precision_recall_curve"].exists())

    def test_binary_metric_helpers(self) -> None:
        y_true = [1, 0, 1, 0]
        scores = [0.9, 0.2, 0.8, 0.1]

        ci = bootstrap_metric_ci(
            y_true,
            scores,
            metric="average_precision",
            iterations=10,
            seed=1,
        )
        ece = expected_calibration_error(y_true, scores, n_bins=2)
        fpr = fpr_by_source_true_label(
            y_true,
            ["FRB", "RFI", "FRB", "NOISE"],
            scores,
            [0.7, 0.6, 0.8, 0.2],
            threshold=0.5,
        )

        self.assertAlmostEqual(ci["estimate"], 1.0)
        self.assertIn("estimate", ece)
        self.assertEqual(fpr["RFI"]["external_false_positives"], 1)


if __name__ == "__main__":
    unittest.main()
