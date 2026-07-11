from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluator import EvaluationConfig, PredictionEvaluator


class EvaluatorTests(unittest.TestCase):
    def test_computes_metrics_and_reports_errors_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            output_dir = root / "results"
            rows = [
                {
                    "sample_id": "sample_000000",
                    "image_path": "dataset/images/samples/sample_000000.png",
                    "true_label": "FRB",
                    "predicted_label": "FRB",
                    "confidence": 0.9,
                    "raw_model_response": "{}",
                    "parsed_response": {"label": "FRB"},
                    "error": None,
                },
                {
                    "sample_id": "sample_000001",
                    "image_path": "dataset/images/samples/sample_000001.png",
                    "true_label": "RFI",
                    "predicted_label": "NOISE",
                    "confidence": 0.4,
                    "raw_model_response": "{}",
                    "parsed_response": {"label": "NOISE"},
                    "error": None,
                },
                {
                    "sample_id": "sample_000002",
                    "image_path": "dataset/images/samples/sample_000002.png",
                    "true_label": "NOISE",
                    "predicted_label": None,
                    "confidence": None,
                    "raw_model_response": None,
                    "parsed_response": None,
                    "error": "model unavailable",
                },
            ]
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            for filename in [
                "metrics.json",
                "classification_report.txt",
                "confusion_matrix.png",
                "summary.csv",
                "evaluation_report.pdf",
            ]:
                (output_dir / filename).write_text("old", encoding="utf-8")

            evaluator = PredictionEvaluator(
                config=EvaluationConfig(
                    predictions_path=predictions,
                    output_dir=output_dir,
                ),
                project_root=root,
            )
            evaluator.evaluate(overwrite=True)

            metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["total_count"], 3)
            self.assertEqual(metrics["evaluated_count"], 2)
            self.assertEqual(metrics["skipped_count"], 1)
            self.assertEqual(metrics["error_count"], 1)
            self.assertAlmostEqual(metrics["accuracy"], 0.5)
            self.assertTrue((output_dir / "classification_report.txt").exists())
            self.assertTrue((output_dir / "confusion_matrix.png").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertTrue((output_dir / "evaluation_report.pdf").exists())
            self.assertEqual(
                (output_dir / "evaluation_report.pdf").read_bytes()[:4],
                b"%PDF",
            )


if __name__ == "__main__":
    unittest.main()
