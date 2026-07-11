from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluator_continuous import (
    ContinuousEvaluationConfig,
    ContinuousPredictionEvaluator,
)


class ContinuousEvaluatorTests(unittest.TestCase):
    def test_writes_continuous_metrics_and_plots_for_two_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output_dir = root / "continuous"
            labels = ["FRB", "FRB", "NON_FRB", "NON_FRB"]
            self._write_predictions(first, labels, [0.9, 0.7, 0.3, 0.1])
            self._write_predictions(second, labels, [0.8, 0.4, 0.6, 0.2])
            evaluator = ContinuousPredictionEvaluator(
                config=ContinuousEvaluationConfig(
                    predictions=(first, second),
                    names=("vlm", "external"),
                    output_dir=output_dir,
                    threshold_steps=11,
                    calibration_bins=5,
                ),
                project_root=root,
            )

            outputs = evaluator.evaluate(overwrite=True)

            for path in outputs.values():
                self.assertTrue(path.exists(), path)
            metrics = json.loads(outputs["metrics_continuous"].read_text(encoding="utf-8"))
            self.assertEqual(metrics["task"], "frb-binary")
            self.assertEqual(set(metrics["models"]), {"vlm", "external"})
            self.assertIn("f1_optimal", metrics["models"]["vlm"]["operating_points"])
            self.assertIn("youden_j_optimal", metrics["models"]["vlm"]["operating_points"])
            self.assertIn("0.01", metrics["models"]["vlm"]["prior_adjusted"])
            self.assertEqual(metrics["paired_comparison"]["paired_count"], 4)
            self.assertIn("mcnemar_at_threshold", metrics["paired_comparison"])

    def _write_predictions(
        self,
        path: Path,
        labels: list[str],
        scores: list[float],
    ) -> None:
        records = []
        for index, (label, score) in enumerate(zip(labels, scores)):
            predicted_label = "FRB" if score >= 0.5 else "NON_FRB"
            records.append(
                {
                    "sample_id": f"sample_{index:06d}",
                    "true_label": label,
                    "source_true_label": label,
                    "predicted_label": predicted_label,
                    "confidence": max(score, 1.0 - score),
                    "raw_model_response": "{}",
                    "parsed_response": {
                        "label": predicted_label,
                        "confidence": max(score, 1.0 - score),
                        "frb_probability": score,
                    },
                    "error": None,
                    "content_warning": None,
                }
            )
        path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
