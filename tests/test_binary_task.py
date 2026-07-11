from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm_classifier.classifier import ClassifierConfig, VLMClassifier
from vlm_classifier.models import BaseVLM


class StaticBinaryModel(BaseVLM):
    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        self.prompt = prompt
        return (
            '{"label": "RFI", "frb_probability": 0.1, '
            '"confidence": 0.9, "reason": "not dispersed"}'
        )


class InvalidJsonBinaryModel(BaseVLM):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        self.calls += 1
        return "not json"


class TransientThenValidBinaryModel(BaseVLM):
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("timed out")
        return (
            '{"label": "FRB", "frb_probability": 0.8, '
            '"confidence": 0.8, "reason": "dispersed"}'
        )


class BinaryTaskTests(unittest.TestCase):
    def test_classifier_maps_truth_and_prediction_to_binary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, output = self._write_manifest(root, true_label="RFI")
            model = StaticBinaryModel()
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=output,
                    task="frb-binary",
                ),
                model=model,
                project_root=root,
            )

            records = classifier.run(overwrite=True)

            self.assertIn("FRB|NON_FRB", model.prompt)
            self.assertEqual(records[0]["true_label"], "NON_FRB")
            self.assertEqual(records[0]["source_true_label"], "RFI")
            self.assertEqual(records[0]["predicted_label"], "NON_FRB")
            self.assertAlmostEqual(
                records[0]["parsed_response"]["frb_probability"],
                0.1,
            )

    def test_content_error_preserves_raw_response_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, output = self._write_manifest(root, true_label="FRB")
            model = InvalidJsonBinaryModel()
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=output,
                    task="frb-binary",
                    max_retries=2,
                    retry_delay=0,
                ),
                model=model,
                project_root=root,
            )

            records = classifier.run(overwrite=True)

            self.assertEqual(model.calls, 1)
            self.assertIsNone(records[0]["error"])
            self.assertEqual(records[0]["raw_model_response"], "not json")
            self.assertIsNone(records[0]["parsed_response"])
            self.assertIsNone(records[0]["predicted_label"])
            self.assertIn("Response has no JSON object", records[0]["content_warning"])

    def test_transient_model_error_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, output = self._write_manifest(root, true_label="FRB")
            model = TransientThenValidBinaryModel()
            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=output,
                    task="frb-binary",
                    max_retries=2,
                    retry_delay=0,
                ),
                model=model,
                project_root=root,
            )

            records = classifier.run(overwrite=True)

            self.assertEqual(model.calls, 2)
            self.assertIsNone(records[0]["error"])
            self.assertIsNone(records[0]["content_warning"])
            self.assertEqual(records[0]["predicted_label"], "FRB")

    def _write_manifest(self, root: Path, *, true_label: str) -> tuple[Path, Path]:
        image = root / "dataset" / "images" / "samples" / "sample_000000.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_text("fake image", encoding="utf-8")
        manifest = root / "dataset" / "metadata" / "image_manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "sample_id": "sample_000000",
                    "image_path": "dataset/images/samples/sample_000000.png",
                    "true_label": true_label,
                    "simulation_parameters": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest, root / "results" / "predictions.jsonl"


if __name__ == "__main__":
    unittest.main()
