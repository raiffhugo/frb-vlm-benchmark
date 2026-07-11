from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from plot_dataset.plotter import DatasetPlotter, PlotConfig
from simulate_dataset.config import SimulationConfig
from simulate_dataset.simulator import CLASSES as SIM_CLASSES
from simulate_dataset.simulator import SimulateSearchDatasetGenerator
from vlm_classifier.classifier import ClassifierConfig, VLMClassifier
from vlm_classifier.models import DryRunVLM


class CleanupTests(unittest.TestCase):
    def test_simulation_reset_removes_old_fits_labels_and_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            config = SimulationConfig(
                f1=1230.0,
                f2=1518.0,
                nchan=96,
                tsamp=0.001,
                nbits=2,
                gain=0.7,
                tsys=25.0,
                output_dir=dataset,
                seed=42,
                n_per_class=1,
            )
            generator = SimulateSearchDatasetGenerator(config=config, project_root=root)

            for label in SIM_CLASSES:
                path = dataset / "fits" / label / "old.fits"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old", encoding="utf-8")
            params = dataset / "metadata" / "params" / "old" / "system.params"
            params.parent.mkdir(parents=True, exist_ok=True)
            params.write_text("old", encoding="utf-8")
            labels = dataset / "metadata" / "labels.jsonl"
            labels.parent.mkdir(parents=True, exist_ok=True)
            labels.write_text("old\n", encoding="utf-8")
            (dataset / "metadata" / "labels.csv").write_text("old\n", encoding="utf-8")

            generator._reset_outputs()

            self.assertFalse((dataset / "metadata" / "params").exists())
            self.assertFalse((dataset / "metadata" / "labels.jsonl").exists())
            self.assertFalse((dataset / "metadata" / "labels.csv").exists())
            for label in SIM_CLASSES:
                self.assertFalse((dataset / "fits" / label).exists())

    def test_plot_reset_removes_old_images_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "dataset" / "images"
            manifest = root / "dataset" / "metadata" / "image_manifest.jsonl"
            plotter = DatasetPlotter(
                config=PlotConfig(
                    labels_path=root / "dataset" / "metadata" / "labels.jsonl",
                    output_dir=output_dir,
                    manifest_path=manifest,
                ),
                project_root=root,
            )

            for path in (
                output_dir / "frb" / "old.png",
                output_dir / "samples" / "old.png",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old", encoding="utf-8")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("old\n", encoding="utf-8")

            plotter._reset_outputs()

            self.assertFalse(manifest.exists())
            self.assertFalse(output_dir.exists())

    def test_classifier_overwrite_replaces_old_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "dataset" / "images" / "samples" / "sample_000000.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_text("not used by dry-run", encoding="utf-8")

            manifest = root / "dataset" / "metadata" / "image_manifest.jsonl"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_000000",
                        "source_sample_id": "frb_00000",
                        "image_path": "dataset/images/samples/sample_000000.png",
                        "true_label": "FRB",
                        "simulation_parameters": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output = root / "results" / "predictions.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("old\n", encoding="utf-8")

            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=manifest,
                    output_path=output,
                ),
                model=DryRunVLM(seed=1),
                project_root=root,
            )
            classifier.run(overwrite=True)

            text = output.read_text(encoding="utf-8")
            self.assertNotIn("old", text)
            self.assertEqual(len(text.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
