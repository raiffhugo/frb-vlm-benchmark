from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plot_dataset.plotter import (
    ANONYMIZED_IMAGE_DIR,
    DEFAULT_IMAGE_DPI,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DatasetPlotter,
    PlotConfig,
)


class PlotConfigTests(unittest.TestCase):
    def test_uses_high_resolution_defaults(self) -> None:
        config = PlotConfig(
            labels_path=Path("dataset/metadata/labels.jsonl"),
            output_dir=Path("dataset/images"),
            manifest_path=Path("dataset/metadata/image_manifest.jsonl"),
        )

        self.assertEqual(config.width, DEFAULT_IMAGE_WIDTH)
        self.assertEqual(config.height, DEFAULT_IMAGE_HEIGHT)
        self.assertEqual(config.dpi, DEFAULT_IMAGE_DPI)
        self.assertEqual(config.width, 1536)
        self.assertEqual(config.height, 1152)
        self.assertEqual(config.dpi, 192)
        self.assertFalse(config.include_title)

    def test_uses_anonymized_image_paths(self) -> None:
        root = Path("/tmp/project")
        config = PlotConfig(
            labels_path=Path("dataset/metadata/labels.jsonl"),
            output_dir=Path("dataset/images"),
            manifest_path=Path("dataset/metadata/image_manifest.jsonl"),
        )
        plotter = DatasetPlotter(config=config, project_root=root)

        image_path = plotter._image_path_for_index(0)

        self.assertEqual(
            image_path,
            root / "dataset" / "images" / ANONYMIZED_IMAGE_DIR / "sample_000000.png",
        )
        lower_path = image_path.as_posix().lower()
        self.assertNotIn("frb", lower_path)
        self.assertNotIn("rfi", lower_path)
        self.assertNotIn("noise", lower_path)

    def test_manifest_uses_anonymous_sample_id(self) -> None:
        class DummyPlotter(DatasetPlotter):
            def _read_with_astropy(self, path):  # type: ignore[no-untyped-def]
                return object()

            def _plot_spectrum(self, spectrum, image_id, image_path):  # type: ignore[no-untyped-def]
                image_path.write_text("png", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fits_path = root / "dataset" / "fits" / "frb" / "frb_000000.fits"
            fits_path.parent.mkdir(parents=True, exist_ok=True)
            fits_path.write_text("fits", encoding="utf-8")
            config = PlotConfig(
                labels_path=Path("dataset/metadata/labels.jsonl"),
                output_dir=Path("dataset/images"),
                manifest_path=Path("dataset/metadata/image_manifest.jsonl"),
            )
            plotter = DummyPlotter(config=config, project_root=root)
            plotter._create_directories()

            record = plotter._plot_one(
                {
                    "sample_id": "frb_000000",
                    "fits_path": "dataset/fits/frb/frb_000000.fits",
                    "label": "FRB",
                    "parameters": {},
                },
                image_index=0,
                overwrite=True,
            )

            self.assertEqual(record["sample_id"], "sample_000000")
            self.assertEqual(record["source_sample_id"], "frb_000000")
            self.assertEqual(
                record["image_path"], "dataset/images/samples/sample_000000.png"
            )
            self.assertTrue(record["plot"]["anonymized_image_path"])


if __name__ == "__main__":
    unittest.main()
