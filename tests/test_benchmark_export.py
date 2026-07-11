from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from benchmark_export.converter import (
    BenchmarkExportConfig,
    BenchmarkExporter,
    convert_to_benchmark_fits,
)


class BenchmarkExportTests(unittest.TestCase):
    def test_convert_to_benchmark_fits_drops_history_and_adds_nbin_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.fits"
            dst = root / "sample_000000.fits"
            self._write_source_fits(src)

            convert_to_benchmark_fits(src, dst)

            with fits.open(dst, lazy_load_hdus=False) as hdul:
                self.assertEqual(len(hdul), 2)
                self.assertEqual(hdul[1].name, "SUBINT")
                self.assertEqual(hdul[0].header["STT_IMJD"], 58456)
                self.assertEqual(hdul[0].header["STT_SMJD"], 36400)
                data = hdul[1].data["DATA"]
                self.assertEqual(data.ndim, 5)
                self.assertEqual(data.shape[-1], 1)

    def test_exporter_writes_neutral_files_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dataset" / "fits" / "frb" / "frb_00000.fits"
            self._write_source_fits(source)
            labels = root / "dataset" / "metadata" / "labels.jsonl"
            labels.parent.mkdir(parents=True, exist_ok=True)
            labels.write_text(
                json.dumps(
                    {
                        "sample_id": "frb_00000",
                        "label": "FRB",
                        "fits_path": "dataset/fits/frb/frb_00000.fits",
                        "parameters": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "benchmark_input"
            exporter = BenchmarkExporter(
                config=BenchmarkExportConfig(
                    labels_path=labels,
                    output_dir=output_dir,
                ),
                project_root=root,
            )

            records = exporter.export(overwrite=True)

            exported_path = output_dir / "fits" / "sample_000000.fits"
            manifest = output_dir / "metadata" / "benchmark_manifest.jsonl"
            self.assertTrue(exported_path.exists())
            self.assertTrue(manifest.exists())
            self.assertEqual(records[0]["sample_id"], "sample_000000")
            self.assertEqual(records[0]["source_sample_id"], "frb_00000")
            self.assertEqual(records[0]["true_label"], "FRB")
            self.assertEqual(
                records[0]["exported_fits_path"],
                "benchmark_input/fits/sample_000000.fits",
            )
            self.assertEqual(records[0]["conversion_status"], "ok")
            lower_output_path = exported_path.as_posix().lower()
            self.assertNotIn("frb", lower_output_path)
            self.assertNotIn("rfi", lower_output_path)
            self.assertNotIn("noise", lower_output_path)

    def _write_source_fits(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        primary = fits.PrimaryHDU()
        primary.header["STT_IMJD"] = 58456
        primary.header["STT_SMJD"] = 36400
        primary.header["STT_OFFS"] = 0.0
        primary.header["RA"] = 0.0
        primary.header["DEC"] = 0.0
        history = fits.ImageHDU(name="HISTORY")
        raw = np.arange(2 * 3 * 1 * 4, dtype=np.uint8).reshape(2, 3, 1, 4)
        subint = fits.BinTableHDU.from_columns(
            [
                fits.Column(
                    name="DATA",
                    format="12B",
                    dim="(4,1,3)",
                    array=raw,
                )
            ],
            name="SUBINT",
        )
        fits.HDUList([primary, history, subint]).writeto(path, overwrite=True)


if __name__ == "__main__":
    unittest.main()
