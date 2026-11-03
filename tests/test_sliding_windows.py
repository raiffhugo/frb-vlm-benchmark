from __future__ import annotations

import argparse
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_real.builder import RealDatasetPreparer, RealPrepConfig
from prepare_real.cli import add_prepare_real_parser

try:  # discover -s tests poe tests/ no sys.path; -m tests.x nao
    from test_prepare_real import _write_synthetic_psrfits
except ModuleNotFoundError:
    from tests.test_prepare_real import _write_synthetic_psrfits


NCHAN, NSBLK, NSUB = 8, 128, 8
NSAMP = NSBLK * NSUB
TSAMP = 1e-3


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fits_path", "label", "id", "toa", "dm"])
        writer.writeheader()
        writer.writerows(rows)


def _config(
    manifest: Path,
    output_dir: str,
    *,
    window_seconds: float = 0.256,
    stride_seconds: float = 0.2,
) -> RealPrepConfig:
    return RealPrepConfig(
        input_manifest=manifest,
        output_dir=Path(output_dir),
        window_seconds=window_seconds,
        sliding_windows=True,
        window_stride_seconds=stride_seconds,
        time_bins=64,
        freq_bins=8,
        width=160,
        height=120,
        dpi=40,
    )


def _write_fits(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    _write_synthetic_psrfits(
        path,
        float_data_file_order=rng.normal(20.0, 2.0, size=(NCHAN, NSAMP)).astype(np.float32),
        freqs_file_order=np.linspace(1000.0, 1500.0, NCHAN),
        tsamp_s=TSAMP,
        nsblk=NSBLK,
    )


class SlidingWindowConfigTests(unittest.TestCase):
    def test_stride_validation_and_cli_flags(self) -> None:
        base = {"input_manifest": Path("manifest.csv"), "output_dir": Path("dataset_real")}
        for stride in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(stride=stride):
                with self.assertRaisesRegex(ValueError, "window_stride_seconds"):
                    RealPrepConfig(**base, window_stride_seconds=stride).validate()
        with self.assertRaisesRegex(ValueError, "to cover the file"):
            RealPrepConfig(
                **base,
                window_seconds=1.0,
                sliding_windows=True,
                window_stride_seconds=1.1,
            ).validate()

        parser = argparse.ArgumentParser()
        add_prepare_real_parser(parser)
        defaults = parser.parse_args(["--input-manifest", "manifest.csv"])
        explicit = parser.parse_args(
            [
                "--input-manifest",
                "manifest.csv",
                "--sliding-windows",
                "--window-stride-seconds",
                "0.25",
            ]
        )

        self.assertFalse(defaults.sliding_windows)
        self.assertEqual(defaults.window_stride_seconds, 1.0)
        self.assertTrue(explicit.sliding_windows)
        self.assertEqual(explicit.window_stride_seconds, 0.25)


class SlidingWindowBuilderTests(unittest.TestCase):
    def test_windows_ignore_truth_and_cover_file_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fits_path = root / "shared.fits"
            _write_fits(fits_path, seed=20260716)
            frb_manifest = root / "frb.csv"
            other_manifest = root / "other.csv"
            _write_manifest(
                frb_manifest,
                [{"fits_path": str(fits_path), "label": "FRB", "id": "src", "toa": 0.1, "dm": 111.0}],
            )
            _write_manifest(
                other_manifest,
                [{"fits_path": str(fits_path), "label": "RFI", "id": "src", "toa": 0.9, "dm": 999.0}],
            )

            frb_records = RealDatasetPreparer(
                config=_config(frb_manifest, "dataset_frb"), project_root=root
            ).prepare(overwrite=True)
            other_records = RealDatasetPreparer(
                config=_config(other_manifest, "dataset_other"), project_root=root
            ).prepare(overwrite=True)

            def signature(record):  # type: ignore[no-untyped-def]
                return (
                    record["sample_id"],
                    record["file_group_id"],
                    record["window_index"],
                    record["preprocessing"]["window_start_sample"],
                    record["preprocessing"]["window_stop_sample"],
                    (root / record["image_path"]).read_bytes(),
                )

            self.assertEqual(
                [signature(record) for record in frb_records],
                [signature(record) for record in other_records],
            )
            starts = [record["preprocessing"]["window_start_sample"] for record in frb_records]
            self.assertEqual(starts, [0, 200, 400, 600, 768])
            self.assertEqual(frb_records[-1]["preprocessing"]["window_stop_sample"], NSAMP)
            self.assertEqual(len({record["file_group_id"] for record in frb_records}), 1)
            for index, record in enumerate(frb_records):
                preprocessing = record["preprocessing"]
                self.assertEqual(preprocessing["window_mode"], "sliding")
                self.assertIsNone(preprocessing["toa_reference"])
                self.assertEqual(record["window_index"], index)
                self.assertEqual(record["window_count"], 5)
                self.assertTrue(
                    record["sample_id"].endswith(
                        f"_w{index:04d}_s{preprocessing['window_start_sample']:09d}"
                    )
                )
                self.assertAlmostEqual(
                    record["window_stop_seconds"] - record["window_start_seconds"], 0.256
                )
            # o catalogo continua registrado, para o casamento temporal posterior
            self.assertEqual(frb_records[0]["real_parameters"]["toa_s"], 0.1)
            self.assertEqual(frb_records[0]["real_parameters"]["dm"], 111.0)

    def test_ids_and_order_do_not_depend_on_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows: list[dict[str, object]] = []
            for seed, source_id in enumerate(("alpha", "beta")):
                fits_path = root / f"{source_id}.fits"
                _write_fits(fits_path, seed=seed)
                rows.append(
                    {"fits_path": str(fits_path), "label": "NOISE", "id": source_id, "toa": "", "dm": ""}
                )
            forward_manifest = root / "forward.csv"
            reverse_manifest = root / "reverse.csv"
            _write_manifest(forward_manifest, rows)
            _write_manifest(reverse_manifest, list(reversed(rows)))

            forward = RealDatasetPreparer(
                config=_config(forward_manifest, "fwd", window_seconds=0.512, stride_seconds=0.5),
                project_root=root,
            ).prepare(overwrite=True)
            reverse = RealDatasetPreparer(
                config=_config(reverse_manifest, "rev", window_seconds=0.512, stride_seconds=0.5),
                project_root=root,
            ).prepare(overwrite=True)

            def signature(record):  # type: ignore[no-untyped-def]
                return (
                    record["sample_id"],
                    record["source_sample_id"],
                    record["window_index"],
                    record["preprocessing"]["window_start_sample"],
                )

            self.assertEqual([signature(r) for r in forward], [signature(r) for r in reverse])
            self.assertEqual(len(forward), 6)

    def test_fast_frex_geometry_gives_six_windows(self) -> None:
        # FAST-FREX: 61440 amostras de 98,304 us (6,04 s); janelas de 2 s e passo
        # de 1 s (plano, secao 10.5)
        tsamp = 98.304e-6
        layout = type(
            "Layout",
            (),
            {"duration_s": 61440 * tsamp, "tsamp_s": tsamp, "nsamp_total": 61440},
        )()
        preparer = RealDatasetPreparer(
            config=RealPrepConfig(
                input_manifest=Path("m.csv"),
                output_dir=Path("out"),
                sliding_windows=True,
                window_stride_seconds=1.0,
            ),
            project_root=Path("/tmp"),
        )

        plans = preparer._sliding_window_plans(layout)

        self.assertEqual(len(plans), 6)
        self.assertEqual(plans[0].start_sample, 0)
        self.assertEqual(plans[-1].stop_sample, 61440)
        self.assertEqual({plan.n_samples for plan in plans}, {20480})
        for current, following in zip(plans, plans[1:]):
            self.assertLessEqual(following.start_sample, current.stop_sample)



class ParallelPrepareTests(unittest.TestCase):
    def test_workers_do_not_change_records_or_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows: list[dict[str, object]] = []
            for seed, (source_id, label) in enumerate((("a", "FRB"), ("b", "NOISE"), ("c", "RFI"))):
                fits_path = root / f"{source_id}.fits"
                _write_fits(fits_path, seed=seed)
                rows.append(
                    {
                        "fits_path": str(fits_path),
                        "label": label,
                        "id": source_id,
                        "toa": 0.4 if label == "FRB" else "",
                        "dm": 50.0 if label == "FRB" else "",
                    }
                )
            manifest = root / "manifest.csv"
            _write_manifest(manifest, rows)

            for sliding in (True, False):
                outputs = {}
                for workers in (1, 2):
                    config = RealPrepConfig(
                        input_manifest=manifest,
                        output_dir=Path(f"out_{sliding}_{workers}"),
                        window_seconds=0.256,
                        sliding_windows=sliding,
                        window_stride_seconds=0.2,
                        time_bins=64,
                        freq_bins=8,
                        width=160,
                        height=120,
                        dpi=40,
                    )
                    outputs[workers] = RealDatasetPreparer(config=config, project_root=root).prepare(
                        workers=workers
                    )

                def signature(record):  # type: ignore[no-untyped-def]
                    metadata = {
                        key: value
                        for key, value in record.items()
                        if key not in {"image_path"}
                    }
                    return metadata, (root / record["image_path"]).read_bytes()

                with self.subTest(sliding=sliding):
                    self.assertEqual(len(outputs[1]), 15 if sliding else 3)
                    self.assertEqual(
                        [signature(r) for r in outputs[1]],
                        [signature(r) for r in outputs[2]],
                    )


if __name__ == "__main__":
    unittest.main()
