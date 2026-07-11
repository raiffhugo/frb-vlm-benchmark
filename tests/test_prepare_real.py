from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

from prepare_real.builder import RealDatasetPreparer, RealPrepConfig
from prepare_real.preprocess import (
    averaged_frequency_axis,
    block_average,
    burst_center_s,
    channel_keep_mask,
    dispersion_delay_s,
    normalize_per_channel,
    plan_window,
    robust_channel_stats,
)
from prepare_real.reader import read_block, read_layout
from vlm_classifier.classifier import ClassifierConfig, VLMClassifier
from vlm_classifier.models import DryRunVLM
from vlm_classifier.parser import classes_for_task


def _write_synthetic_psrfits(
    path: Path,
    *,
    float_data_file_order: np.ndarray,
    freqs_file_order: np.ndarray,
    tsamp_s: float,
    nsblk: int,
    zero_off: float = 0.0,
) -> None:
    nchan, nsamp = float_data_file_order.shape
    if nsamp % nsblk != 0:
        raise ValueError("nsamp must be a multiple of nsblk in the synthetic file.")
    nsub = nsamp // nsblk

    raw = np.zeros((nsub, nsblk, nchan), dtype=np.uint8)
    scales = np.zeros((nsub, nchan), dtype=np.float32)
    offsets = np.zeros((nsub, nchan), dtype=np.float32)
    for sub in range(nsub):
        block = float_data_file_order[:, sub * nsblk : (sub + 1) * nsblk].T
        cmin = block.min(axis=0)
        cmax = block.max(axis=0)
        scl = (cmax - cmin) / (255.0 - zero_off)
        scl[scl <= 0] = 1.0
        quantized = np.clip(np.round((block - cmin) / scl + zero_off), 0, 255)
        raw[sub] = quantized.astype(np.uint8)
        scales[sub] = scl
        offsets[sub] = cmin

    columns = [
        fits.Column(
            name="TSUBINT",
            format="1D",
            array=np.full(nsub, nsblk * tsamp_s),
        ),
        fits.Column(
            name="OFFS_SUB",
            format="1D",
            array=(np.arange(nsub) + 0.5) * nsblk * tsamp_s,
        ),
        fits.Column(
            name="DAT_FREQ",
            format=f"{nchan}D",
            array=np.tile(freqs_file_order, (nsub, 1)),
        ),
        fits.Column(
            name="DAT_WTS",
            format=f"{nchan}E",
            array=np.ones((nsub, nchan), dtype=np.float32),
        ),
        fits.Column(name="DAT_OFFS", format=f"{nchan}E", array=offsets),
        fits.Column(name="DAT_SCL", format=f"{nchan}E", array=scales),
        fits.Column(
            name="DATA",
            format=f"{nsblk * nchan}B",
            dim=f"({nchan},1,{nsblk})",
            array=raw.reshape(nsub, nsblk * nchan),
        ),
    ]
    subint = fits.BinTableHDU.from_columns(columns, name="SUBINT")
    subint.header["NCHAN"] = nchan
    subint.header["NPOL"] = 1
    subint.header["POL_TYPE"] = "AA+BB"
    subint.header["NSBLK"] = nsblk
    subint.header["NBITS"] = 8
    subint.header["NBIN"] = 1
    subint.header["NCHNOFFS"] = 0
    subint.header["NSUBOFFS"] = 0
    subint.header["TBIN"] = tsamp_s
    subint.header["ZERO_OFF"] = zero_off
    subint.header["CHAN_BW"] = float(freqs_file_order[1] - freqs_file_order[0])

    # Standard search-mode PSRFITS cards, required by external readers such as
    # the 'your' library (is_PSRFITS + SpectraInfo).
    primary = fits.PrimaryHDU()
    primary.header["FITSTYPE"] = "PSRFITS"
    primary.header["OBS_MODE"] = "SEARCH"
    primary.header["TELESCOP"] = "FAST"
    primary.header["OBSERVER"] = "synthetic"
    primary.header["SRC_NAME"] = "SYNTH"
    primary.header["FRONTEND"] = "SYNTH"
    primary.header["BACKEND"] = "SYNTH"
    primary.header["PROJID"] = "TEST"
    primary.header["DATE-OBS"] = "2020-01-01T00:00:00"
    primary.header["FD_POLN"] = "LIN"
    primary.header["RA"] = "00:00:00.0"
    primary.header["DEC"] = "00:00:00.0"
    primary.header["BMIN"] = 0.0
    primary.header["TRK_MODE"] = "TRACK"
    primary.header["STT_IMJD"] = 58849
    primary.header["STT_SMJD"] = 0
    primary.header["STT_OFFS"] = 0.0
    primary.header["OBSFREQ"] = float(freqs_file_order.mean())
    primary.header["OBSBW"] = float(freqs_file_order[-1] - freqs_file_order[0])
    primary.header["OBSNCHAN"] = nchan
    fits.HDUList([primary, subint]).writeto(path, overwrite=True)


def _inject_burst(
    data_ascending: np.ndarray,
    freqs_ascending: np.ndarray,
    tsamp_s: float,
    *,
    toa_top_s: float,
    dm: float,
    amplitude: float,
    width_samples: int,
) -> None:
    f_top = float(freqs_ascending[-1])
    nsamp = data_ascending.shape[1]
    for index, freq in enumerate(freqs_ascending):
        start = int(round((toa_top_s + dispersion_delay_s(dm, float(freq), f_top)) / tsamp_s))
        stop = min(start + width_samples, nsamp)
        if 0 <= start < nsamp:
            data_ascending[index, start:stop] += amplitude


class ReaderTests(unittest.TestCase):
    def test_reader_applies_scales_and_orders_frequency_ascending(self) -> None:
        nchan, nsblk, nsub = 32, 128, 8
        nsamp = nsblk * nsub
        tsamp = 1e-3
        freqs_desc = np.linspace(1500.0, 1000.0, nchan)
        times = np.arange(nsamp) % nsblk
        float_desc = (
            100.0
            + 5.0 * np.arange(nchan)[:, None]
            + 0.05 * times[None, :]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.fits"
            _write_synthetic_psrfits(
                path,
                float_data_file_order=float_desc,
                freqs_file_order=freqs_desc,
                tsamp_s=tsamp,
                nsblk=nsblk,
            )

            layout = read_layout(path)
            self.assertEqual(layout.nchan, nchan)
            self.assertEqual(layout.nsamp_total, nsamp)
            self.assertAlmostEqual(layout.tsamp_s, tsamp)
            self.assertTrue(layout.frequency_descending)
            self.assertAlmostEqual(float(layout.frequencies_mhz[0]), 1000.0)
            self.assertAlmostEqual(float(layout.frequencies_mhz[-1]), 1500.0)

            block = read_block(path, start_sample=0, stop_sample=nsamp)
            self.assertEqual(block.data.shape, (nchan, nsamp))
            np.testing.assert_allclose(block.data, float_desc[::-1], atol=0.1)

            partial = read_block(path, start_sample=200, stop_sample=500)
            np.testing.assert_allclose(partial.data, block.data[:, 200:500], atol=1e-6)

    def test_reader_applies_nonzero_zero_off(self) -> None:
        nchan, nsblk, nsub = 8, 64, 2
        nsamp = nsblk * nsub
        freqs = np.linspace(1000.0, 1500.0, nchan)
        float_asc = 40.0 + np.arange(nchan)[:, None] + 0.1 * (np.arange(nsamp) % nsblk)[None, :]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero_off.fits"
            _write_synthetic_psrfits(
                path,
                float_data_file_order=float_asc,
                freqs_file_order=freqs,
                tsamp_s=1e-3,
                nsblk=nsblk,
                zero_off=10.0,
            )
            block = read_block(path, start_sample=0, stop_sample=nsamp)
            np.testing.assert_allclose(block.data, float_asc, atol=0.2)


class PreprocessTests(unittest.TestCase):
    def test_dispersion_delay_matches_reference_value(self) -> None:
        delay = dispersion_delay_s(100.0, 1000.0, 1500.0)
        self.assertAlmostEqual(delay, 0.2304893, places=6)
        self.assertAlmostEqual(dispersion_delay_s(100.0, 1500.0, 1500.0), 0.0)

    def test_burst_center_respects_toa_reference(self) -> None:
        span = dispersion_delay_s(100.0, 1000.0, 1500.0)
        center_top = burst_center_s(
            toa_s=1.0, dm=100.0, f_low_mhz=1000.0, f_high_mhz=1500.0, toa_reference="top"
        )
        center_bottom = burst_center_s(
            toa_s=1.0, dm=100.0, f_low_mhz=1000.0, f_high_mhz=1500.0, toa_reference="bottom"
        )
        self.assertAlmostEqual(center_top, 1.0 + span / 2.0)
        self.assertAlmostEqual(center_bottom, 1.0 - span / 2.0)
        self.assertGreater(
            burst_center_s(
                toa_s=1.0,
                dm=100.0,
                f_low_mhz=1000.0,
                f_high_mhz=1500.0,
                toa_reference="infinite",
            ),
            center_top,
        )

    def test_plan_window_centers_and_clips(self) -> None:
        plan = plan_window(
            center_s=0.512,
            window_seconds=0.256,
            tsamp_s=1e-3,
            nsamp_total=1024,
            time_bins=128,
        )
        self.assertEqual(plan.time_factor, 2)
        self.assertEqual(plan.n_samples, 256)
        self.assertEqual(plan.start_sample, 384)

        early = plan_window(
            center_s=0.0,
            window_seconds=0.256,
            tsamp_s=1e-3,
            nsamp_total=1024,
            time_bins=128,
        )
        self.assertEqual(early.start_sample, 0)

        late = plan_window(
            center_s=10.0,
            window_seconds=0.256,
            tsamp_s=1e-3,
            nsamp_total=1024,
            time_bins=128,
        )
        self.assertEqual(late.stop_sample, 1024)

    def test_channel_keep_mask_flags_loud_dead_and_bright_channels(self) -> None:
        rng = np.random.default_rng(7)
        data = rng.normal(0.0, 1.0, size=(64, 4096)).astype(np.float32)
        data[5] *= 20.0
        data[9] = 0.0
        data[20] += 50.0

        median, sigma = robust_channel_stats(data)
        keep = channel_keep_mask(median=median, sigma=sigma, zap_sigma=5.0)

        self.assertFalse(keep[5])
        self.assertFalse(keep[9])
        self.assertFalse(keep[20])
        self.assertEqual(int(keep.sum()), 61)

    def test_block_average_is_mask_aware(self) -> None:
        zscored = np.array(
            [
                [1.0, 1.0, 3.0, 3.0],
                [9.0, 9.0, 9.0, 9.0],
                [2.0, 2.0, 4.0, 4.0],
                [6.0, 6.0, 8.0, 8.0],
            ],
            dtype=np.float32,
        )
        keep = np.array([True, False, True, True])

        averaged, freq_factor, trim_low = block_average(
            zscored, keep=keep, time_factor=2, freq_bins=2
        )

        self.assertEqual(freq_factor, 2)
        self.assertEqual(trim_low, 0)
        # Mask-aware means (1.0, 3.0) and (4.0, 6.0), rescaled by
        # sqrt(time_factor * n_kept_channels): sqrt(2) and sqrt(4).
        np.testing.assert_allclose(
            averaged[0], np.array([1.0, 3.0]) * np.sqrt(2.0), rtol=1e-6
        )
        np.testing.assert_allclose(averaged[1], [8.0, 12.0])

        axis = averaged_frequency_axis(
            np.array([1000.0, 1010.0, 1020.0, 1030.0]),
            freq_bins=2,
            freq_factor=2,
            trim_low=0,
        )
        np.testing.assert_allclose(axis, [1005.0, 1025.0])

    def test_weak_dispersed_burst_becomes_visible_after_chain(self) -> None:
        rng = np.random.default_rng(42)
        nchan, nsamp = 128, 8192
        tsamp = 1e-3
        freqs = np.linspace(1000.0, 1500.0, nchan)

        signal = rng.normal(0.0, 1.0, size=(nchan, nsamp))
        dm, toa_top, width = 870.0, 3.0, 64
        _inject_burst(
            signal,
            freqs,
            tsamp,
            toa_top_s=toa_top,
            dm=dm,
            amplitude=1.0,
            width_samples=width,
        )
        gain = 1.0 + 0.5 * np.sin(np.linspace(0.0, 3.0 * np.pi, nchan))
        offset = 50.0 + 0.5 * np.arange(nchan)
        data = (signal * gain[:, None] + offset[:, None]).astype(np.float32)

        median, sigma = robust_channel_stats(data)
        keep = channel_keep_mask(median=median, sigma=sigma, zap_sigma=5.0)
        self.assertTrue(keep.all())

        zscored = normalize_per_channel(data, median=median, sigma=sigma, keep=keep)
        time_factor, freq_bins = 32, 32
        averaged, freq_factor, trim_low = block_average(
            zscored, keep=keep, time_factor=time_factor, freq_bins=freq_bins
        )

        background = np.median(np.abs(averaged - np.median(averaged))) * 1.4826
        peak = float(averaged.max())
        self.assertGreater(peak, 6.0 * background)

        axis = averaged_frequency_axis(
            freqs, freq_bins=freq_bins, freq_factor=freq_factor, trim_low=trim_low
        )
        bin_seconds = time_factor * tsamp
        for row in (4, 12, 20, 28):
            expected_arrival = toa_top + dispersion_delay_s(dm, float(axis[row]), 1500.0)
            expected_bin = (expected_arrival + width * tsamp / 2.0) / bin_seconds
            observed_bin = int(np.argmax(averaged[row]))
            self.assertLessEqual(abs(observed_bin - expected_bin), 2.0)


class YourReaderParityTests(unittest.TestCase):
    def test_your_backend_matches_astropy_reader(self) -> None:
        try:
            import your  # noqa: F401
        except ImportError:
            self.skipTest("the 'your' library is not installed (optional [real] extra)")
        from prepare_real import reader_your

        nchan, nsblk, nsub = 32, 128, 8
        nsamp = nsblk * nsub
        tsamp = 1e-3
        freqs_desc = np.linspace(1500.0, 1000.0, nchan)
        rng = np.random.default_rng(5)
        float_desc = (
            50.0
            + 3.0 * np.arange(nchan)[:, None]
            + 5.0 * rng.normal(size=(nchan, nsamp))
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.fits"
            _write_synthetic_psrfits(
                path,
                float_data_file_order=float_desc.astype(np.float32),
                freqs_file_order=freqs_desc,
                tsamp_s=tsamp,
                nsblk=nsblk,
            )

            layout_a = read_layout(path)
            layout_y = reader_your.read_layout(path)
            self.assertEqual(layout_y.nchan, layout_a.nchan)
            self.assertEqual(layout_y.nsamp_total, layout_a.nsamp_total)
            self.assertAlmostEqual(layout_y.tsamp_s, layout_a.tsamp_s)
            np.testing.assert_allclose(
                layout_y.frequencies_mhz, layout_a.frequencies_mhz
            )

            block_a = read_block(path, start_sample=100, stop_sample=612)
            block_y = reader_your.read_block(path, start_sample=100, stop_sample=612)
            self.assertEqual(block_y.data.shape, block_a.data.shape)
            np.testing.assert_allclose(block_y.data, block_a.data, atol=1e-3)


class BuilderEndToEndTests(unittest.TestCase):
    def test_prepare_real_manifest_feeds_classifier_dry_run(self) -> None:
        nchan, nsblk, nsub = 64, 256, 16
        nsamp = nsblk * nsub
        tsamp = 1e-3
        freqs_desc = np.linspace(1500.0, 1000.0, nchan)
        freqs_asc = freqs_desc[::-1].copy()
        rng = np.random.default_rng(3)

        frb_asc = rng.normal(0.0, 1.0, size=(nchan, nsamp))
        _inject_burst(
            frb_asc,
            freqs_asc,
            tsamp,
            toa_top_s=2.0,
            dm=50.0,
            amplitude=8.0,
            width_samples=16,
        )
        bandpass = 30.0 + 0.4 * np.arange(nchan)[:, None]
        frb_file = (frb_asc + bandpass)[::-1].astype(np.float32)

        rfi_asc = rng.normal(0.0, 1.0, size=(nchan, nsamp))
        rfi_asc[30] += 40.0 * np.sin(2.0 * np.pi * 5.0 * np.arange(nsamp) * tsamp)
        rfi_file = (rfi_asc + bandpass)[::-1].astype(np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frb_path = root / "frb.fits"
            rfi_path = root / "rfi.fits"
            _write_synthetic_psrfits(
                frb_path,
                float_data_file_order=frb_file,
                freqs_file_order=freqs_desc,
                tsamp_s=tsamp,
                nsblk=nsblk,
            )
            _write_synthetic_psrfits(
                rfi_path,
                float_data_file_order=rfi_file,
                freqs_file_order=freqs_desc,
                tsamp_s=tsamp,
                nsblk=nsblk,
            )

            input_manifest = root / "real_manifest.csv"
            with input_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["fits_path", "label", "toa", "dm", "source"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "fits_path": str(frb_path),
                        "label": "FRB",
                        "toa": "2.0",
                        "dm": "50.0",
                        "source": "SYNTH_FRB",
                    }
                )
                writer.writerow(
                    {
                        "fits_path": str(rfi_path),
                        "label": "RFI",
                        "toa": "",
                        "dm": "",
                        "source": "",
                    }
                )

            config = RealPrepConfig(
                input_manifest=input_manifest,
                output_dir=Path("dataset_real"),
                window_seconds=1.0,
                time_bins=128,
                freq_bins=32,
                width=256,
                height=192,
                dpi=96,
            )
            preparer = RealDatasetPreparer(config=config, project_root=root)
            records = preparer.prepare(overwrite=True)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["sample_id"], "sample_000000")
            self.assertEqual(records[0]["true_label"], "FRB")
            self.assertEqual(records[0]["source_sample_id"], "frb")
            self.assertEqual(records[1]["true_label"], "RFI")
            self.assertGreaterEqual(records[1]["preprocessing"]["channels_zapped"], 1)
            for record in records:
                for key in ("sample_id", "image_path", "true_label", "simulation_parameters"):
                    self.assertIn(key, record)
                self.assertTrue((root / record["image_path"]).exists())
                lower_path = record["image_path"].lower()
                self.assertNotIn("frb", lower_path)
                self.assertNotIn("rfi", lower_path)

            window_start = records[0]["preprocessing"]["window_start_seconds"]
            window_len = records[0]["preprocessing"]["window_seconds_actual"]
            self.assertLess(window_start, 2.0)
            self.assertGreater(window_start + window_len, 2.0)

            manifest_lines = (
                (root / "dataset_real" / "metadata" / "image_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
                .splitlines()
            )
            self.assertEqual(len(manifest_lines), 2)
            self.assertEqual(json.loads(manifest_lines[0])["sample_id"], "sample_000000")

            classifier = VLMClassifier(
                config=ClassifierConfig(
                    manifest_path=preparer.manifest_path,
                    output_path=root / "results" / "predictions.jsonl",
                    task="frb-binary",
                ),
                model=DryRunVLM(seed=7, labels=classes_for_task("frb-binary")),
                project_root=root,
            )
            predictions = classifier.run(overwrite=True)

            self.assertEqual(len(predictions), 2)
            self.assertIsNone(predictions[0]["error"])
            self.assertIsNone(predictions[1]["error"])
            self.assertEqual(predictions[0]["true_label"], "FRB")
            self.assertEqual(predictions[1]["true_label"], "NON_FRB")
            self.assertEqual(predictions[1]["source_true_label"], "RFI")
            self.assertIn("frb_probability", predictions[0]["parsed_response"])


if __name__ == "__main__":
    unittest.main()
