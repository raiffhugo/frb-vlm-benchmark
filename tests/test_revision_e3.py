from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_real.preprocess import dispersion_delay_s
from revision_v4.e3_tiling import (
    FileWindows,
    Sweep,
    catalog_sweeps,
    compare_p0,
    group_windows,
    sigma_bin_of,
    tiling_metrics,
    window_contains,
)


def _window(sample_id: str, source_id: str, index: int, start: float, label: str) -> dict:
    return {
        "sample_id": sample_id,
        "source_sample_id": source_id,
        "window_index": index,
        "window_start_seconds": start,
        "window_stop_seconds": start + 2.0,
        "true_label": label,
    }


def _prediction(sample_id: str, probability: float | None) -> dict:
    if probability is None:
        return {"sample_id": sample_id, "error": "parse", "parsed_response": None}
    return {"sample_id": sample_id, "error": None, "parsed_response": {"frb_probability": probability}}


class SweepTests(unittest.TestCase):
    def test_full_band_sweep_starts_at_toa_and_spans_the_whole_band(self) -> None:
        freqs = np.linspace(1000.0, 1500.0, 11)

        sweeps = catalog_sweeps(toa_s=1.0, dm=500.0, frequencies_mhz=freqs, edge_channels=1)

        full = sweeps["full_band"]
        self.assertEqual(full.start, 1.0)
        self.assertAlmostEqual(full.stop, 1.0 + dispersion_delay_s(500.0, 1000.0, 1500.0))
        visible = sweeps["visible_band"]
        self.assertGreater(visible.start, full.start)
        self.assertLess(visible.stop, full.stop)
        self.assertAlmostEqual(visible.stop, 1.0 + dispersion_delay_s(500.0, 1050.0, 1500.0))

    def test_window_must_hold_the_whole_sweep(self) -> None:
        window = {"window_start_seconds": 0.0, "window_stop_seconds": 2.013}

        self.assertTrue(window_contains(window, Sweep(0.5, 1.8)))
        self.assertFalse(window_contains(window, Sweep(0.5, 2.1)))
        self.assertFalse(window_contains(window, Sweep(-0.1, 1.0)))

    def test_sigma_bins_follow_the_protocol_edges(self) -> None:
        self.assertEqual(sigma_bin_of(4.99), "[0, 5)")
        self.assertEqual(sigma_bin_of(5.0), "[5, 8)")
        self.assertEqual(sigma_bin_of(300.0), ">= 300")


class TilingMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest = [
            _window("w0", "pos_a", 0, 0.0, "FRB"),
            _window("w1", "pos_a", 1, 1.0, "FRB"),
            _window("w2", "pos_b", 0, 0.0, "FRB"),
            _window("w3", "pos_b", 1, 1.0, "FRB"),
            _window("w4", "neg_a", 0, 0.0, "RFI"),
            _window("w5", "neg_a", 1, 1.0, "RFI"),
            _window("w6", "neg_b", 0, 0.0, "RFI"),
            _window("w7", "neg_b", 1, 1.0, "RFI"),
        ]
        predictions = [
            _prediction("w0", 0.1),
            _prediction("w1", 0.9),  # pos_a: detectado so numa janela sem o sweep
            _prediction("w2", 0.85),  # pos_b: detectado na janela com o sweep
            _prediction("w3", None),
            _prediction("w4", 0.6),
            _prediction("w5", 0.1),
            _prediction("w6", 0.2),
            _prediction("w7", 0.3),
        ]
        self.files = group_windows(manifest, predictions)
        self.sweeps = {
            "pos_a": {"full_band": Sweep(0.2, 1.5), "visible_band": Sweep(0.3, 1.4)},
            "pos_b": {"full_band": Sweep(0.5, 1.9), "visible_band": Sweep(0.6, 1.8)},
        }
        self.centered = {"pos_a": True, "pos_b": True, "neg_a": True, "neg_b": True}

    def test_any_window_and_time_matched_recall_differ(self) -> None:
        metrics = tiling_metrics(self.files, self.sweeps, self.centered)

        self.assertEqual(metrics["recall_any_window"]["k"], 2)
        self.assertEqual(metrics["full_band"]["recall_time_matched"]["k"], 1)
        self.assertEqual(metrics["full_band"]["fraction_with_full_sweep_window"]["k"], 2)
        self.assertEqual(metrics["invalid_windows"], 1)

    def test_false_positive_rates_per_file_and_per_window(self) -> None:
        metrics = tiling_metrics(self.files, self.sweeps, self.centered)

        self.assertEqual((metrics["fpr_per_file"]["k"], metrics["fpr_per_file"]["n"]), (1, 2))
        self.assertEqual((metrics["fpr_per_window"]["k"], metrics["fpr_per_window"]["n"]), (1, 4))

    def test_paired_comparison_with_the_centered_window(self) -> None:
        metrics = tiling_metrics(self.files, self.sweeps, self.centered)

        paired = metrics["paired_vs_centered"]
        self.assertEqual(paired["recall_centered"]["k"], 2)
        self.assertEqual(paired["time_matched_minus_centered"]["mcnemar"]["only_second_correct"], 1)

    def test_file_without_valid_window_is_reported_and_excluded(self) -> None:
        files = dict(self.files)
        files["neg_c"] = FileWindows("neg_c", False, ({"window_index": 0},), (None,))

        metrics = tiling_metrics(files, self.sweeps, self.centered)

        self.assertEqual(metrics["files_without_valid_window"], ["neg_c"])
        self.assertEqual(metrics["fpr_per_file"]["n"], 2)

    def test_predictions_must_cover_the_manifest(self) -> None:
        with self.assertRaises(ValueError):
            group_windows([_window("w0", "pos_a", 0, 0.0, "FRB")], [])


class CheckP0Tests(unittest.TestCase):
    def test_flags_png_and_field_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, payload in (("old.png", b"a"), ("same.png", b"a"), ("other.png", b"b")):
                (root / name).write_bytes(payload)
            base = {
                "sample_id": "sample_000000",
                "fits_path": "fast_frex/x.fits",
                "source_sample_id": "x",
                "true_label": "FRB",
                "real_parameters": {"dm": 1.0},
                "preprocessing": {"zap_sigma": 3.0},
            }
            paper = [{**base, "image_path": "old.png"}]

            same = compare_p0(paper, [{**base, "image_path": "same.png", "extra": 1}], project_root=root)
            other = compare_p0(
                paper,
                [{**base, "image_path": "other.png", "preprocessing": {"zap_sigma": 5.0}}],
                project_root=root,
            )

            self.assertTrue(same["passed"])
            self.assertFalse(other["passed"])
            self.assertEqual(other["mismatches"][0]["problems"], ["preprocessing.zap_sigma", "png"])


if __name__ == "__main__":
    unittest.main()
