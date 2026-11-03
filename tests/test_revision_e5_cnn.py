from __future__ import annotations

import unittest

import numpy as np

from revision_v4.e3_tiling import Sweep
from revision_v4.e5_cnn import (
    SIGMA_RANGE,
    WIDTH_RANGE_MS,
    WINDOWS_PER_FOLD,
    assign_positives,
    c4_rule,
    discounted_c4,
    file_folds,
    fold_plan,
    off_burst,
    threshold_at_fpr,
    window_rates,
)
from revision_v4.injection import P0


def _layouts(files: list[str]) -> dict[str, dict]:
    return {
        f: {"nsamp_total": 61440, "tsamp_s": 9.8304e-05, "f_low_mhz": 1000.0, "f_high_mhz": 1499.88,
            "visible_low_mhz": 1050.0, "visible_high_mhz": 1449.8}
        for f in files
    }


class PlanTests(unittest.TestCase):
    def test_files_split_in_two_halves(self) -> None:
        files = [f"fast_frex/{i:05d}_neg_sample.fits" for i in range(1000)]

        folds = file_folds(files)

        self.assertEqual(sorted(folds), files)
        self.assertEqual(sum(folds.values()), 500)

    def test_fold_plan_ranges_pairs_and_validation(self) -> None:
        files = [f"f{i}" for i in range(10)]

        plan = fold_plan(1, files, _layouts(files), P0())

        self.assertEqual(len(plan), WINDOWS_PER_FOLD)
        self.assertEqual(sum(e["split"] == "validation" for e in plan), WINDOWS_PER_FOLD // 10)
        sigmas = np.array([e["target_sigma"] for e in plan])
        widths = np.array([e["fwhm_s"] for e in plan]) * 1e3
        self.assertTrue(SIGMA_RANGE[0] <= sigmas.min() and sigmas.max() <= SIGMA_RANGE[1])
        self.assertTrue(WIDTH_RANGE_MS[0] <= widths.min() and widths.max() <= WIDTH_RANGE_MS[1])
        self.assertAlmostEqual(np.mean([e["band_fwhm_mhz"] is None for e in plan]), 0.5, delta=0.03)
        names = [e["injected_sample_id"] for e in plan] + [e["control_sample_id"] for e in plan]
        self.assertEqual(len(set(names)), 2 * WINDOWS_PER_FOLD)
        self.assertEqual(plan, fold_plan(1, files, _layouts(files), P0()))

    def test_positives_alternate_within_each_stratum(self) -> None:
        sigma = {f"p{i:03d}": s for i, s in enumerate([2.0] * 11 + [6.0] * 10 + [40.0] * 9)}

        assignment = assign_positives(sigma)

        for value in (2.0, 6.0, 40.0):
            members = [assignment[i] for i, s in sigma.items() if s == value]
            self.assertLessEqual(abs(sum(members) - len(members) / 2), 0.5)


class AnalysisTests(unittest.TestCase):
    def test_threshold_keeps_fpr_under_target(self) -> None:
        scores = np.array([0.01, 0.02, 0.05, 0.2, 0.9])

        threshold = threshold_at_fpr(scores, 0.2)

        self.assertEqual(threshold, 0.9)
        self.assertLessEqual(np.mean(scores >= threshold), 0.2)

    def test_c4_rule_needs_ten_points_and_significance(self) -> None:
        reference = [True] * 40 + [False] * 60
        vlm = [False] * 25 + [True] * 15 + [False] * 60

        rule = c4_rule(reference, vlm)

        self.assertAlmostEqual(rule["delta_recall"]["estimate"], 0.25)
        self.assertTrue(rule["passes"])
        self.assertFalse(c4_rule(reference, reference)["passes"])


class OffBurstControlTests(unittest.TestCase):
    def test_off_burst_needs_the_margin_on_both_sides(self) -> None:
        sweep = Sweep(2.0, 3.0)

        def window(start: float) -> dict:
            return {"window_start_seconds": start, "window_stop_seconds": start + 1.0}

        self.assertTrue(off_burst(window(0.4), sweep))   # termina em 1,4 < 1,5
        self.assertFalse(off_burst(window(0.6), sweep))  # termina em 1,6, dentro da folga
        self.assertFalse(off_burst(window(2.2), sweep))  # toca o sweep
        self.assertFalse(off_burst(window(3.4), sweep))  # comeca dentro da folga
        self.assertTrue(off_burst(window(3.6), sweep))

    def test_window_rates_pool_windows_and_resample_files(self) -> None:
        off = [("a", True), ("a", False), ("b", False), ("b", False)]
        negative = [(f"n{i}", i == 0) for i in range(8)]

        rates = window_rates(off, negative, iterations=2000, seed=1)

        self.assertEqual((rates["off_burst"]["files"], rates["off_burst"]["windows"]), (2, 4))
        self.assertAlmostEqual(rates["off_burst"]["estimate"], 0.25)
        self.assertAlmostEqual(rates["negative"]["estimate"], 0.125)
        self.assertAlmostEqual(rates["off_burst_minus_negative"]["estimate"], 0.125)
        for name in ("off_burst", "negative", "off_burst_minus_negative"):
            self.assertLessEqual(rates[name]["low"], rates[name]["estimate"])
            self.assertGreaterEqual(rates[name]["high"], rates[name]["estimate"])

    def test_discounted_delta_decides_with_ten_points(self) -> None:
        kept = discounted_c4([True] * 9 + [False], [True] + [False] * 9, 0.1)
        lost = discounted_c4([True] * 3 + [False] * 7, [True] + [False] * 9, 0.2)

        self.assertAlmostEqual(kept["cnn_recall_discounted"], 0.8 / 0.9)
        self.assertTrue(kept["holds"])
        self.assertAlmostEqual(lost["delta_discounted"], 0.125 - 0.1)
        self.assertFalse(lost["holds"])


if __name__ == "__main__":
    unittest.main()
