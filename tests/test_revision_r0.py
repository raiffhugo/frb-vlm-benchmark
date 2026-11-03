from __future__ import annotations

import unittest

import numpy as np

from revision_v4.r0_reanalysis import (
    cross_fitted,
    rates,
    stratified_folds,
    threshold_matching_fpr,
    threshold_matching_recall,
    threshold_max_accuracy,
)


Y = np.array([True, True, True, True, False, False, False, False])
SCORES = np.array([0.99, 0.9, 0.8, 0.3, 0.85, 0.4, 0.2, 0.1])


class ThresholdMatchingTests(unittest.TestCase):
    def test_fpr_match_takes_smallest_threshold_within_target(self) -> None:
        # FPR <= 0,25 admite um negativo acima do limiar: 0,85 -> recall 3/4
        self.assertEqual(threshold_matching_fpr(Y, SCORES, 0.25), 0.8)
        self.assertEqual(threshold_matching_fpr(Y, SCORES, 0.0), 0.9)

    def test_recall_match_takes_largest_threshold_reaching_target(self) -> None:
        self.assertEqual(threshold_matching_recall(Y, SCORES, 0.75), 0.8)
        self.assertEqual(threshold_matching_recall(Y, SCORES, 1.0), 0.3)

    def test_max_accuracy_threshold_breaks_ties_toward_recall(self) -> None:
        # 0,3, 0,8 e 0,9 acertam 6 de 8; fica o menor
        self.assertEqual(threshold_max_accuracy(Y, SCORES), 0.3)

    def test_cross_fitting_applies_each_threshold_to_the_other_fold(self) -> None:
        folds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        seen = []

        def choose(y_sel, s_sel, mask):  # type: ignore[no-untyped-def]
            seen.append(mask.copy())
            return 0.5 if mask[0] else 0.95

        predicted, thresholds = cross_fitted(Y, SCORES, folds, choose)

        self.assertEqual(thresholds, [0.5, 0.95])
        # fold 1 avaliado com 0,5 (escolhido no fold 0) e fold 0 com 0,95
        np.testing.assert_array_equal(predicted[folds == 1], SCORES[folds == 1] >= 0.5)
        np.testing.assert_array_equal(predicted[folds == 0], SCORES[folds == 0] >= 0.95)


class FoldTests(unittest.TestCase):
    def test_folds_are_balanced_within_strata_and_deterministic(self) -> None:
        ids = [f"sample_{i:06d}" for i in range(12)]
        strata = ["FRB"] * 6 + ["RFI"] * 4 + ["NOISE"] * 2

        folds = stratified_folds(ids, strata)
        again = stratified_folds(list(reversed(ids)), list(reversed(strata)))

        for stratum in ("FRB", "RFI", "NOISE"):
            members = [f for f, s in zip(folds, strata) if s == stratum]
            self.assertEqual(sum(members), len(members) // 2)
        np.testing.assert_array_equal(folds, again[::-1])


class RatesTests(unittest.TestCase):
    def test_rates_counts_and_intervals(self) -> None:
        result = rates(Y, SCORES >= 0.5)

        self.assertEqual((result["tp"], result["fn"], result["fp"], result["tn"]), (3, 1, 1, 3))
        self.assertAlmostEqual(result["accuracy"]["estimate"], 0.75)
        self.assertAlmostEqual(result["recall"]["estimate"], 0.75)
        self.assertAlmostEqual(result["fpr"]["estimate"], 0.25)
        self.assertAlmostEqual(result["precision_frb"]["estimate"], 0.75)


if __name__ == "__main__":
    unittest.main()
