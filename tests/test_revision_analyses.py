from __future__ import annotations

import unittest

import numpy as np

from revision_v4.e1_analysis import Run
from revision_v4.e2_analysis import fit_logistic, sigma50, sigma50_with_bootstrap
from revision_v4.e4_preprocessing import multiplies_false_positives
from revision_v4.e6_analysis import paraphrase_range
from revision_v4.e8_decimated import explains_difference
from revision_v4.paired_runs import recall_by_bin, recall_by_bin_paired, restrict, unpaired_by_bin, versus


def _run(y: list[bool], hits: list[bool], keys: list[str] | None = None) -> Run:
    ids = tuple(f"s{i:04d}" for i in range(len(y)))
    return Run(
        sample_ids=ids,
        source_ids=tuple(keys) if keys else ids,
        y_true=np.array(y),
        scores=np.where(np.array(hits), 0.9, 0.1),
        source_labels=tuple("FRB" if v else "NOISE" for v in y),
    )


class PairedRunTests(unittest.TestCase):
    def test_versus_counts_false_positives_on_negatives_only(self) -> None:
        y = [True] * 4 + [False] * 6
        first = _run(y, [True, True, False, False] + [True, True, False, False, False, False])
        second = _run(y, [True, True, True, False] + [False] * 6)

        result = versus(first, second, bootstrap=False)

        self.assertEqual(result["false_positives"], {"first": 2, "second": 0, "n": 6})
        self.assertEqual(result["frb_recall"]["mcnemar"]["only_second_correct"], 1)
        self.assertEqual(result["specificity"]["mcnemar"]["only_second_correct"], 2)

    def test_recall_by_bin_uses_the_sigma_of_each_burst(self) -> None:
        y = [True, True, True, False]
        run = _run(y, [True, False, True, True], keys=["a", "b", "c", "n"])
        other = _run(y, [False, False, True, False], keys=["a", "b", "c", "n"])
        sigma = {"a": 3.0, "b": 3.5, "c": 400.0}

        by_bin = recall_by_bin(run, sigma)
        paired = recall_by_bin_paired(run, other, sigma)

        self.assertEqual(by_bin["[0, 5)"]["n"], 2)
        self.assertAlmostEqual(by_bin["[0, 5)"]["estimate"], 0.5)
        self.assertEqual(by_bin[">= 300"]["estimate"], 1.0)
        self.assertEqual(paired["[0, 5)"]["first_minus_second"]["mcnemar"]["only_first_correct"], 1)

    def test_unpaired_by_bin_skips_small_bins(self) -> None:
        first = {"[0, 5)": {"n": 50, "estimate": 0.9, "low": 0.8, "high": 0.96}, ">= 300": {"n": 5, "estimate": 1.0, "low": 0.5, "high": 1.0}}
        second = {"[0, 5)": {"n": 40, "estimate": 0.4, "low": 0.26, "high": 0.55}, ">= 300": {"n": 40, "estimate": 1.0, "low": 0.9, "high": 1.0}}

        result = unpaired_by_bin(first, second, 10)

        self.assertEqual(list(result), ["[0, 5)"])
        self.assertTrue(result["[0, 5)"]["significant"])

    def test_restrict_keeps_order_and_labels(self) -> None:
        run = _run([True, False, True], [True, True, False])

        kept = restrict(run, {"s0002", "s0000"})

        self.assertEqual(kept.sample_ids, ("s0000", "s0002"))
        self.assertEqual(kept.y_true.tolist(), [True, True])


class DecisionRuleTests(unittest.TestCase):
    def test_p2_multiplies_false_positives_only_with_a_significant_increase(self) -> None:
        y = [False] * 1000
        p0 = _run(y, [i < 3 for i in range(1000)])
        many = _run(y, [i < 20 for i in range(1000)])
        few = _run(y, [i < 5 for i in range(1000)])

        self.assertTrue(multiplies_false_positives(p0, many)["multiplies"])
        verdict = multiplies_false_positives(p0, few)
        self.assertGreaterEqual(verdict["ratio"], 1.6)
        self.assertFalse(verdict["multiplies"])

    def test_e8_explains_only_if_the_gap_closes_in_every_bin(self) -> None:
        original = {"a": {"significant": True}, "b": {"significant": False}}

        self.assertTrue(explains_difference(original, {"a": {"significant": False}, "b": {"significant": False}})["explains"])
        self.assertFalse(explains_difference(original, {"a": {"significant": False}, "b": {"significant": True}})["explains"])
        self.assertFalse(explains_difference({"a": {"significant": False}}, {"a": {"significant": False}})["explains"])

    def test_paraphrase_range(self) -> None:
        def entry(accuracy: float) -> dict:
            value = {"estimate": accuracy}
            return {"binary_0.5": {"accuracy": value, "recall": value, "fpr": value}}

        summaries = {"pa": entry(0.8), "pb": entry(0.9), "pc": entry(0.85), "pd": entry(0.1)}

        self.assertEqual(paraphrase_range(summaries)["accuracy"], {"min": 0.8, "max": 0.9})


class LogisticTests(unittest.TestCase):
    def test_sigma50_recovers_the_true_completeness_point(self) -> None:
        rng = np.random.default_rng(0)
        sigma = 10 ** rng.uniform(0.3, 1.8, 4000)
        p = 1 / (1 + np.exp(-(8.0 * (np.log10(sigma) - 1.0))))
        detected = rng.random(sigma.size) < p

        self.assertAlmostEqual(sigma50(sigma, detected), 10.0, delta=0.5)

    def test_fit_fails_on_a_decreasing_or_separated_curve(self) -> None:
        x = np.linspace(0, 1, 50)
        self.assertIsNone(fit_logistic(x, (x < 0.5).astype(float)))
        self.assertIsNone(fit_logistic(x, (x > 0.5).astype(float)))

    def test_bootstrap_interval_contains_the_point(self) -> None:
        rng = np.random.default_rng(1)
        sigma = 10 ** rng.uniform(0.3, 1.8, 300)
        detected = rng.random(300) < 1 / (1 + np.exp(-(6.0 * (np.log10(sigma) - 1.0))))

        result = sigma50_with_bootstrap(sigma, detected, iterations=200, seed=3)

        self.assertLessEqual(result["low"], result["sigma50"])
        self.assertLessEqual(result["sigma50"], result["high"])
        self.assertEqual(result["iterations"], 200)


if __name__ == "__main__":
    unittest.main()
