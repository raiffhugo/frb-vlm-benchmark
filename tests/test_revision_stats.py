from __future__ import annotations

import math
import unittest

import numpy as np

from revision_v4.stats import (
    PairedCounts,
    bootstrap_paired_difference,
    mcnemar_chi2_corrected,
    mcnemar_exact,
    newcombe_paired_difference,
    newcombe_unpaired_difference,
    wilson,
)


class WilsonTests(unittest.TestCase):
    def test_closed_form_limits(self) -> None:
        zero = wilson(0, 10)
        full = wilson(10, 10)

        self.assertEqual(zero.low, 0.0)
        self.assertAlmostEqual(zero.high, 0.2775, places=4)
        self.assertAlmostEqual(full.low, 1 - 0.2775, places=4)
        self.assertEqual(full.high, 1.0)

    def test_matches_the_v3_real_data_interval(self) -> None:
        # Tabela 12 da v3: recall do E2B nos 600 bursts reais, 171/600 -> [0,250; 0,322]
        interval = wilson(171, 600)

        self.assertAlmostEqual(interval.estimate, 0.285)
        self.assertAlmostEqual(interval.low, 0.250, places=3)
        self.assertAlmostEqual(interval.high, 0.322, places=3)

    def test_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            wilson(1, 0)
        with self.assertRaises(ValueError):
            wilson(5, 4)


class PairedDifferenceTests(unittest.TestCase):
    def test_counts_from_outcomes(self) -> None:
        counts = PairedCounts.from_outcomes(
            [True, True, False, False, True], [True, False, True, False, False]
        )

        self.assertEqual((counts.a, counts.b, counts.c, counts.d), (1, 2, 1, 1))

    def test_newcombe_brackets_estimate_and_approaches_wald(self) -> None:
        counts = PairedCounts(a=1800, b=60, c=36, d=104)
        interval = newcombe_paired_difference(counts)
        n = counts.n
        theta = (counts.b - counts.c) / n
        wald_se = math.sqrt((counts.b + counts.c) / n - theta * theta) / math.sqrt(n)

        self.assertAlmostEqual(interval.estimate, theta)
        self.assertLess(interval.low, theta)
        self.assertGreater(interval.high, theta)
        self.assertAlmostEqual(interval.low, theta - 1.96 * wald_se, delta=0.002)
        self.assertAlmostEqual(interval.high, theta + 1.96 * wald_se, delta=0.002)

    def test_newcombe_identical_classifiers_gives_zero_centered_interval(self) -> None:
        interval = newcombe_paired_difference(PairedCounts(a=90, b=0, c=0, d=10))

        self.assertEqual(interval.estimate, 0.0)
        self.assertAlmostEqual(interval.low, -interval.high)

    def test_bootstrap_is_reproducible_and_close_to_newcombe(self) -> None:
        rng = np.random.default_rng(0)
        first = rng.random(2000) < 0.94
        second = np.where(rng.random(2000) < 0.9, first, ~first)

        once = bootstrap_paired_difference(first, second, iterations=2000)
        again = bootstrap_paired_difference(first, second, iterations=2000)
        newcombe = newcombe_paired_difference(PairedCounts.from_outcomes(first, second))

        self.assertEqual(once, again)
        self.assertAlmostEqual(once.low, newcombe.low, delta=0.004)
        self.assertAlmostEqual(once.high, newcombe.high, delta=0.004)


class UnpairedDifferenceTests(unittest.TestCase):
    def test_matches_newcombe_1998_worked_example(self) -> None:
        # Newcombe (1998), Stat. Med. 17, 873, tabela II, exemplo (a): 56/70 - 48/80.
        interval = newcombe_unpaired_difference(wilson(56, 70), wilson(48, 80))

        self.assertAlmostEqual(interval.estimate, 0.2)
        self.assertAlmostEqual(interval.low, 0.0524, places=4)
        self.assertAlmostEqual(interval.high, 0.3339, places=4)


class McNemarTests(unittest.TestCase):
    def test_exact_p_value_is_binomial_on_discordant_pairs(self) -> None:
        result = mcnemar_exact(PairedCounts(a=10, b=1, c=9, d=10))

        # 2 * P(X <= 1 | n=10, p=1/2) = 2 * 11/1024
        self.assertAlmostEqual(result["p_value_exact"], 22 / 1024)
        self.assertEqual(result["discordant"], 10)

    def test_no_discordance_gives_p_one(self) -> None:
        counts = PairedCounts(a=5, b=0, c=0, d=5)

        self.assertEqual(mcnemar_exact(counts)["p_value_exact"], 1.0)
        self.assertEqual(mcnemar_chi2_corrected(counts), 1.0)


if __name__ == "__main__":
    unittest.main()
