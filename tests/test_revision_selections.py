from __future__ import annotations

import unittest

from revision_v4.selections import E5_QUOTAS, R200_QUOTAS, real_selections, stratum_of


def _population() -> tuple[dict[str, float], list[str]]:
    sigmas = [1.0] * 80 + [6.0] * 80 + [10.0] * 80 + [20.0] * 80 + [50.0] * 80
    positives = {f"pos_{i:03d}": s for i, s in enumerate(sigmas)}
    negatives = [f"neg_{i:04d}" for i in range(1000)]
    return positives, negatives


class SelectionTests(unittest.TestCase):
    def test_quotas_and_disjoint_audit(self) -> None:
        positives, negatives = _population()

        selections = real_selections(positives, negatives)

        e5 = selections["e5_humans"]
        e5_strata = [stratum_of(positives[i]) for i in e5 if i in positives]
        self.assertEqual(
            [e5_strata.count(s) for s in ("[0, 5)", "[5, 8)", "[8, 15)", "[15, 30)", ">= 30")],
            list(E5_QUOTAS),
        )
        self.assertEqual(sum(i.startswith("neg") for i in e5), 100)
        r200 = selections["e6_r200"]
        self.assertEqual(sum(i in positives for i in r200), sum(R200_QUOTAS))
        audit = set(selections["e4_audit"])
        self.assertEqual(len(audit), 200)
        self.assertFalse(audit & set(e5))

    def test_draws_are_reproducible(self) -> None:
        positives, negatives = _population()

        self.assertEqual(real_selections(positives, negatives), real_selections(positives, negatives[::-1]))


if __name__ == "__main__":
    unittest.main()
