from __future__ import annotations

import json
import unittest
from pathlib import Path

from revision_v4.e1_heldout import composition, replay_sample_seeds, seed_checks


class E1HeldoutTests(unittest.TestCase):
    def test_replay_matches_the_benchmark_seeds(self) -> None:
        labels = Path("dataset/metadata/labels.jsonl")
        if not labels.exists():
            self.skipTest("benchmark local ausente")
        seeds = [json.loads(line)["seed"] for line in labels.open(encoding="utf-8")]

        self.assertEqual(replay_sample_seeds(42, len(seeds)), seeds)

    def test_seed_checks_flag_collisions_and_duplicates(self) -> None:
        e1 = [{"seed": seed} for seed in replay_sample_seeds(20260927, 5)]
        benchmark = [{"seed": e1[2]["seed"]}, {"seed": 1}]

        result = seed_checks(e1, benchmark)
        duplicated = seed_checks(e1 + [e1[0]], [])

        self.assertEqual(result["collisions_with_benchmark"], 1)
        self.assertEqual(result["internal_duplicates"], 0)
        self.assertTrue(result["replays_master_seed"])
        self.assertEqual(duplicated["internal_duplicates"], 1)
        self.assertFalse(duplicated["replays_master_seed"])

    def test_composition_counts_classes_and_rfi_subtypes(self) -> None:
        records = [
            {"true_label": "FRB", "simulation_parameters": {}},
            {"true_label": "RFI", "simulation_parameters": {"rfi_types": ["wifi"]}},
            {"true_label": "RFI", "simulation_parameters": {"rfi_types": ["wifi"]}},
            {"true_label": "NOISE", "simulation_parameters": {}},
        ]

        result = composition(records)

        self.assertEqual(result["by_class"], {"FRB": 1, "NOISE": 1, "RFI": 2})
        self.assertEqual(result["rfi_subtypes"], {"wifi": 2})


if __name__ == "__main__":
    unittest.main()
