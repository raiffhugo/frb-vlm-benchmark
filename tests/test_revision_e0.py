from __future__ import annotations

import unittest

from revision_v4.e0_rebuild import (
    compare_manifests,
    compare_responses,
    select_check_records,
)


def _records(n: int) -> list[dict[str, object]]:
    return [
        {"sample_id": f"sample_{index:06d}", "true_label": "FRB" if index % 2 else "RFI"}
        for index in range(n)
    ]


class E0RebuildTests(unittest.TestCase):
    def test_identical_manifests_pass(self) -> None:
        records = _records(3)

        result = compare_manifests(records, [dict(record) for record in records])

        self.assertTrue(result["identical"])
        self.assertEqual(result["mismatched_records"], 0)

    def test_manifest_mismatch_and_length_are_reported(self) -> None:
        reference = _records(3)
        rebuilt = [dict(record) for record in reference]
        rebuilt[1]["true_label"] = "NOISE"

        result = compare_manifests(reference, rebuilt)
        shorter = compare_manifests(reference, rebuilt[:2])

        self.assertFalse(result["identical"])
        self.assertEqual(result["first_mismatches"], ["sample_000001"])
        self.assertFalse(shorter["identical"])

    def test_check_selection_is_deterministic_and_order_free(self) -> None:
        records = _records(200)

        first = select_check_records(records, n=50, seed=42)
        shuffled = select_check_records(list(reversed(records)), n=50, seed=42)

        self.assertEqual(len(first), 50)
        self.assertEqual(first, shuffled)
        self.assertEqual(
            [record["sample_id"] for record in first],
            sorted(record["sample_id"] for record in first),
        )

    def test_response_comparison_requires_exact_raw_text(self) -> None:
        reference = [
            {"sample_id": "a", "raw_model_response": "{x}", "predicted_label": "FRB"},
            {"sample_id": "b", "raw_model_response": "{y}", "predicted_label": "NON_FRB"},
        ]
        same = compare_responses(reference, [dict(record) for record in reference])
        changed = compare_responses(
            reference,
            [
                dict(reference[0]),
                {"sample_id": "b", "raw_model_response": "{y} ", "predicted_label": "NON_FRB"},
            ],
        )

        self.assertTrue(same["passed"])
        self.assertFalse(changed["passed"])
        self.assertEqual(changed["raw_identical"], 1)
        self.assertEqual(changed["label_identical"], 2)
        self.assertEqual(changed["mismatches"][0]["sample_id"], "b")


if __name__ == "__main__":
    unittest.main()
