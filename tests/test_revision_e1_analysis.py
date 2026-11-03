from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from revision_v4.e1_analysis import (
    Run,
    hybrids_transfer,
    load_run,
    operating_points_transfer,
    prompt_variants,
    recall_by_sigma_bin,
)


def _run(scores: list[float], y: list[bool], prefix: str = "s") -> Run:
    n = len(scores)
    return Run(
        sample_ids=tuple(f"{prefix}{i:03d}" for i in range(n)),
        source_ids=tuple(f"src{i:03d}" for i in range(n)),
        y_true=np.array(y),
        scores=np.array(scores, dtype=float),
        source_labels=tuple("FRB" if label else "RFI" for label in y),
    )


class RunTests(unittest.TestCase):
    def test_failed_answer_becomes_the_wrong_decision(self) -> None:
        run = _run([0.9, np.nan, np.nan, 0.1], [True, True, False, False])

        np.testing.assert_array_equal(run.predict(0.5), [True, False, True, False])

    def test_load_run_marks_unparsed_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            records = [
                {"sample_id": "a", "true_label": "FRB", "source_true_label": "FRB",
                 "error": None, "parsed_response": {"frb_probability": 0.85}},
                {"sample_id": "b", "true_label": "NON_FRB", "source_true_label": "NOISE",
                 "error": "ValueError: Resposta sem objeto JSON.", "parsed_response": None},
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            run = load_run(path, {"a": "frb_0", "b": "noise_0"})

            self.assertEqual(run.source_ids, ("frb_0", "noise_0"))
            self.assertEqual(run.scores[0], 0.85)
            self.assertTrue(np.isnan(run.scores[1]))
            self.assertEqual(run.source_labels, ("FRB", "NOISE"))


class TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        y = [True] * 4 + [False] * 4
        # VLM quantizado; SwinYNet continuo e quase separavel
        self.vlm_select = _run([0.85, 0.85, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05], y)
        self.swin_select = _run([0.99, 0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05], y)
        self.vlm_apply = _run([0.85, 0.25, 0.85, 0.05, 0.05, 0.25, 0.05, 0.05], y, "t")
        self.swin_apply = _run([0.95, 0.85, 0.75, 0.65, 0.3, 0.2, 0.1, 0.0], y, "t")

    def test_thresholds_come_from_the_selection_set(self) -> None:
        rows = operating_points_transfer(
            self.vlm_select, self.swin_select, self.vlm_apply, self.swin_apply
        )

        by_level = {row["vlm_level"]: row for row in rows}
        # no nivel 0,85 o VLM tem FPR 0 na selecao: o SwinYNet precisa de limiar > 0,6
        self.assertEqual(by_level[0.85]["fpr_matched"]["target_on_selection"], 0.0)
        self.assertEqual(by_level[0.85]["fpr_matched"]["swinynet_threshold"], 0.7)
        # aplicado ao outro conjunto, o limiar da selecao perde o FRB de score 0,65
        self.assertEqual(by_level[0.85]["fpr_matched"]["swinynet"]["tp"], 3)
        # no nivel 0,25 o recall alvo e 1,0: o maior limiar com recall total e 0,7
        self.assertEqual(by_level[0.25]["recall_matched"]["swinynet_threshold"], 0.7)

    def test_hybrids_use_the_swinynet_threshold_of_the_selection_set(self) -> None:
        result = hybrids_transfer(self.vlm_select, self.swin_select, self.vlm_apply, self.swin_apply)

        self.assertEqual(result["fixed_0.5"]["swinynet_threshold"], 0.5)
        self.assertEqual(result["max_accuracy"]["swinynet_threshold"], 0.7)
        self.assertEqual(result["max_accuracy"]["swinynet_alone"]["accuracy"]["estimate"], 0.875)
        self.assertEqual(result["max_accuracy"]["and"]["hybrid"]["tp"], 2)


class PromptVariantTests(unittest.TestCase):
    def test_pairs_and_faint_subset(self) -> None:
        y = [True, True, True, False]
        arms = {
            "binary_full": _run([0.85, 0.85, 0.25, 0.05], y),
            "binary_lean": _run([0.85, 0.25, 0.25, 0.05], y),
            "binary_lean_reason": _run([0.85, 0.85, 0.85, 0.05], y),
        }
        sigma = {"src000": 40.0, "src001": 10.0, "src002": 6.0}

        result = prompt_variants(arms, np.array([True, True, False, False]), sigma)

        self.assertEqual(result["arms"]["binary_lean_reason"]["binary"]["tp"], 3)
        pair = result["pairs"]["binary_lean_reason_minus_binary_lean"]
        self.assertEqual(pair["frb_recall"]["mcnemar"]["only_first_correct"], 2)
        self.assertEqual(pair["frb_recall_faint_descriptive"]["mcnemar"]["discordant"], 2)
        self.assertIn("n = 2", result["faint_definition"])

    def test_recall_by_sigma_bin_uses_protocol_bins(self) -> None:
        run = _run([0.85, 0.05, 0.85, 0.05], [True, True, True, False])
        sigma = {"src000": 4.0, "src001": 4.5, "src002": 70.0}

        bins = recall_by_sigma_bin(run, sigma)

        self.assertEqual(list(bins), ["[0, 5)", "[60, 100)"])
        self.assertEqual(bins["[0, 5)"]["n"], 2)
        self.assertEqual(bins["[0, 5)"]["estimate"], 0.5)


if __name__ == "__main__":
    unittest.main()
