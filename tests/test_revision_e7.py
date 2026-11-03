from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from revision_v4.e1_analysis import Run
from revision_v4.e7_prompts import FILLER_TOKENS, V2_FILE, build, filler
from revision_v4.e7_reasoning import ARMS, compare, evaluate_rule, select


ROOT = Path(__file__).resolve().parents[1]


def _schema(text: str) -> dict:
    return json.loads(text[text.rindex("{") :])


class E7PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.v2 = (ROOT / V2_FILE).read_text(encoding="utf-8")
        self.lean = (ROOT / "prompts_v4" / "binary_lean.txt").read_text(encoding="utf-8")
        self.prompts = build(self.v2)

    def test_v1o_is_the_lean_prompt_with_the_decision_fields_in_v2_order(self) -> None:
        v1o = self.prompts["e7_v1o_no_reason"]

        self.assertEqual(list(_schema(v1o)), ["frb_probability", "label"])
        self.assertEqual(list(_schema(self.lean)), ["label", "frb_probability"])
        self.assertEqual(v1o.splitlines()[:-1], self.lean.splitlines()[:-1])

    def test_variants_change_only_the_instruction_and_the_schema(self) -> None:
        body = self.v2[: self.v2.index("Describe the visual evidence first")]

        for name, text in self.prompts.items():
            with self.subTest(prompt=name):
                self.assertTrue(text.startswith(body))
                self.assertEqual(len(text.splitlines()), len(self.v2.splitlines()))
                self.assertFalse(text.endswith("\n"))
                self.assertEqual(list(_schema(text))[-2:] if name != "e7_v3_reason_after" else list(_schema(text))[:2],
                                 ["frb_probability", "label"])

    def test_reason_position_and_filler_length(self) -> None:
        self.assertEqual(list(_schema(self.prompts["e7_v3_reason_after"])), ["frb_probability", "label", "reason"])
        self.assertEqual(list(_schema(self.prompts["e7_v5_reason_reworded"]))[0], "reason")
        self.assertNotIn("Describe the visual evidence first", self.prompts["e7_v5_reason_reworded"])
        for model, tokens in FILLER_TOKENS.items():
            with self.subTest(model=model):
                schema = _schema(self.prompts[f"e7_v4_filler_{model}"])
                self.assertEqual(list(schema), ["filler", "frb_probability", "label"])
                self.assertEqual(schema["filler"], filler(tokens))
                self.assertEqual(schema["filler"].count("."), tokens)


def _records() -> list[dict]:
    records = [{"sample_id": f"f{i:04d}", "true_label": "FRB", "simulation_parameters": {}} for i in range(1000)]
    for k, subtype in enumerate(("a", "b", "c", "d", "e")):
        records += [
            {"sample_id": f"r{k}{i:03d}", "true_label": "RFI", "simulation_parameters": {"rfi_category": subtype}}
            for i in range(100)
        ]
    records += [{"sample_id": f"n{i:03d}", "true_label": "NOISE", "simulation_parameters": {}} for i in range(500)]
    return records


class E7SelectionTests(unittest.TestCase):
    def test_composition_and_reproducibility(self) -> None:
        records = _records()

        chosen = select(records)

        labels = [r["true_label"] for r in chosen]
        self.assertEqual((labels.count("FRB"), labels.count("RFI"), labels.count("NOISE")), (200, 100, 100))
        subtypes = [r["simulation_parameters"]["rfi_category"] for r in chosen if r["true_label"] == "RFI"]
        self.assertEqual({s: subtypes.count(s) for s in set(subtypes)}, {s: 20 for s in "abcde"})
        self.assertEqual([r["sample_id"] for r in select(records[::-1])], [r["sample_id"] for r in chosen])


def _run(hits: np.ndarray) -> Run:
    """200 FRB (acertos dados) e 200 NON_FRB todos rejeitados."""
    y = np.array([True] * 200 + [False] * 200)
    scores = np.concatenate([np.where(hits, 0.9, 0.1), np.full(200, 0.1)])
    ids = tuple(f"s{i:03d}" for i in range(400))
    return Run(ids, ids, y, scores, tuple("FRB" if v else "NOISE" for v in y))


def _arms(recall: dict[str, int]) -> dict[str, Run]:
    return {arm: _run(np.arange(200) < recall[arm]) for arm in ARMS}


class E7RuleTests(unittest.TestCase):
    BASE = {
        "v1_lean": 110, "v1o_no_reason": 120, "v2_reason_before": 180,
        "v3_reason_after": 125, "v4_filler": 122, "v5_reason_reworded": 175,
    }

    def test_claim_returns_when_only_the_justification_helps(self) -> None:
        rule = evaluate_rule(compare(_arms(self.BASE)), "v1o_no_reason")

        self.assertTrue(rule["causal_claim_returns"], rule)

    def test_claim_fails_when_the_filler_reproduces_the_gain(self) -> None:
        rule = evaluate_rule(compare(_arms({**self.BASE, "v4_filler": 178})), "v1o_no_reason")

        self.assertFalse(rule["conditions"]["v4_does_not_reproduce_gain_over_v1o_no_reason"])
        self.assertFalse(rule["causal_claim_returns"])

    def test_claim_fails_when_the_field_order_explains_the_gain(self) -> None:
        rule = evaluate_rule(compare(_arms({**self.BASE, "v1o_no_reason": 178, "v4_filler": 179})), "v1o_no_reason")

        self.assertTrue(rule["conditions"]["v2_gt_v1_lean"])
        self.assertFalse(rule["conditions"]["v2_gt_v1o_no_reason"])
        self.assertFalse(rule["causal_claim_returns"])


if __name__ == "__main__":
    unittest.main()
