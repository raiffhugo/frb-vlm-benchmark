from __future__ import annotations

import json
import unittest
from pathlib import Path

from revision_v4.e6_prompts import P0_FILE, build


ROOT = Path(__file__).resolve().parents[1]


class E6PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p0 = (ROOT / P0_FILE).read_text(encoding="utf-8")
        self.prompts = build(self.p0)

    def test_reordered_prompt_has_the_same_lines_as_p0(self) -> None:
        reordered = self.prompts["e6_pb_non_frb_first"]

        self.assertEqual(sorted(reordered.splitlines()), sorted(
            line.replace("1. FRB", "2. FRB").replace("2. Low", "3. Low").replace("3. NON", "1. NON")
            for line in self.p0.splitlines()
        ))
        self.assertLess(reordered.index("NON_FRB cases"), reordered.index("FRB candidates"))

    def test_ablations_remove_only_their_content(self) -> None:
        no_low_snr = self.prompts["e6_pe_no_low_snr"]
        generic = self.prompts["e6_pd_generic_non_frb"]

        self.assertNotIn("Low-S/N", no_low_snr)
        self.assertIn("SINGLE localized transient", no_low_snr)
        for family in ("persistent horizontal", "zero-DM", "periodic stripes", "saturated_or_clipped"):
            self.assertNotIn(family, generic)
        self.assertIn("Low-S/N sensitivity", generic)

    def test_relaxed_prompt_drops_single_and_broadband_requirements(self) -> None:
        relaxed = self.prompts["e6_r_relaxed_morphology"]

        self.assertNotIn("SINGLE", relaxed)
        self.assertNotIn("two or more temporally separated", relaxed)
        self.assertIn("same dispersion sweep", relaxed)
        self.assertIn("zero-DM broadband vertical impulses", relaxed)
        self.assertIn("persistent horizontal frequency bands", relaxed)

    def test_every_variant_keeps_the_parsed_fields(self) -> None:
        for name, text in self.prompts.items():
            with self.subTest(prompt=name):
                start = text.index("{")
                schema = json.loads(text[start : text.rindex("}") + 1])
                self.assertEqual(schema["label"], "FRB|NON_FRB")
                self.assertIn("frb_probability", schema)
                self.assertTrue(text.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
