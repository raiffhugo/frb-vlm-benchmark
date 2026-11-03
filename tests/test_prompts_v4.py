from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import prompt_timing_experiment
from revision_v4 import e7_prompts
from revision_v4.e6_prompts import BUILDERS, build
from vlm_classifier.prompt import BINARY_FRB_PROMPT, CLASSIFICATION_PROMPT


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "binary_full": BINARY_FRB_PROMPT,
    "binary_lean": prompt_timing_experiment.LEAN_BINARY_PROMPT,
    "binary_lean_reason": prompt_timing_experiment.LEAN_REASON_BINARY_PROMPT,
    "multiclass": CLASSIFICATION_PROMPT,
}


class FrozenPromptTests(unittest.TestCase):
    def test_registry_hashes_match_files(self) -> None:
        registry = json.loads((ROOT / "prompts_v4" / "registry.json").read_text(encoding="utf-8"))

        for name, entry in registry.items():
            with self.subTest(prompt=name):
                content = (ROOT / entry["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])

    def test_exported_prompts_are_byte_identical_to_their_sources(self) -> None:
        for name, text in SOURCES.items():
            with self.subTest(prompt=name):
                self.assertEqual(
                    (ROOT / "prompts_v4" / f"{name}.txt").read_bytes(), text.encode("utf-8")
                )

    def test_e6_prompts_are_the_frozen_output_of_the_builder(self) -> None:
        p0 = (ROOT / "prompts_v4" / "binary_full.txt").read_text(encoding="utf-8")
        registry = json.loads((ROOT / "prompts_v4" / "registry.json").read_text(encoding="utf-8"))

        for name, text in build(p0).items():
            with self.subTest(prompt=name):
                self.assertIn(name, registry)
                self.assertEqual(
                    (ROOT / "prompts_v4" / f"{name}.txt").read_bytes(), text.encode("utf-8")
                )
        self.assertEqual(len(BUILDERS), 6)

    def test_e7_prompts_are_the_frozen_output_of_the_builder(self) -> None:
        v2 = (ROOT / e7_prompts.V2_FILE).read_text(encoding="utf-8")
        registry = json.loads((ROOT / "prompts_v4" / "registry.json").read_text(encoding="utf-8"))

        built = e7_prompts.build(v2)
        for name, text in built.items():
            with self.subTest(prompt=name):
                self.assertIn(name, registry)
                self.assertEqual(
                    (ROOT / "prompts_v4" / f"{name}.txt").read_bytes(), text.encode("utf-8")
                )
        self.assertEqual(set(built), set(e7_prompts.DESCRIPTIONS))


if __name__ == "__main__":
    unittest.main()
