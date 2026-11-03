from __future__ import annotations

import unittest
from types import SimpleNamespace

from vlm_classifier.models import (
    DEFAULT_GEMMA4_MODEL_ID,
    Gemma4VLM,
    build_generation_kwargs,
    resolve_gemma4_model_id,
)


class Gemma4ModelTests(unittest.TestCase):
    def test_resolves_gemma4_alias_to_default_model_id(self) -> None:
        self.assertEqual(resolve_gemma4_model_id("gemma4"), DEFAULT_GEMMA4_MODEL_ID)
        self.assertEqual(resolve_gemma4_model_id("gemma-4"), DEFAULT_GEMMA4_MODEL_ID)

    def test_preserves_explicit_huggingface_model_id(self) -> None:
        model_id = "google/custom-gemma-4"

        self.assertEqual(resolve_gemma4_model_id(model_id), model_id)

    def test_greedy_generation_omits_sampling_parameters(self) -> None:
        kwargs = build_generation_kwargs(
            max_new_tokens=64,
            cache_implementation="static",
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

        self.assertEqual(kwargs["max_new_tokens"], 64)
        self.assertIs(kwargs["do_sample"], False)
        self.assertEqual(kwargs["cache_implementation"], "static")
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)

    def test_provenance_records_revision_processor_and_software(self) -> None:
        vlm = object.__new__(Gemma4VLM)
        vlm.model_id = "google/gemma-test"
        vlm.revision = "revision-123"
        vlm.model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="revision-123")
        )
        vlm.processor = SimpleNamespace()
        vlm.device_map = "auto"
        vlm.dtype = "auto"
        vlm.max_new_tokens = 64
        vlm.attn_implementation = "sdpa"
        vlm.cache_implementation = "static"
        vlm.do_sample = False
        vlm.temperature = None
        vlm.top_p = None
        vlm.top_k = None
        vlm.seed = None

        provenance = vlm.provenance()

        self.assertEqual(provenance["requested_revision"], "revision-123")
        self.assertEqual(provenance["model_revision"], "revision-123")
        self.assertIn("SimpleNamespace", provenance["processor_class"])
        self.assertEqual(
            set(provenance["software_versions"]),
            {"torch", "transformers", "pillow"},
        )

    def test_sampling_generation_includes_sampling_parameters(self) -> None:
        kwargs = build_generation_kwargs(
            max_new_tokens=64,
            cache_implementation=None,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

        self.assertIs(kwargs["do_sample"], True)
        self.assertEqual(kwargs["temperature"], 0.7)
        self.assertEqual(kwargs["top_p"], 0.9)
        self.assertEqual(kwargs["top_k"], 50)
        self.assertNotIn("cache_implementation", kwargs)


if __name__ == "__main__":
    unittest.main()
