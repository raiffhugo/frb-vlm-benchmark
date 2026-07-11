from __future__ import annotations

import unittest

from vlm_classifier.parser import (
    derive_binary_label_from_probability,
    map_label_for_task,
    normalize_label,
    parse_model_response,
)


class ParserTests(unittest.TestCase):
    def test_extracts_json_from_text_and_normalizes_label(self) -> None:
        parsed = parse_model_response(
            'Model says: {"label": "fast radio burst", "confidence": "91%", '
            '"reason": "dispersed sweep"}'
        )

        self.assertEqual(parsed["label"], "FRB")
        self.assertAlmostEqual(parsed["confidence"], 0.91)
        self.assertEqual(parsed["reason"], "dispersed sweep")

    def test_normalizes_common_labels(self) -> None:
        self.assertEqual(normalize_label("radio frequency interference"), "RFI")
        self.assertEqual(normalize_label("system_noise"), "NOISE")

    def test_rejects_invalid_label(self) -> None:
        with self.assertRaises(ValueError):
            normalize_label("PULSAR")

    def test_binary_parser_maps_non_frb_aliases(self) -> None:
        parsed = parse_model_response(
            '{"label": "radio frequency interference", "frb_probability": 0.2, '
            '"confidence": 0.8}',
            task="frb-binary",
        )

        self.assertEqual(parsed["label"], "NON_FRB")
        self.assertEqual(parsed["model_self_label"], "NON_FRB")
        self.assertAlmostEqual(parsed["frb_probability"], 0.2)
        self.assertTrue(parsed["label_probability_consistent"])
        self.assertEqual(normalize_label("not a fast radio burst", task="frb-binary"), "NON_FRB")
        self.assertEqual(map_label_for_task("NOISE", task="frb-binary"), "NON_FRB")
        self.assertEqual(map_label_for_task("FRB", task="frb-binary"), "FRB")

    def test_binary_parser_rejects_missing_probability(self) -> None:
        with self.assertRaises(ValueError):
            parse_model_response(
                '{"label": "FRB", "confidence": 0.9}',
                task="frb-binary",
            )

    def test_binary_parser_preserves_probability_and_derives_label(self) -> None:
        parsed = parse_model_response(
            '{"label": "NON_FRB", "frb_probability": 0.7, "confidence": 0.8}',
            task="frb-binary",
        )

        self.assertEqual(parsed["model_self_label"], "NON_FRB")
        self.assertEqual(parsed["label"], "FRB")
        self.assertAlmostEqual(parsed["frb_probability"], 0.7)
        self.assertFalse(parsed["label_probability_consistent"])
        self.assertIn("content_warning", parsed)

    def test_derives_binary_label_from_configurable_threshold(self) -> None:
        self.assertEqual(
            derive_binary_label_from_probability(0.4, threshold=0.3),
            "FRB",
        )
        self.assertEqual(
            derive_binary_label_from_probability(0.4, threshold=0.5),
            "NON_FRB",
        )
        with self.assertRaises(ValueError):
            derive_binary_label_from_probability(0.4, threshold=1.5)


if __name__ == "__main__":
    unittest.main()
