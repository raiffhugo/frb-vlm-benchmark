from __future__ import annotations

import unittest

from vlm_classifier.classifier import _is_fatal_model_error


class ClassifierTests(unittest.TestCase):
    def test_detects_cuda_device_side_assert_as_fatal(self) -> None:
        error = "AcceleratorError: CUDA error: device-side assert triggered"

        self.assertTrue(_is_fatal_model_error(error))

    def test_treats_parser_error_as_non_fatal(self) -> None:
        error = "ValueError: Resposta sem objeto JSON."

        self.assertFalse(_is_fatal_model_error(error))


if __name__ == "__main__":
    unittest.main()
