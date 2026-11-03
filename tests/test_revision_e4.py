from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from revision_v4.e4_preprocessing import process_variant
from revision_v4.injection import P0, process_window, zscore


def _block(seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    data = rng.normal(10.0, 2.0, size=(64, 256)).astype(np.float32)
    data[7] = 3.0  # canal morto
    data[20] *= 9.0  # canal ruidoso, cai no zap
    weights = np.ones(64)
    weights[30] = 0.0  # DAT_WTS
    return SimpleNamespace(data=data, frequencies_mhz=np.linspace(1000.0, 1500.0, 64), channel_weights=weights)


class VariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.p0 = P0(time_bins=64, time_factor=4, freq_bins=16, zap_sigma=3.0, edge_channels=4)

    def test_p0_variant_is_the_paper_path(self) -> None:
        block = _block()
        zscored, keep = zscore(block, self.p0)

        expected = process_window(zscored, keep, block.frequencies_mhz, 1e-3, self.p0)
        variant = process_variant(block, 1e-3, "P0", self.p0)

        np.testing.assert_array_equal(variant.image, expected.image)

    def test_p1_skips_only_the_row_flattening(self) -> None:
        block = _block()

        p0 = process_variant(block, 1e-3, "P0", self.p0)
        p1 = process_variant(block, 1e-3, "P1", self.p0)

        np.testing.assert_array_equal(p1.image, p0.averaged)
        np.testing.assert_array_equal(p1.keep, p0.keep)

    def test_p2_keeps_every_live_channel(self) -> None:
        block = _block()

        p2 = process_variant(block, 1e-3, "P2", self.p0)

        self.assertFalse(p2.keep[7])
        self.assertTrue(p2.keep[20] and p2.keep[30] and p2.keep[0])
        self.assertEqual(p2.channels_zapped, 1)


if __name__ == "__main__":
    unittest.main()
