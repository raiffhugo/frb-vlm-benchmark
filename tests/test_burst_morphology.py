from __future__ import annotations

import unittest

import numpy as np

from check_burst_visibility import burst_morphology, dedispersed_snr


NCHAN, NBINS = 64, 512
FREQS = np.linspace(1000.0, 1500.0, NCHAN)
BIN_SECONDS = 2e-3


def _image(*, band: slice, amplitude: float, width: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.normal(size=(NCHAN, NBINS)).astype(np.float32)
    image[band, 200 : 200 + width] += amplitude
    return image


class BurstMorphologyTests(unittest.TestCase):
    def test_peak_matches_dedispersed_snr_exactly(self) -> None:
        image = _image(band=slice(0, NCHAN), amplitude=1.5, width=4)

        morphology = burst_morphology(image, FREQS, BIN_SECONDS, 0.0)
        sigma, _ = dedispersed_snr(image, FREQS, BIN_SECONDS, 0.0)

        self.assertEqual(morphology["peak_sigma"], sigma)

    def test_broadband_burst_occupies_the_band_and_recovers_width(self) -> None:
        image = _image(band=slice(0, NCHAN), amplitude=1.5, width=8)

        morphology = burst_morphology(image, FREQS, BIN_SECONDS, 0.0)

        self.assertGreaterEqual(morphology["band_fraction"], 0.85)
        self.assertIn(morphology["width_bins"], (4, 8, 16))
        self.assertAlmostEqual(morphology["width_ms"], morphology["width_bins"] * 2.0)

    def test_band_limited_burst_gives_its_fraction_and_edges(self) -> None:
        image = _image(band=slice(48, NCHAN), amplitude=3.0, width=8)

        morphology = burst_morphology(image, FREQS, BIN_SECONDS, 0.0)

        self.assertAlmostEqual(morphology["band_fraction"], 0.25, delta=0.07)
        self.assertGreater(morphology["band_low_mhz"], 1330.0)
        self.assertEqual(morphology["band_high_mhz"], 1500.0)

    def test_dispersed_burst_is_aligned_before_measuring(self) -> None:
        rng = np.random.default_rng(5)
        image = rng.normal(size=(NCHAN, NBINS)).astype(np.float32)
        dm = 100.0
        delays = 4.148808e3 * dm * (FREQS**-2 - FREQS.max() ** -2)
        for channel, delay in enumerate(np.round(delays / BIN_SECONDS).astype(int)):
            image[channel, 100 + delay : 104 + delay] += 1.5

        aligned = burst_morphology(image, FREQS, BIN_SECONDS, dm)
        wrong_dm = burst_morphology(image, FREQS, BIN_SECONDS, 0.0)

        self.assertGreater(aligned["peak_sigma"], 2 * wrong_dm["peak_sigma"])
        self.assertGreaterEqual(aligned["band_fraction"], 0.85)

    def test_empty_image_returns_nan_occupancy(self) -> None:
        morphology = burst_morphology(np.zeros((NCHAN, NBINS), np.float32), FREQS, BIN_SECONDS, 0.0)

        self.assertEqual(morphology["peak_sigma"], 0.0)
        self.assertTrue(np.isnan(morphology["band_fraction"]))


if __name__ == "__main__":
    unittest.main()
