from __future__ import annotations

import unittest

import numpy as np

from check_burst_visibility import _boxcar_search, dedispersed_snr
from prepare_real.preprocess import DISPERSION_CONSTANT_S_MHZ2, block_average
from revision_v4.injection import (
    P0,
    Pulse,
    dedispersed_profile,
    expected_sigma_per_unit,
    process_window,
    pulse_unit,
)


TSAMP = 1e-3
FREQS = np.linspace(1000.0, 1500.0, 128)


class PulseShapeTests(unittest.TestCase):
    def test_unsmeared_pulse_peaks_at_one_at_its_arrival_time(self) -> None:
        pulse = Pulse(t_top_s=0.2005, dm=0.0, fwhm_s=0.01)

        unit = pulse_unit(pulse, FREQS, 1000, TSAMP)

        self.assertAlmostEqual(float(unit.max()), 1.0, places=5)
        self.assertTrue(np.all(np.argmax(unit, axis=1) == 200))

    def test_dispersion_delay_follows_the_cold_plasma_law(self) -> None:
        pulse = Pulse(t_top_s=0.1005, dm=300.0, fwhm_s=0.005)

        unit = pulse_unit(pulse, FREQS, 1000, TSAMP)

        expected = 0.1005 + DISPERSION_CONSTANT_S_MHZ2 * 300.0 * (FREQS**-2 - FREQS.max() ** -2)
        peaks = (np.argmax(unit, axis=1) + 0.5) * TSAMP
        np.testing.assert_allclose(peaks, expected, atol=TSAMP)

    def test_scattering_keeps_fluence_and_lowers_the_peak(self) -> None:
        sharp = pulse_unit(Pulse(t_top_s=0.2, dm=0.0, fwhm_s=0.004), FREQS, 1000, TSAMP)
        scattered = pulse_unit(
            Pulse(t_top_s=0.2, dm=0.0, fwhm_s=0.004, tau_1250_s=0.005), FREQS, 1000, TSAMP
        )

        np.testing.assert_allclose(scattered.sum(axis=1), sharp.sum(axis=1), rtol=2e-3)
        self.assertLess(scattered[0].max(), scattered[-1].max())  # tau maior no canal baixo
        self.assertLess(scattered.max(), sharp.max())

    def test_band_limited_envelope_is_gaussian_in_frequency(self) -> None:
        pulse = Pulse(
            t_top_s=0.2, dm=0.0, fwhm_s=0.01, band_center_mhz=1250.0, band_fwhm_mhz=100.0
        )

        peaks = pulse_unit(pulse, FREQS, 1000, TSAMP).max(axis=1)

        self.assertEqual(int(np.argmax(peaks)), int(np.argmin(np.abs(FREQS - 1250.0))))
        half = peaks >= 0.5 * peaks.max()
        self.assertAlmostEqual(FREQS[half].max() - FREQS[half].min(), 100.0, delta=8.0)


class DiagnosticEquivalenceTests(unittest.TestCase):
    def test_profile_and_boxcars_reproduce_dedispersed_snr(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.normal(size=(64, 512)).astype(np.float32)
        image[[3, 10, 40]] = 0.0
        image[:, 200:203] += 1.5
        freqs = np.linspace(1000.0, 1500.0, 64)

        for dm in (0.0, 120.0, 400.0):
            with self.subTest(dm=dm):
                profile = dedispersed_profile(image, freqs, 0.002, dm)
                sigma, time, _, _ = _boxcar_search(profile, profile.size, 0.002)
                self.assertEqual((sigma, time), dedispersed_snr(image, freqs, 0.002, dm))


class CalibrationTests(unittest.TestCase):
    def test_target_sigma_is_recovered_in_white_noise(self) -> None:
        p0 = P0(time_bins=512, time_factor=4, freq_bins=64, edge_channels=0)
        freqs = np.linspace(1000.0, 1500.0, 256)
        tsamp = 4e-4
        pulse = Pulse(t_top_s=0.15, dm=200.0, fwhm_s=0.004)
        unit = pulse_unit(pulse, freqs, p0.window_samples, tsamp)
        measured = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            zscored = rng.normal(size=(freqs.size, p0.window_samples)).astype(np.float32)
            keep = np.ones(freqs.size, dtype=bool)
            background = process_window(zscored, keep, freqs, tsamp, p0)
            pulse_averaged, _, _ = block_average(
                unit, keep=keep, time_factor=p0.time_factor, freq_bins=p0.freq_bins
            )
            amplitude = 15.0 / expected_sigma_per_unit(background, pulse_averaged, pulse.dm)
            injected = process_window(zscored, keep, freqs, tsamp, p0, injection=amplitude * unit)
            sigma, _ = dedispersed_snr(
                injected.image, injected.frequencies_mhz, injected.bin_seconds, pulse.dm
            )
            measured.append(sigma)

        self.assertAlmostEqual(float(np.mean(measured)), 15.0, delta=1.5)

    def test_no_injection_leaves_the_p0_array_untouched(self) -> None:
        p0 = P0(time_bins=128, time_factor=2, freq_bins=32, edge_channels=0)
        rng = np.random.default_rng(3)
        zscored = rng.normal(size=(64, p0.window_samples)).astype(np.float32)
        keep = np.ones(64, dtype=bool)
        keep[5] = False

        window = process_window(zscored, keep, np.linspace(1000.0, 1500.0, 64), 1e-3, p0)
        zero = process_window(
            zscored, keep, np.linspace(1000.0, 1500.0, 64), 1e-3, p0,
            injection=np.zeros_like(zscored),
        )

        np.testing.assert_array_equal(window.image, zero.image)


if __name__ == "__main__":
    unittest.main()


class RedrawTests(unittest.TestCase):
    def test_only_corrupted_windows_get_a_new_start(self) -> None:
        from revision_v4.e2_injection import redraw_corrupted

        plan = [{"fits_path": "a", "start_sample": s, "injection_id": f"i{s}"} for s in (0, 100, 200)]
        layouts = {"a": {"nsamp_total": 61440}}
        corrupted = {100}

        result = redraw_corrupted(
            [dict(e) for e in plan], layouts, P0(), stream=(7,), workers=1,
            is_clean=lambda _path, start, _n: start not in corrupted,
        )

        self.assertEqual([e["start_sample"] for e in result][0::2], [0, 200])
        self.assertNotEqual(result[1]["start_sample"], 100)
        self.assertEqual(result[1]["redrawn"], {"original_start_sample": 100, "attempts": 1})
        self.assertNotIn("redrawn", result[0])
