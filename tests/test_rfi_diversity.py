from __future__ import annotations

import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from simulate_dataset.config import SimulationConfig
from simulate_dataset.simulator import (
    NOISE_TSYS_SCALE_RANGE,
    RFI_CATEGORIES,
    SampleContext,
    SimulateSearchDatasetGenerator,
)


class RfiDiversityTests(unittest.TestCase):
    def test_system_params_use_configured_telescope_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SimulationConfig(
                f1=1050.0,
                f2=1450.0,
                nchan=2048,
                tsamp=1.96608e-4,
                nbits=8,
                gain=0.7,
                tsys=25.0,
                output_dir=root / "dataset",
                seed=42,
                n_per_class=1,
                name="FAST L-band sim",
                telescope="FAST",
                observer="benchmark test",
                raj="0",
                decj="0",
                use_angle=0,
                imjd=58456,
                smjd=36400,
                levelset=1,
            )
            generator = SimulateSearchDatasetGenerator(config=config, project_root=root)
            params_dir = root / "params"
            work_dir = root / "work"
            params_dir.mkdir()
            work_dir.mkdir()
            context = SampleContext(
                sample_id="frb_00000",
                label="frb",
                sample_seed=123,
                rng=random.Random(123),
                system_tsys=25.0,
                system_tsys_scale=1.0,
                params_dir=params_dir,
                work_dir=work_dir,
                output_path=root / "frb_00000.fits",
            )

            generator._write_system_params(context)
            text = (work_dir / "system.params").read_text(encoding="utf-8")

            self.assertIn("name: FAST L-band sim", text)
            self.assertIn("telescope: FAST", text)
            self.assertIn("observer: benchmark test", text)
            self.assertIn("f1: 1050", text)
            self.assertIn("f2: 1450", text)
            self.assertIn("nchan: 2048", text)
            self.assertIn("tsamp: 0.000196608", text)
            self.assertIn("imjd: 58456", text)
            self.assertIn("smjd: 36400", text)

    def test_noise_tsys_scale_varies_reproducibly_by_sample_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = SimulationConfig(
                f1=1230.0,
                f2=1518.0,
                nchan=192,
                tsamp=0.0005,
                nbits=8,
                gain=0.7,
                tsys=25.0,
                output_dir=Path(tmp) / "dataset",
                seed=42,
                n_per_class=1,
            )
            generator = SimulateSearchDatasetGenerator(
                config=config,
                project_root=Path(tmp),
            )
            scales = [generator._sample_tsys_scale(seed) for seed in range(100, 105)]
            low, high = NOISE_TSYS_SCALE_RANGE

            self.assertEqual(generator._sample_tsys_scale(100), scales[0])
            self.assertGreater(len({round(scale, 6) for scale in scales}), 1)
            self.assertTrue(all(low <= scale <= high for scale in scales))

    def test_balanced_rfi_categories_cover_supported_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = SimulationConfig(
                f1=1230.0,
                f2=1518.0,
                nchan=192,
                tsamp=0.0005,
                nbits=8,
                gain=0.7,
                tsys=25.0,
                output_dir=Path(tmp) / "dataset",
                seed=42,
                n_per_class=10,
            )
            generator = SimulateSearchDatasetGenerator(
                config=config,
                project_root=Path(tmp),
            )

            categories = generator._balanced_rfi_categories()
            counts = Counter(categories)

            self.assertEqual(len(categories), config.n_per_class)
            self.assertEqual(set(counts), set(RFI_CATEGORIES))
            self.assertNotIn("birdies_tone", categories)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_wifi_rfi_writes_multiple_narrowband_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SimulationConfig(
                f1=1230.0,
                f2=1518.0,
                nchan=192,
                tsamp=0.0005,
                nbits=8,
                gain=0.7,
                tsys=25.0,
                output_dir=root / "dataset",
                seed=42,
                n_per_class=1,
            )
            generator = SimulateSearchDatasetGenerator(config=config, project_root=root)
            params_dir = root / "params"
            work_dir = root / "work"
            params_dir.mkdir()
            work_dir.mkdir()
            context = SampleContext(
                sample_id="rfi_00000",
                label="rfi",
                sample_seed=123,
                rng=random.Random(123),
                system_tsys=25.0,
                system_tsys_scale=1.0,
                params_dir=params_dir,
                work_dir=work_dir,
                output_path=root / "rfi_00000.fits",
            )

            params = generator._write_wifi_rfi(context)
            text = (work_dir / "wifi.params").read_text(encoding="utf-8")

            self.assertIn(params["band_count"], {2, 3})
            self.assertEqual(text.count("rfi_narrowBand:"), params["band_count"])
            self.assertEqual(len(params["bands"]), params["band_count"])

    def test_far_sidelobe_rfis_stay_inside_observing_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = SimulationConfig(
                f1=1230.0,
                f2=1518.0,
                nchan=192,
                tsamp=0.0005,
                nbits=8,
                gain=0.7,
                tsys=25.0,
                output_dir=root / "dataset",
                seed=42,
                n_per_class=1,
            )
            generator = SimulateSearchDatasetGenerator(config=config, project_root=root)
            params_dir = root / "params"
            work_dir = root / "work"
            params_dir.mkdir()
            work_dir.mkdir()
            context = SampleContext(
                sample_id="rfi_00000",
                label="rfi",
                sample_seed=123,
                rng=random.Random(123),
                system_tsys=25.0,
                system_tsys_scale=1.0,
                params_dir=params_dir,
                work_dir=work_dir,
                output_path=root / "rfi_00000.fits",
            )

            point_to_point = generator._write_point_to_point_rfi(context)
            satellite_like = generator._write_satellite_like_rfi(context)

            self.assertGreaterEqual(point_to_point["frequency"], config.f1)
            self.assertLessEqual(point_to_point["frequency"], config.f2)
            for source in satellite_like["sources"]:
                self.assertGreaterEqual(source["frequency"], config.f1)
                self.assertLessEqual(source["frequency"], config.f2)


if __name__ == "__main__":
    unittest.main()
