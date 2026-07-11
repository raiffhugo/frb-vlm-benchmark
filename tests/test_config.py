from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simulate_dataset.config import TASK_MULTICLASS, SimulationConfig


class SimulationConfigTests(unittest.TestCase):
    def test_loads_example_config_and_computes_samples(self) -> None:
        config = SimulationConfig.from_yaml(Path("config.yaml"))

        self.assertEqual(config.t0, 0.0)
        self.assertEqual(config.t1, 2.0)
        self.assertEqual(config.nchan, 2048)
        self.assertEqual(config.tsamp, 1.96608e-4)
        self.assertEqual(config.nbits, 8)
        self.assertEqual(config.samples_per_file, 10173)
        self.assertEqual(config.output_dir.name, "dataset")
        self.assertEqual(config.frb_max_width, 0.005)
        self.assertEqual(config.name, "FAST L-band sim")
        self.assertEqual(config.telescope, "FAST")
        self.assertEqual(config.observer, "benchmark test")
        self.assertEqual(config.smjd, 36400)
        self.assertEqual(config.task, TASK_MULTICLASS)

    def test_loads_binary_config(self) -> None:
        config = SimulationConfig.from_yaml(Path("config_binary.yaml"))

        self.assertEqual(config.name, "FAST L-band sim")
        self.assertEqual(config.telescope, "FAST")
        self.assertEqual(config.observer, "benchmark test")
        self.assertEqual(config.f1, 1000)
        self.assertEqual(config.f2, 1500)
        self.assertEqual(config.nchan, 2048)
        self.assertEqual(config.tsamp, 1.96608e-4)
        self.assertEqual(config.raj, "0")
        self.assertEqual(config.decj, "0")
        self.assertEqual(config.use_angle, 0)
        self.assertEqual(config.imjd, 58456)
        self.assertEqual(config.smjd, 36400)
        self.assertEqual(config.levelset, 1)
        self.assertEqual(config.output_dir.name, "dataset_binary")

    def test_rejects_non_two_second_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_config.yaml"
            path.write_text(
                "\n".join(
                    [
                        "f1: 1230.0",
                        "f2: 1518.0",
                        "nchan: 96",
                        "t0: 0.0",
                        "t1: 3.0",
                        "tsamp: 0.001",
                        "nbits: 2",
                        "gain: 0.7",
                        "tsys: 25.0",
                        "output_dir: dataset",
                        "seed: 42",
                        "n_per_class: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                SimulationConfig.from_yaml(path)

    def test_rejects_unsupported_nbits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_nbits.yaml"
            path.write_text(
                "\n".join(
                    [
                        "f1: 1230.0",
                        "f2: 1518.0",
                        "nchan: 192",
                        "t0: 0.0",
                        "t1: 2.0",
                        "tsamp: 0.0005",
                        "nbits: 3",
                        "gain: 0.7",
                        "tsys: 25.0",
                        "output_dir: dataset",
                        "seed: 42",
                        "n_per_class: 1",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                SimulationConfig.from_yaml(path)

    def test_rejects_too_small_frb_max_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_width.yaml"
            path.write_text(
                "\n".join(
                    [
                        "f1: 1230.0",
                        "f2: 1518.0",
                        "nchan: 96",
                        "t0: 0.0",
                        "t1: 2.0",
                        "tsamp: 0.001",
                        "nbits: 2",
                        "gain: 0.7",
                        "tsys: 25.0",
                        "output_dir: dataset",
                        "seed: 42",
                        "n_per_class: 1",
                        "frb_max_width: 0.001",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                SimulationConfig.from_yaml(path)

    def test_rejects_odd_n_per_class_for_binary_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_binary_count.yaml"
            path.write_text(
                "\n".join(
                    [
                        "f1: 1230.0",
                        "f2: 1518.0",
                        "nchan: 192",
                        "t0: 0.0",
                        "t1: 2.0",
                        "tsamp: 0.0005",
                        "nbits: 8",
                        "gain: 0.7",
                        "tsys: 25.0",
                        "output_dir: dataset",
                        "seed: 42",
                        "n_per_class: 5",
                        "task: frb-binary",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "even n_per_class"):
                SimulationConfig.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
