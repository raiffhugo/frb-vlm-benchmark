from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_pipeline import build_parser
from simulate_dataset.config import (
    TASK_FRB_BINARY,
    TASK_MULTICLASS,
    SimulationConfig,
)
from simulate_dataset.simulator import SimulateSearchDatasetGenerator


def make_config(
    *,
    n_per_class: int = 10,
    task: str = TASK_MULTICLASS,
) -> SimulationConfig:
    return SimulationConfig(
        f1=1230.0,
        f2=1518.0,
        nchan=192,
        tsamp=0.0005,
        nbits=8,
        gain=0.7,
        tsys=25.0,
        output_dir=Path("dataset"),
        seed=42,
        n_per_class=n_per_class,
        task=task,
    )


class SimulationTaskTests(unittest.TestCase):
    def test_multiclass_counts_use_n_per_class_for_each_source_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(n_per_class=10)
            generator = SimulateSearchDatasetGenerator(
                config=config,
                project_root=Path(tmp),
            )

            self.assertEqual(
                generator._class_counts(),
                {"frb": 10, "rfi": 10, "noise": 10},
            )

    def test_binary_counts_balance_frb_against_non_frb_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(n_per_class=10, task=TASK_FRB_BINARY)
            generator = SimulateSearchDatasetGenerator(
                config=config,
                project_root=Path(tmp),
            )

            self.assertEqual(
                generator._class_counts(),
                {"frb": 10, "rfi": 5, "noise": 5},
            )

    def test_binary_rfi_category_count_matches_actual_rfi_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(n_per_class=10, task=TASK_FRB_BINARY)
            generator = SimulateSearchDatasetGenerator(
                config=config,
                project_root=Path(tmp),
            )

            self.assertEqual(len(generator._balanced_rfi_categories(5)), 5)

    def test_run_pipeline_parses_task_for_simulate_and_all(self) -> None:
        parser = build_parser()

        simulate_args = parser.parse_args(
            ["simulate", "--task", "frb-binary", "--n-per-class", "10"]
        )
        all_args = parser.parse_args(
            ["all", "--task", "frb-binary", "--n-per-class", "10", "--dry-run"]
        )

        self.assertEqual(simulate_args.task, TASK_FRB_BINARY)
        self.assertEqual(all_args.task, TASK_FRB_BINARY)


if __name__ == "__main__":
    unittest.main()
