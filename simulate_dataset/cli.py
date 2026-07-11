from __future__ import annotations

import argparse
import logging
from pathlib import Path

from simulate_dataset.config import SimulationConfig, VALID_SIMULATION_TASKS
from simulate_dataset.simulator import SimulateSearchDatasetGenerator


def add_simulate_dataset_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="Number of samples per class; in frb-binary, per binary class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for the synthetic parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--task",
        choices=VALID_SIMULATION_TASKS,
        default=None,
        help=(
            "Simulation task. multiclass generates N per class; frb-binary "
            "generates N FRB, N/2 RFI, and N/2 NOISE."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previously generated files with the same names.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging on the console and in the log file.",
    )


def configure_logging(project_root: Path, *, verbose: bool = False) -> Path:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "simulation.log"

    level = logging.DEBUG if verbose else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return log_file


def run_simulate_dataset(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser()
    project_root = config_path.resolve().parent
    log_file = configure_logging(project_root, verbose=args.verbose)

    config = SimulationConfig.from_yaml(config_path).with_overrides(
        n_per_class=args.n_per_class,
        seed=args.seed,
        output_dir=args.output_dir,
        task=args.task,
    )
    generator = SimulateSearchDatasetGenerator(
        config=config,
        project_root=project_root,
    )
    records = generator.generate(overwrite=args.overwrite)

    logging.getLogger(__name__).info(
        "Generation finished: %d samples, metadata in %s, logs in %s",
        len(records),
        generator.metadata_dir,
        log_file,
    )
