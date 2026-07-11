from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark_export.converter import BenchmarkExportConfig, BenchmarkExporter
from benchmark_predictions.compare import ModelComparisonConfig, ModelComparator
from benchmark_predictions.importer import (
    ExternalPredictionImportConfig,
    ExternalPredictionImporter,
)
from evaluation.cli import add_evaluation_parser, run_evaluation
from evaluation.evaluator import EvaluationConfig, PredictionEvaluator
from evaluation.evaluator_continuous import (
    ContinuousEvaluationConfig,
    ContinuousPredictionEvaluator,
)
from plot_dataset.cli import add_plot_dataset_parser, run_plot_dataset
from prepare_real.cli import add_prepare_real_parser, run_prepare_real
from plot_dataset.plotter import (
    DEFAULT_IMAGE_DPI,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DatasetPlotter,
    PlotConfig,
)
from simulate_dataset.cli import (
    add_simulate_dataset_parser,
    configure_logging,
    run_simulate_dataset,
)
from simulate_dataset.benchmark_subset import (
    BinaryBenchmarkSubsetConfig,
    BinaryBenchmarkSubsetSelector,
)
from simulate_dataset.config import SimulationConfig, VALID_SIMULATION_TASKS
from simulate_dataset.simulator import SimulateSearchDatasetGenerator
from vlm_classifier.cli import add_vlm_classifier_parser, run_vlm_classifier
from vlm_classifier.classifier import ClassifierConfig, VLMClassifier
from vlm_classifier.models import DryRunVLM, Gemma4VLM
from vlm_classifier.parser import classes_for_task, normalize_task


LOGGER = logging.getLogger(__name__)


def _load_config(
    config_path: Path,
    *,
    n_per_class: int | None = None,
    seed: int | None = None,
    output_dir: Path | None = None,
    task: str | None = None,
) -> tuple[SimulationConfig, Path]:
    config_path = config_path.expanduser()
    project_root = config_path.resolve().parent
    config = SimulationConfig.from_yaml(config_path).with_overrides(
        n_per_class=n_per_class,
        seed=seed,
        output_dir=output_dir,
        task=task,
    )
    return config, project_root


def _paths(config: SimulationConfig, project_root: Path) -> dict[str, Path]:
    output_dir = config.output_dir.resolve()
    return {
        "labels": output_dir / "metadata" / "labels.jsonl",
        "images": output_dir / "images",
        "image_manifest": output_dir / "metadata" / "image_manifest.jsonl",
        "predictions": project_root / "results" / "predictions.jsonl",
        "results": project_root / "results",
    }


def _simulate(
    *,
    config: SimulationConfig,
    project_root: Path,
    overwrite: bool,
) -> None:
    generator = SimulateSearchDatasetGenerator(config=config, project_root=project_root)
    records = generator.generate(overwrite=overwrite)
    LOGGER.info("Simulation finished: %d samples.", len(records))


def _plot(
    *,
    config: SimulationConfig,
    project_root: Path,
    overwrite: bool,
    cmap: str,
    normalization: str,
    percentile_low: float,
    percentile_high: float,
    width: int,
    height: int,
    dpi: int,
    include_title: bool,
) -> None:
    paths = _paths(config, project_root)
    plot_config = PlotConfig(
        labels_path=paths["labels"],
        output_dir=paths["images"],
        manifest_path=paths["image_manifest"],
        cmap=cmap,
        normalization=normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        width=width,
        height=height,
        dpi=dpi,
        include_title=include_title,
    )
    plotter = DatasetPlotter(config=plot_config, project_root=project_root)
    records = plotter.plot_all(overwrite=overwrite)
    LOGGER.info("Plotting finished: %d images.", len(records))


def _classify(
    *,
    config: SimulationConfig,
    project_root: Path,
    model_name: str,
    model_id: str | None,
    device_map: str,
    dtype: str,
    max_new_tokens: int,
    attn_implementation: str | None,
    cache_implementation: str | None,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    generation_seed: int | None,
    dry_run: bool,
    seed: int,
    max_retries: int,
    retry_delay: float,
    overwrite: bool,
    task: str,
    decision_threshold: float,
    output: Path | None = None,
    manifest: Path | None = None,
) -> None:
    paths = _paths(config, project_root)
    task = normalize_task(task)
    if dry_run:
        model = DryRunVLM(seed=seed, labels=classes_for_task(task))
        LOGGER.info("Classification in dry-run mode.")
    else:
        model = Gemma4VLM(
            model_name=model_id or model_name,
            device_map=device_map,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
            cache_implementation=cache_implementation,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=generation_seed,
        )
        LOGGER.info("Classification with Gemma 4: %s", model.model_id)

    classifier_config = ClassifierConfig(
        manifest_path=manifest or paths["image_manifest"],
        output_path=output or paths["predictions"],
        max_retries=max_retries,
        retry_delay=retry_delay,
        task=task,
        decision_threshold=decision_threshold,
    )
    classifier = VLMClassifier(
        config=classifier_config,
        model=model,
        project_root=project_root,
    )
    records = classifier.run(overwrite=overwrite)
    LOGGER.info("Classification finished: %d predictions.", len(records))


def _evaluate(
    *,
    config: SimulationConfig,
    project_root: Path,
    overwrite: bool,
    predictions: Path | None = None,
    output_dir: Path | None = None,
    task: str = "multiclass",
) -> None:
    paths = _paths(config, project_root)
    evaluator_config = EvaluationConfig(
        predictions_path=predictions or paths["predictions"],
        output_dir=output_dir or paths["results"],
        task=task,
    )
    evaluator = PredictionEvaluator(config=evaluator_config, project_root=project_root)
    evaluator.evaluate(overwrite=overwrite)


def _export_benchmark(
    *,
    config: SimulationConfig,
    project_root: Path,
    output_dir: Path,
    overwrite: bool,
    manifest_jsonl: Path | None = None,
    manifest_csv: Path | None = None,
) -> None:
    paths = _paths(config, project_root)
    exporter = BenchmarkExporter(
        config=BenchmarkExportConfig(
            labels_path=paths["labels"],
            output_dir=output_dir,
            manifest_jsonl=manifest_jsonl,
            manifest_csv=manifest_csv,
        ),
        project_root=project_root,
    )
    records = exporter.export(overwrite=overwrite)
    LOGGER.info(
        "Benchmark export finished: %d files in %s.",
        len(records),
        exporter.output_dir,
    )


def _import_benchmark_predictions(
    *,
    project_root: Path,
    benchmark_manifest: Path,
    external_predictions: Path,
    output: Path,
    threshold: float,
    aggregation: str,
    overwrite: bool,
) -> None:
    importer = ExternalPredictionImporter(
        config=ExternalPredictionImportConfig(
            benchmark_manifest=benchmark_manifest,
            external_predictions=external_predictions,
            output_path=output,
            threshold=threshold,
            aggregation=aggregation,
        ),
        project_root=project_root,
    )
    records = importer.import_predictions(overwrite=overwrite)
    LOGGER.info("External predictions imported: %d samples.", len(records))


def _compare_models(
    *,
    project_root: Path,
    vlm_metrics: Path,
    external_metrics: Path,
    vlm_predictions: Path,
    external_predictions: Path,
    output_dir: Path,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    calibration_bins: int,
    overwrite: bool,
) -> None:
    comparator = ModelComparator(
        config=ModelComparisonConfig(
            vlm_metrics=vlm_metrics,
            external_metrics=external_metrics,
            vlm_predictions=vlm_predictions,
            external_predictions=external_predictions,
            output_dir=output_dir,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            calibration_bins=calibration_bins,
        ),
        project_root=project_root,
    )
    outputs = comparator.compare(overwrite=overwrite)
    LOGGER.info("Comparison finished: %s", outputs["comparison_json"])


def run_simulate(args: argparse.Namespace) -> None:
    config, project_root = _load_config(
        args.config,
        n_per_class=args.n_per_class,
        seed=args.seed,
        output_dir=args.output_dir,
        task=args.task,
    )
    configure_logging(project_root, verbose=args.verbose)
    _simulate(config=config, project_root=project_root, overwrite=args.overwrite)


def run_plot(args: argparse.Namespace) -> None:
    config, project_root = _load_config(args.config)
    configure_logging(project_root, verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _plot(
        config=config,
        project_root=project_root,
        overwrite=args.overwrite,
        cmap=args.cmap,
        normalization=args.normalization,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        include_title=args.include_title,
    )


def _select_binary_benchmark(
    *,
    config: SimulationConfig,
    project_root: Path,
    output_dir: Path,
    n_frb: int,
    per_rfi_subtype: int,
    n_noise: int,
    subset_seed: int,
    overwrite: bool,
) -> None:
    paths = _paths(config, project_root)
    selector = BinaryBenchmarkSubsetSelector(
        config=BinaryBenchmarkSubsetConfig(
            source_labels=paths["labels"],
            source_image_manifest=paths["image_manifest"],
            source_labels_csv=paths["labels"].with_name("labels.csv"),
            output_dir=output_dir,
            n_frb=n_frb,
            per_rfi_subtype=per_rfi_subtype,
            n_noise=n_noise,
            seed=subset_seed,
        ),
        project_root=project_root,
    )
    records = selector.select(overwrite=overwrite)
    LOGGER.info("Binary benchmark selection finished: %d samples.", len(records))


def run_select_binary_benchmark(args: argparse.Namespace) -> None:
    config, project_root = _load_config(args.config)
    configure_logging(project_root, verbose=args.verbose)
    _select_binary_benchmark(
        config=config,
        project_root=project_root,
        output_dir=args.output_dir,
        n_frb=args.n_frb,
        per_rfi_subtype=args.per_rfi_subtype,
        n_noise=args.n_noise,
        subset_seed=args.seed,
        overwrite=args.overwrite,
    )


def run_classify(args: argparse.Namespace) -> None:
    config, project_root = _load_config(args.config)
    configure_logging(project_root, verbose=args.verbose)
    _classify(
        config=config,
        project_root=project_root,
        model_name=args.model,
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        cache_implementation=args.cache_implementation,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generation_seed=args.generation_seed,
        dry_run=args.dry_run,
        seed=args.seed,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        overwrite=args.overwrite,
        task=args.task,
        decision_threshold=args.decision_threshold,
        output=args.output,
        manifest=args.manifest,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    config, project_root = _load_config(args.config)
    configure_logging(project_root, verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _evaluate(
        config=config,
        project_root=project_root,
        overwrite=args.overwrite,
        predictions=args.predictions,
        output_dir=args.output_dir,
        task=args.task,
    )


def run_evaluate_continuous(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    configure_logging(project_root, verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    evaluator = ContinuousPredictionEvaluator(
        config=ContinuousEvaluationConfig(
            predictions=tuple(path.expanduser() for path in args.predictions),
            output_dir=args.output_dir.expanduser(),
            task=args.task,
            names=tuple(args.names) if args.names is not None else None,
            threshold=args.threshold,
            priors=tuple(args.prior or ContinuousEvaluationConfig.priors),
            threshold_steps=args.threshold_steps,
            calibration_bins=args.calibration_bins,
        ),
        project_root=project_root,
    )
    evaluator.evaluate(overwrite=args.overwrite)


def run_export_benchmark(args: argparse.Namespace) -> None:
    config, project_root = _load_config(args.config)
    configure_logging(project_root, verbose=args.verbose)
    _export_benchmark(
        config=config,
        project_root=project_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        manifest_jsonl=args.manifest_jsonl,
        manifest_csv=args.manifest_csv,
    )


def run_import_benchmark_predictions(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    configure_logging(project_root, verbose=args.verbose)
    _import_benchmark_predictions(
        project_root=project_root,
        benchmark_manifest=args.benchmark_manifest,
        external_predictions=args.external_predictions,
        output=args.output,
        threshold=args.threshold,
        aggregation=args.aggregation,
        overwrite=args.overwrite,
    )


def run_compare_models(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    configure_logging(project_root, verbose=args.verbose)
    _compare_models(
        project_root=project_root,
        vlm_metrics=args.vlm_metrics,
        external_metrics=args.external_metrics,
        vlm_predictions=args.vlm_predictions,
        external_predictions=args.external_predictions,
        output_dir=args.output_dir,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        calibration_bins=args.calibration_bins,
        overwrite=args.overwrite,
    )


def run_all(args: argparse.Namespace) -> None:
    config, project_root = _load_config(
        args.config,
        n_per_class=args.n_per_class,
        seed=args.seed,
        output_dir=args.output_dir,
        task=args.task,
    )
    configure_logging(project_root, verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _simulate(config=config, project_root=project_root, overwrite=args.overwrite)
    _plot(
        config=config,
        project_root=project_root,
        overwrite=args.overwrite,
        cmap=args.cmap,
        normalization=args.normalization,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        include_title=args.include_title,
    )
    _classify(
        config=config,
        project_root=project_root,
        model_name=args.model,
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
        cache_implementation=args.cache_implementation,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        generation_seed=args.generation_seed,
        dry_run=args.dry_run,
        seed=config.seed,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        overwrite=args.overwrite,
        task=args.task,
        decision_threshold=args.decision_threshold,
    )
    _evaluate(
        config=config,
        project_root=project_root,
        overwrite=args.overwrite,
        task=args.task,
    )


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="YAML configuration file.",
    )


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the stage's existing artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def _add_simulation_args(
    parser: argparse.ArgumentParser,
    *,
    include_task: bool = True,
) -> None:
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
    if include_task:
        parser.add_argument(
            "--task",
            default=None,
            choices=VALID_SIMULATION_TASKS,
            help=(
                "Simulation task. multiclass generates N per class; frb-binary "
                "generates N FRB, N/2 RFI, and N/2 NOISE."
            ),
        )


def _add_plot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument(
        "--normalization",
        choices=("linear", "minmax", "zscore", "percentile"),
        default="percentile",
        help="Normalization applied before plotting.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DEFAULT_IMAGE_DPI)
    title_group = parser.add_mutually_exclusive_group()
    title_group.add_argument(
        "--include-title",
        action="store_true",
        help="Add an anonymous title to the PNG. The default is no title.",
    )
    title_group.add_argument(
        "--no-title",
        action="store_true",
        help="Kept for compatibility: images are already generated without a title by default.",
    )


def _add_classify_args(
    parser: argparse.ArgumentParser,
    *,
    include_output: bool,
    include_seed: bool = True,
) -> None:
    parser.add_argument(
        "--model",
        default="gemma4",
        choices=("gemma4",),
        help="Target VLM model.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face ID of the Gemma 4 model. Default: google/gemma-4-E4B-it.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map used in the Transformers from_pretrained call.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help='dtype used in from_pretrained. Use "none" to omit it.',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens generated by the VLM.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help='Transformers attention implementation. Use "none" to omit it.',
    )
    parser.add_argument(
        "--cache-implementation",
        default="static",
        help='cache_implementation used in generate. Use "none" to omit it.',
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. The default is deterministic greedy decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature used only with --do-sample.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling used only with --do-sample.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling used only with --do-sample.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Transformers seed set before each generate call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate reproducible random predictions without loading a real VLM.",
    )
    if include_seed:
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Seed used by --dry-run mode.",
        )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    if include_output:
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="JSONL predictions file.",
        )
        parser.add_argument(
            "--manifest",
            type=Path,
            default=None,
            help=(
                "Alternative image manifest (e.g. "
                "dataset_real/metadata/image_manifest.jsonl). Defaults to the "
                "manifest of the dataset named in the config."
            ),
        )
    parser.add_argument(
        "--task",
        default="multiclass",
        choices=("multiclass", "frb-binary"),
        help="Pipeline task: multiclass or FRB vs NON_FRB.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Threshold on frb_probability to derive FRB in binary mode.",
    )


def _add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="JSONL predictions file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for the metrics.",
    )
    parser.add_argument(
        "--task",
        default="multiclass",
        choices=("multiclass", "frb-binary"),
        help="Evaluation task: multiclass or FRB vs NON_FRB.",
    )


def _add_evaluate_continuous_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or two predictions.jsonl files with frb_probability.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Optional names for the prediction files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_continuous"),
        help="Output directory for the continuous evaluation.",
    )
    parser.add_argument(
        "--task",
        default="frb-binary",
        choices=("frb-binary",),
        help="Continuous evaluation uses frb-binary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Operating point used for McNemar and the current summary.",
    )
    parser.add_argument(
        "--prior",
        type=float,
        action="append",
        default=None,
        help=(
            "FRB prior used to rescale precision/PR-AUC. May be repeated. "
            "Default: 1e-2, 1e-3, 1e-4."
        ),
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=101,
        help="Number of points in the threshold sweep.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for the reused calibration metrics.",
    )


def _add_select_binary_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory of the derived binary benchmark (e.g., dataset_binary).",
    )
    parser.add_argument(
        "--n-frb",
        type=int,
        default=1000,
        help="Number of selected FRB samples.",
    )
    parser.add_argument(
        "--per-rfi-subtype",
        type=int,
        default=100,
        help="Number of selected RFI samples per subtype.",
    )
    parser.add_argument(
        "--n-noise",
        type=int,
        default=500,
        help="Number of selected NOISE samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed of the deterministic subset selection.",
    )


def _add_benchmark_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Neutral directory to store the FITS files exported for benchmarking.",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional path for benchmark_manifest.jsonl.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional path for benchmark_manifest.csv.",
    )


def _add_import_benchmark_predictions_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--benchmark-manifest",
        type=Path,
        required=True,
        help="benchmark_manifest.jsonl generated by export-benchmark.",
    )
    parser.add_argument(
        "--external-predictions",
        type=Path,
        required=True,
        help="manifest.json or JSONL file with the external model predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Normalized predictions.jsonl file for evaluation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="FRB probability threshold to predict FRB.",
    )
    parser.add_argument(
        "--aggregation",
        choices=("max", "mean", "median"),
        default="max",
        help="How to aggregate multiple rows/candidates per external sample_id.",
    )


def _add_compare_models_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vlm-metrics", type=Path, required=True)
    parser.add_argument("--external-metrics", type=Path, required=True)
    parser.add_argument("--vlm-predictions", type=Path, required=True)
    parser.add_argument("--external-predictions", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the comparison.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Number of paired resamples for the 95%% CI in AUC/AP.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed of the paired bootstrap.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of bins for ECE and the reliability diagram.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main CLI of the frb-vlm-benchmark pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Simulate the synthetic dataset.")
    _add_config_arg(simulate)
    _add_simulation_args(simulate)
    _add_common_runtime_args(simulate)
    simulate.set_defaults(func=run_simulate)

    plot = subparsers.add_parser("plot", help="Plot the dataset PNG images.")
    _add_config_arg(plot)
    _add_plot_args(plot)
    _add_common_runtime_args(plot)
    plot.set_defaults(func=run_plot)

    prepare_real = subparsers.add_parser(
        "prepare-real",
        help=(
            "Prepare real dynamic spectra (e.g. FAST-FREX) for the VLM: "
            "windowing, per-channel normalization, zapping and block-averaging."
        ),
    )
    add_prepare_real_parser(prepare_real)
    prepare_real.set_defaults(func=run_prepare_real)

    select_binary = subparsers.add_parser(
        "select-binary-benchmark",
        help="Select the binary benchmark as a subset of the multiclass dataset.",
    )
    _add_config_arg(select_binary)
    _add_select_binary_benchmark_args(select_binary)
    _add_common_runtime_args(select_binary)
    select_binary.set_defaults(func=run_select_binary_benchmark)

    classify = subparsers.add_parser("classify", help="Classify images with a VLM.")
    _add_config_arg(classify)
    _add_classify_args(classify, include_output=True)
    _add_common_runtime_args(classify)
    classify.set_defaults(func=run_classify)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions.")
    _add_config_arg(evaluate)
    _add_evaluation_args(evaluate)
    _add_common_runtime_args(evaluate)
    evaluate.set_defaults(func=run_evaluate)

    evaluate_continuous = subparsers.add_parser(
        "evaluate-continuous",
        help="Evaluate frb_probability with ROC/PR and a threshold sweep.",
    )
    _add_evaluate_continuous_args(evaluate_continuous)
    _add_common_runtime_args(evaluate_continuous)
    evaluate_continuous.set_defaults(func=run_evaluate_continuous)

    export_benchmark = subparsers.add_parser(
        "export-benchmark",
        help="Export FITS files to a neutral external-benchmark folder.",
    )
    _add_config_arg(export_benchmark)
    _add_benchmark_export_args(export_benchmark)
    _add_common_runtime_args(export_benchmark)
    export_benchmark.set_defaults(func=run_export_benchmark)

    import_benchmark = subparsers.add_parser(
        "import-benchmark-predictions",
        help="Normalize external predictions into a binary predictions.jsonl.",
    )
    _add_import_benchmark_predictions_args(import_benchmark)
    _add_common_runtime_args(import_benchmark)
    import_benchmark.set_defaults(func=run_import_benchmark_predictions)

    compare_models = subparsers.add_parser(
        "compare-models",
        help="Compare metrics and predictions between the VLM and the external model.",
    )
    _add_compare_models_args(compare_models)
    _add_common_runtime_args(compare_models)
    compare_models.set_defaults(func=run_compare_models)

    all_parser = subparsers.add_parser("all", help="Run the full pipeline.")
    _add_config_arg(all_parser)
    _add_simulation_args(all_parser, include_task=False)
    _add_plot_args(all_parser)
    _add_classify_args(all_parser, include_output=False, include_seed=False)
    _add_common_runtime_args(all_parser)
    all_parser.set_defaults(func=run_all)

    simulate_parser = subparsers.add_parser(
        "simulate-dataset",
        help="Legacy alias: generate the balanced synthetic dataset with SimulateSearch.",
    )
    add_simulate_dataset_parser(simulate_parser)
    simulate_parser.set_defaults(func=run_simulate_dataset)

    plot_parser = subparsers.add_parser(
        "plot-dataset",
        help="Legacy alias: generate PNG images from the simulated PSRFITS.",
    )
    add_plot_dataset_parser(plot_parser)
    plot_parser.set_defaults(func=run_plot_dataset)

    vlm_parser = subparsers.add_parser(
        "vlm-classifier",
        help="Legacy alias: classify images with a VLM.",
    )
    add_vlm_classifier_parser(vlm_parser)
    vlm_parser.set_defaults(func=run_vlm_classifier)

    eval_parser = subparsers.add_parser(
        "evaluation",
        help="Legacy alias: evaluate the VLM predictions.",
    )
    add_evaluation_parser(eval_parser)
    eval_parser.set_defaults(func=run_evaluation)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
