from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plot_dataset.plotter import DEFAULT_IMAGE_DPI
from prepare_real.builder import (
    NEGATIVE_WINDOW_MODES,
    READER_BACKENDS,
    RealDatasetPreparer,
    RealPrepConfig,
)
from prepare_real.preprocess import TOA_REFERENCES
from simulate_dataset.cli import configure_logging


def add_prepare_real_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-manifest",
        type=Path,
        required=True,
        help=(
            "CSV or JSONL listing the real samples: fits_path,label[,toa,dm,source,id]. "
            "toa in seconds within the file; dm in pc/cm^3."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_real"),
        help="Output directory (images/ and metadata/image_manifest.jsonl).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="Duration of the extracted window, in seconds.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=1024,
        help="Number of time bins after block-averaging.",
    )
    parser.add_argument(
        "--freq-bins",
        type=int,
        default=512,
        help="Number of channels after block-averaging.",
    )
    parser.add_argument(
        "--zap-sigma",
        type=float,
        default=5.0,
        help=(
            "Robust threshold (in sigmas) for masking channels with anomalous "
            "median or variance. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--edge-channels",
        type=int,
        default=0,
        help="Number of channels masked at each edge of the band.",
    )
    parser.add_argument(
        "--toa-ref",
        choices=TOA_REFERENCES,
        default="top",
        help=(
            "Convention for the manifest toa: arrival at the highest frequency (top), "
            "the lowest (bottom), or at infinite frequency (infinite)."
        ),
    )
    parser.add_argument(
        "--negative-window",
        choices=NEGATIVE_WINDOW_MODES,
        default="random",
        help="Window placement for RFI/NOISE samples without a toa.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the random placement of negative windows.",
    )
    parser.add_argument(
        "--reader",
        choices=READER_BACKENDS,
        default="astropy",
        help=(
            "PSRFITS reading backend. 'astropy' uses the built-in reader; "
            "'your' uses the your library (optional [real] extra) as a "
            "cross-check."
        ),
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap.")
    parser.add_argument(
        "--normalization",
        choices=("linear", "minmax", "zscore", "percentile"),
        default="percentile",
        help="Display normalization applied after preprocessing.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (benchmark protocol: 1024).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Image height in pixels (benchmark protocol: 768).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_IMAGE_DPI,
        help="DPI used to convert pixels into figure size.",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Add an anonymized title to the PNG. The default is no title.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs and manifest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )


def run_prepare_real(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    log_file = configure_logging(project_root, verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    config = RealPrepConfig(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        time_bins=args.time_bins,
        freq_bins=args.freq_bins,
        zap_sigma=args.zap_sigma,
        edge_channels=args.edge_channels,
        toa_reference=args.toa_ref,
        negative_window=args.negative_window,
        seed=args.seed,
        reader=args.reader,
        cmap=args.cmap,
        normalization=args.normalization,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        include_title=args.include_title,
    )
    preparer = RealDatasetPreparer(config=config, project_root=project_root)
    records = preparer.prepare(overwrite=args.overwrite)
    logging.getLogger(__name__).info(
        "Preparation finished: %d images, manifest in %s, logs in %s",
        len(records),
        preparer.manifest_path,
        log_file,
    )
