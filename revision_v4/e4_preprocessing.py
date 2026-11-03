"""E4: ablacao do pre-processamento (plano, secao 10.6; definicoes na 10.12).

As 1600 janelas do artigo, pelas variantes:
  - P0: o pre-processamento do artigo (refeito aqui so para conferir: tem de reproduzir os PNGs
    de dataset_real_full byte a byte);
  - P1: o P0 sem ``flatten_rows``;
  - P2: so decodificacao, normalizacao por canal e ``block_average``: sem zap de 3 sigma, sem
    DAT_WTS, sem corte de bordas e sem ``flatten_rows``. Canais de variancia zero continuam
    fora, porque nao tem dado (a normalizacao dividiria por zero).
Cada variante sai em dataset_real_e4_<variante>/, com os mesmos sample_id do artigo, para que a
comparacao seja pareada.

``analyze`` compara as rodadas do P1 e do P2 com as do artigo (P0) nas mesmas 1600 janelas: FP
nos 1000 negativos, recall nos 600 positivos e acuracia, com McNemar exato, e a regra do C6
(secao 10.12): o P2 "multiplica os FP" quando FP(P2) >= 2 FP(P0) e o McNemar nos negativos da
p < 0,05, avaliado por modelo.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from prepare_real import reader as astropy_reader
from prepare_real.preprocess import (
    averaged_frequency_axis,
    block_average,
    channel_keep_mask,
    flatten_rows,
    normalize_per_channel,
    robust_channel_stats,
)
from revision_v4.common import RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json, write_jsonl
from revision_v4.injection import P0, Window, render
from revision_v4.paired_runs import (
    ALPHA,
    Run,
    load,
    read_visibility,
    recall_by_bin,
    recall_by_bin_paired,
    summary,
    versus,
)


OUTPUT_DIR = RESULTS_ROOT / "e4"
PAPER_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")
VARIANTS = ("P0", "P1", "P2")
PAPER_RUNS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
FP_MULTIPLIER = 2.0


def process_variant(block, tsamp_s: float, variant: str, p0: P0) -> Window:
    median, sigma = robust_channel_stats(block.data)
    if variant == "P2":
        keep = channel_keep_mask(median=median, sigma=sigma, weights=None, zap_sigma=0.0, edge_channels=0)
    else:
        keep = channel_keep_mask(
            median=median,
            sigma=sigma,
            weights=block.channel_weights,
            zap_sigma=p0.zap_sigma,
            edge_channels=p0.edge_channels,
        )
    zscored = normalize_per_channel(block.data, median=median, sigma=sigma, keep=keep)
    averaged, freq_factor, trim_low = block_average(
        zscored, keep=keep, time_factor=p0.time_factor, freq_bins=p0.freq_bins
    )
    image = flatten_rows(averaged) if variant == "P0" else averaged
    return Window(
        image=image,
        averaged=averaged,
        frequencies_mhz=averaged_frequency_axis(
            block.frequencies_mhz, freq_bins=p0.freq_bins, freq_factor=freq_factor, trim_low=trim_low
        ),
        bin_seconds=p0.time_factor * tsamp_s,
        keep=keep,
        channels_zapped=int(np.count_nonzero(~keep)),
    )


def render_record(record: dict[str, Any], variant: str, dataset_dir: Path) -> dict[str, Any]:
    # O fator de tempo e o do artigo para cada janela: 20 nos arquivos de 98,304 us e 40 nos de
    # 49,152 us (FRB 20180301 e FRB 20201124).
    p0 = replace(P0(), time_factor=int(record["preprocessing"]["time_factor"]))
    path = Path(record["fits_path"])
    layout = astropy_reader.read_layout(path)
    start = int(round(record["preprocessing"]["window_start_seconds"] / layout.tsamp_s))
    block = astropy_reader.read_block(
        path, start_sample=start, stop_sample=start + p0.window_samples, layout=layout
    )
    window = process_variant(block, layout.tsamp_s, variant, p0)
    image_path = dataset_dir / "images" / "samples" / Path(record["image_path"]).name
    render(window, image_path, p0)
    return {
        **{key: record[key] for key in ("sample_id", "source_sample_id", "fits_path", "true_label")},
        "image_path": image_path.as_posix(),
        "real_parameters": record["real_parameters"],
        "simulation_parameters": record["simulation_parameters"],
        "preprocessing_variant": variant,
        "window_start_seconds": record["preprocessing"]["window_start_seconds"],
        "channels_removed": window.channels_zapped,
        "identical_to_paper_png": image_path.read_bytes() == Path(record["image_path"]).read_bytes(),
    }


def _render(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    records = read_jsonl(PAPER_MANIFEST)
    for variant in args.variants:
        dataset_dir = Path(f"dataset_real_e4_{variant.lower()}")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rendered = list(
                tqdm(
                    pool.map(
                        render_record,
                        records,
                        itertools.repeat(variant),
                        itertools.repeat(dataset_dir),
                        chunksize=4,
                    ),
                    total=len(records),
                    desc=f"E4 {variant}",
                )
            )
        manifest_path = dataset_dir / "metadata" / "image_manifest.jsonl"
        write_jsonl(manifest_path, rendered)
        identical = sum(r["identical_to_paper_png"] for r in rendered)
        write_json(
            OUTPUT_DIR / f"render_{variant.lower()}.json",
            run_manifest(
                experiment=f"E4-render-{variant}",
                project_root=project_root,
                variant=variant,
                paper_manifest={"path": PAPER_MANIFEST.as_posix(), "sha256": sha256_file(PAPER_MANIFEST)},
                manifest={"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)},
                n=len(rendered),
                identical_to_paper_png=identical,
                channels_removed_median=float(np.median([r["channels_removed"] for r in rendered])),
            ),
        )
        print(f"{variant}: {len(rendered)} PNGs; {identical} identicos aos do artigo")
        if variant == "P0" and identical != len(rendered):
            raise SystemExit("A variante P0 nao reproduz o artigo; P1 e P2 nao foram geradas.")


def variant_output(model: str, variant: str) -> Path:
    return OUTPUT_DIR / f"{model}_{variant.lower()}_binary_full" / "predictions.jsonl"


def multiplies_false_positives(p0: Run, p2: Run) -> dict[str, Any]:
    """C6: FP(P2) >= 2 FP(P0), com mais FP no P2 e McNemar exato p < 0,05 nos negativos."""
    comparison = versus(p2, p0, bootstrap=False)
    fp = comparison["false_positives"]
    p_value = comparison["specificity"]["mcnemar"]["p_value_exact"]
    return {
        "fp_p0": fp["second"],
        "fp_p2": fp["first"],
        "ratio": fp["first"] / fp["second"] if fp["second"] else None,
        "mcnemar_p": p_value,
        "multiplies": fp["first"] > fp["second"]
        and fp["first"] >= FP_MULTIPLIER * fp["second"]
        and p_value < ALPHA,
    }


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    sigma_of = {key: float(row["peak_sigma"]) for key, row in read_visibility(VISIBILITY).items()}
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {"visibility": {"path": VISIBILITY.as_posix(), "sha256": sha256_file(VISIBILITY)}}
    for model in args.models:
        paper = PAPER_RUNS[model]
        runs = {"P0": load(paper), **{v: load(variant_output(model, v)) for v in ("P1", "P2")}}
        results[model] = {
            "variants": {
                variant: {**summary(run), "frb_recall_by_sigma_bin": recall_by_bin(run, sigma_of)}
                for variant, run in runs.items()
            },
            **{
                f"{variant}_minus_P0": {
                    **versus(runs[variant], runs["P0"]),
                    "frb_recall_by_sigma_bin": recall_by_bin_paired(runs[variant], runs["P0"], sigma_of),
                }
                for variant in ("P1", "P2")
            },
            "c6_p2_multiplies_fp": multiplies_false_positives(runs["P0"], runs["P2"]),
        }
        inputs[model] = {
            variant: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for variant, path in (("P0", paper), *((v, variant_output(model, v)) for v in ("P1", "P2")))
        }
    write_json(
        OUTPUT_DIR / ("e4_results.json" if len(args.models) == len(PAPER_RUNS) else f"e4_results_{'_'.join(args.models)}.json"),
        run_manifest(
            experiment="E4-analysis", project_root=project_root, inputs=inputs,
            alpha=ALPHA, fp_multiplier=FP_MULTIPLIER, results=results,
        ),
    )
    for model, entry in results.items():
        for variant, values in entry["variants"].items():
            rates = values["binary_0.5"]
            print(f"{model} {variant}: FP {rates['fp']}/{rates['fp'] + rates['tn']}, "
                  f"recall {rates['recall']['estimate']:.3f}, acuracia {rates['accuracy']['estimate']:.4f}")
        print(f"{model}: o P2 multiplica os FP? {entry['c6_p2_multiplies_fp']['multiplies']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render", help="Renderizar as variantes.")
    render_parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    render_parser.add_argument("--workers", type=int, default=16)
    render_parser.set_defaults(func=_render)
    analyze = subparsers.add_parser("analyze", help="P1 e P2 contra o P0 do artigo.")
    analyze.add_argument("--models", nargs="+", choices=sorted(PAPER_RUNS), default=sorted(PAPER_RUNS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
