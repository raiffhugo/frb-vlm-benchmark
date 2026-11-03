"""E8: renderizacao equivalente do benchmark simulado (plano, secao 10.10; regra na 10.12).

As 2000 amostras do binario sao re-renderizadas a partir do array da grade comum de
``check_burst_visibility_sim.load_spectrum``: z-score robusto por canal, ``block_average`` com
fator 10 no tempo (bins de 1,966 ms, como no real) e 256 canais, e ``flatten_rows``. E o mesmo
array em que o sigma dos bursts simulados foi medido no artigo. A renderizacao e a do artigo
(``render_spectrum_png`` com percentil 1-99, viridis, 1024 x 768). Os FITS simulados tem 10173
amostras; a grade comum descarta as 3 ultimas, o que da 1017 x 256 em vez de 1024 x 256.

``analyze`` compara, pareado pelo sample_id, o caminho decimado com as rodadas binarias do artigo
(recall por bin de sigma com McNemar exato por bin, e a FPR) e os dois caminhos com os 600 bursts
reais (P0 do artigo), so com os simulados de sweep contido e nos bins com >= 10 bursts nos dois
conjuntos. Regra do C3/C10 (secao 10.12): o E8 "explica a diferenca simulado x real" quando ela era
significativa (IC de Newcombe sem o 0) em algum desses bins no caminho original e, no decimado, o
IC contem o 0 em todos.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from check_burst_visibility_sim import build_reader, load_spectrum
from revision_v4.common import RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json, write_jsonl
from revision_v4.injection import P0, Window, render
from revision_v4.paired_runs import (
    load,
    read_visibility,
    recall_by_bin,
    recall_by_bin_paired,
    summary,
    unpaired_by_bin,
    versus,
)


OUTPUT_DIR = RESULTS_ROOT / "e8"
BINARY_MANIFEST = Path("dataset_binary/metadata/image_manifest.jsonl")
DATASET_DIR = Path("dataset_e8_decimated")
PAPER_RUNS = {
    "e2b": Path("results_vlm2b_0t_binary/predictions.jsonl"),
    "e4b": Path("results_vlm_0t_binary/predictions.jsonl"),
}
REAL_RUNS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
SIM_VISIBILITY = Path("dataset/metadata/burst_visibility_sim.csv")
REAL_VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
MIN_BURSTS_PER_BIN = 10


def render_record(record: dict[str, Any]) -> dict[str, Any]:
    p0 = P0()
    project_root = Path.cwd().resolve()
    image, freqs, bin_seconds = load_spectrum(build_reader(project_root), project_root / record["fits_path"])
    window = Window(
        image=image,
        averaged=image,
        frequencies_mhz=freqs,
        bin_seconds=bin_seconds,
        keep=np.ones(freqs.size, dtype=bool),
        channels_zapped=0,
    )
    image_path = DATASET_DIR / "images" / "samples" / Path(record["image_path"]).name
    render(window, image_path, p0)
    return {
        **{key: record[key] for key in ("sample_id", "source_sample_id", "fits_path", "true_label")},
        "image_path": image_path.as_posix(),
        "simulation_parameters": record.get("simulation_parameters"),
        "rendering": {"path": "decimated", "shape": list(image.shape), "bin_seconds": bin_seconds},
    }


def _render(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    records = read_jsonl(BINARY_MANIFEST)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rendered = list(
            tqdm(pool.map(render_record, records, chunksize=8), total=len(records), desc="E8")
        )
    manifest_path = DATASET_DIR / "metadata" / "image_manifest.jsonl"
    write_jsonl(manifest_path, rendered)
    write_json(
        OUTPUT_DIR / "render_manifest.json",
        run_manifest(
            experiment="E8-render",
            project_root=project_root,
            binary_manifest={"path": BINARY_MANIFEST.as_posix(), "sha256": sha256_file(BINARY_MANIFEST)},
            manifest={"path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)},
            n=len(rendered),
            shapes=sorted({tuple(r["rendering"]["shape"]) for r in rendered}),
        ),
    )
    print(f"{len(rendered)} PNGs em {DATASET_DIR}")


def output(model: str) -> Path:
    return OUTPUT_DIR / f"{model}_binary_full" / "predictions.jsonl"


def explains_difference(original: dict[str, Any], decimated: dict[str, Any]) -> dict[str, Any]:
    """Regra do C3/C10 sobre as diferencas simulado - real por bin dos dois caminhos."""
    bins = sorted(set(original) & set(decimated))
    before = [label for label in bins if original[label]["significant"]]
    after = [label for label in bins if decimated[label]["significant"]]
    return {
        "bins": bins,
        "significant_original": before,
        "significant_decimated": after,
        "explains": bool(before) and not after,
    }


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    sim_rows = read_visibility(SIM_VISIBILITY, key="source_sample_id")
    sim_sigma = {key: float(row["peak_sigma"]) for key, row in sim_rows.items()}
    real_sigma = {key: float(row["peak_sigma"]) for key, row in read_visibility(REAL_VISIBILITY).items()}
    source_of = {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(BINARY_MANIFEST)}
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "sim_visibility": {"path": SIM_VISIBILITY.as_posix(), "sha256": sha256_file(SIM_VISIBILITY)},
        "real_visibility": {"path": REAL_VISIBILITY.as_posix(), "sha256": sha256_file(REAL_VISIBILITY)},
    }
    for model in args.models:
        original = load(PAPER_RUNS[model], source_of)
        decimated = load(output(model), source_of)
        real = load(REAL_RUNS[model])
        contained = np.array(
            [is_frb and sim_rows[key]["sweep_contained"] == "1" for key, is_frb in zip(original.source_ids, original.y_true)]
        )
        real_by_bin = recall_by_bin(real, real_sigma)
        sim_minus_real = {
            path: unpaired_by_bin(recall_by_bin(run, sim_sigma, contained), real_by_bin, MIN_BURSTS_PER_BIN)
            for path, run in (("original", original), ("decimated", decimated))
        }
        results[model] = {
            "original": summary(original),
            "decimated": summary(decimated),
            "decimated_minus_original": versus(decimated, original),
            "frb_recall_by_sigma_bin": recall_by_bin_paired(decimated, original, sim_sigma),
            "contained_frb": int(contained.sum()),
            "real_recall_by_sigma_bin": real_by_bin,
            "sim_contained_minus_real": sim_minus_real,
            "c3_c10_e8_explains": explains_difference(sim_minus_real["original"], sim_minus_real["decimated"]),
        }
        inputs[model] = {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in (("original", PAPER_RUNS[model]), ("decimated", output(model)), ("real", REAL_RUNS[model]))
        }
    write_json(
        OUTPUT_DIR / ("e8_results.json" if len(args.models) == len(PAPER_RUNS) else f"e8_results_{'_'.join(args.models)}.json"),
        run_manifest(
            experiment="E8-analysis", project_root=project_root, inputs=inputs,
            min_bursts_per_bin=MIN_BURSTS_PER_BIN, results=results,
        ),
    )
    for model, entry in results.items():
        for path in ("original", "decimated"):
            rates = entry[path]["binary_0.5"]
            print(f"{model} {path}: recall {rates['recall']['estimate']:.3f}, FPR {rates['fpr']['estimate']:.4f}")
        print(f"{model}: o E8 explica a diferenca simulado x real? {entry['c3_c10_e8_explains']['explains']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    render_parser = subparsers.add_parser("render", help="Re-renderizar o binario pela grade comum.")
    render_parser.add_argument("--workers", type=int, default=16)
    render_parser.set_defaults(func=_render)
    analyze = subparsers.add_parser("analyze", help="Decimado x original x real.")
    analyze.add_argument("--models", nargs="+", choices=sorted(PAPER_RUNS), default=sorted(PAPER_RUNS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
