"""E2: injecao-recuperacao em fundos reais (plano, secao 10.4; definicoes na 10.12).

  1. ``plan``: sorteia (seed 20261013) 1024 janelas de fundo entre os 1000 negativos do
     artigo e atribui a cada uma uma celula da grade (8 sigma x 4 larguras x 2 bandas x
     2 espalhamentos, 8 replicas) e os parametros continuos (DM, chegada, banda).
  2. ``render``: para cada janela, o PNG com o pulso e o controle sem pulso, pelo P0, e o
     sigma medido nos dois com o diagnostico dos bursts reais, no DM injetado.
  3. ``check-control``: o mesmo caminho, sem pulso, nas 1000 janelas negativas do artigo,
     comparado byte a byte com dataset_real_full.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from check_burst_visibility import burst_morphology
from prepare_real import reader as astropy_reader
from prepare_real.preprocess import block_average
from revision_v4.common import (
    REAL_SEED,
    RESULTS_ROOT,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)
from revision_v4.injection import (
    P0,
    Pulse,
    dedispersed_profile,
    expected_sigma_per_unit,
    process_window,
    pulse_unit,
    render,
    zscore,
)


OUTPUT_DIR = RESULTS_ROOT / "e2"
PLAN_PATH = OUTPUT_DIR / "injection_plan.jsonl"
DATASET_DIR = Path("dataset_real_e2")
REAL_MANIFEST = Path("fast_frex/real_manifest_full.csv")
PAPER_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")

TARGET_SIGMAS = (3, 5, 8, 12, 15, 20, 30, 50)
WIDTHS_MS = (1, 3, 10, 30)
BANDS = ("broadband", "limited")
SCATTERING = ("none", "tau5ms")
REPLICAS = 8
DM_RANGE = (350.0, 650.0)
BAND_FWHM_RANGE_MHZ = (100.0, 400.0)
TAU_1250_S = 0.005
# folga entre o sweep inteiro e as bordas da janela (> 3 FWHM do pulso mais largo)
EDGE_MARGIN_S = 0.1
# Um fundo com dado decodificado nao finito ou absurdo (DAT_SCL corrompido numa
# sub-integracao) nao testa a deteccao: o inicio da janela e sorteado de novo, num fluxo
# proprio, sem mexer nas outras janelas. Os dados validos sao de 8 bits (|x| <= 255).
MAX_ABS_DECODED = 1e6
MAX_REDRAWS = 100

# Fluxos independentes do gerador, um por sorteio, para que mudar um nao mude os outros.
_STREAMS = {"files": 0, "starts": 1, "cells": 2, "dm": 3, "arrival": 4, "band": 5, "names": 6, "redraw": 7}


def _rng(stream: str) -> np.random.Generator:
    return np.random.default_rng([REAL_SEED, _STREAMS[stream]])


def grid_cells() -> list[dict[str, Any]]:
    return [
        {"target_sigma": s, "width_ms": w, "band": b, "scattering": c}
        for s, w, b, c in itertools.product(TARGET_SIGMAS, WIDTHS_MS, BANDS, SCATTERING)
    ]


def negative_files(path: Path = REAL_MANIFEST) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return sorted(row["fits_path"] for row in csv.DictReader(handle) if row["label"] != "FRB")


def build_plan(files: list[str], layouts: dict[str, Any], p0: P0) -> list[dict[str, Any]]:
    """1024 injecoes: todos os negativos uma vez e mais 24 sorteados, em ordem aleatoria."""
    cells = [cell for cell in grid_cells() for _ in range(REPLICAS)]
    n = len(cells)
    order = list(_rng("files").permutation(len(files)))
    order += list(_rng("files").permutation(len(files))[: n - len(files)])
    cell_order = _rng("cells").permutation(n)
    starts_rng, dm_rng, arrival_rng, band_rng = (
        _rng("starts"), _rng("dm"), _rng("arrival"), _rng("band")
    )
    names = _rng("names").permutation(2 * n)

    plan = []
    for index, (file_index, cell_index) in enumerate(zip(order, cell_order)):
        fits_path = files[int(file_index)]
        layout = layouts[fits_path]
        cell = cells[int(cell_index)]
        start = int(starts_rng.integers(0, layout["nsamp_total"] - p0.window_samples + 1))
        window_s = p0.window_samples * layout["tsamp_s"]
        dm = float(dm_rng.uniform(*DM_RANGE))
        span = Pulse(t_top_s=0.0, dm=dm, fwhm_s=0.0).sweep_s(layout["f_low_mhz"], layout["f_high_mhz"])[1]
        t_top = float(arrival_rng.uniform(EDGE_MARGIN_S, window_s - span - EDGE_MARGIN_S))
        band_center = band_fwhm = None
        fwhm_draw, center_draw = band_rng.uniform(size=2)  # sorteados sempre, para fluxo fixo
        if cell["band"] == "limited":
            band_fwhm = float(BAND_FWHM_RANGE_MHZ[0] + fwhm_draw * np.diff(BAND_FWHM_RANGE_MHZ)[0])
            low, high = layout["visible_low_mhz"], layout["visible_high_mhz"]
            band_center = float(low + center_draw * (high - low))
        plan.append(
            {
                "injection_id": f"inj_{index:04d}",
                "fits_path": fits_path,
                "start_sample": start,
                **cell,
                "dm": dm,
                "t_top_s": t_top,
                "fwhm_s": cell["width_ms"] / 1e3,
                "band_center_mhz": band_center,
                "band_fwhm_mhz": band_fwhm,
                "tau_1250_s": TAU_1250_S if cell["scattering"] == "tau5ms" else 0.0,
                "injected_sample_id": f"sample_{int(names[2 * index]):06d}",
                "control_sample_id": f"sample_{int(names[2 * index + 1]):06d}",
            }
        )
    return plan


def window_is_clean(fits_path: str, start_sample: int, n_samples: int) -> bool:
    path = Path(fits_path)
    layout = astropy_reader.read_layout(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        block = astropy_reader.read_block(
            path, start_sample=start_sample, stop_sample=start_sample + n_samples, layout=layout
        )
        data = block.data
        return bool(np.isfinite(data).all() and np.abs(data).max() <= MAX_ABS_DECODED)


def redraw_corrupted(
    plan: list[dict[str, Any]],
    layouts: dict[str, Any],
    p0: P0,
    *,
    stream: tuple[int, ...],
    workers: int,
    is_clean=window_is_clean,
) -> list[dict[str, Any]]:
    """Troca o inicio das janelas com dado corrompido e registra o original."""
    arguments = (
        [entry["fits_path"] for entry in plan],
        [entry["start_sample"] for entry in plan],
        itertools.repeat(p0.window_samples),
    )
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            clean = list(pool.map(is_clean, *arguments, chunksize=8))
    else:
        clean = list(map(is_clean, *arguments))
    for index, (entry, ok) in enumerate(zip(plan, clean)):
        original, attempt = entry["start_sample"], 0
        while not ok:
            attempt += 1
            if attempt > MAX_REDRAWS:
                raise ValueError(f"{entry['fits_path']}: sem janela limpa em {MAX_REDRAWS} sorteios.")
            rng = np.random.default_rng([REAL_SEED, *stream, index, attempt])
            limit = layouts[entry["fits_path"]]["nsamp_total"] - p0.window_samples + 1
            entry["start_sample"] = int(rng.integers(0, limit))
            ok = is_clean(entry["fits_path"], entry["start_sample"], p0.window_samples)
        if attempt:
            entry["redrawn"] = {"original_start_sample": original, "attempts": attempt}
    return plan


def _layout_summary(fits_path: str, p0: P0) -> dict[str, Any]:
    layout = astropy_reader.read_layout(Path(fits_path))
    freqs = np.sort(np.asarray(layout.frequencies_mhz, dtype=float))
    return {
        "nsamp_total": int(layout.nsamp_total),
        "tsamp_s": float(layout.tsamp_s),
        "f_low_mhz": float(freqs[0]),
        "f_high_mhz": float(freqs[-1]),
        "visible_low_mhz": float(freqs[p0.edge_channels]),
        "visible_high_mhz": float(freqs[freqs.size - 1 - p0.edge_channels]),
    }


def _plan(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    p0 = P0()
    files = negative_files()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(_layout_summary, files, itertools.repeat(p0), chunksize=16))
    layouts = dict(zip(files, summaries))
    plan = redraw_corrupted(
        build_plan(files, layouts, p0), layouts, p0, stream=(_STREAMS["redraw"],), workers=args.workers
    )
    write_jsonl(PLAN_PATH, plan)
    write_json(
        OUTPUT_DIR / "plan_manifest.json",
        run_manifest(
            experiment="E2-plan",
            project_root=project_root,
            seed=REAL_SEED,
            streams=_STREAMS,
            n_injections=len(plan),
            n_files=len(files),
            grid={
                "target_sigma": TARGET_SIGMAS,
                "width_ms": WIDTHS_MS,
                "band": BANDS,
                "scattering": SCATTERING,
                "replicas": REPLICAS,
            },
            dm_range=DM_RANGE,
            band_fwhm_range_mhz=BAND_FWHM_RANGE_MHZ,
            tau_1250_s=TAU_1250_S,
            edge_margin_s=EDGE_MARGIN_S,
            max_abs_decoded=MAX_ABS_DECODED,
            redrawn={e["injection_id"]: e["redrawn"] for e in plan if "redrawn" in e},
            real_manifest={"path": REAL_MANIFEST.as_posix(), "sha256": sha256_file(REAL_MANIFEST)},
            plan={"path": PLAN_PATH.as_posix(), "sha256": sha256_file(PLAN_PATH)},
        ),
    )
    print(f"{len(plan)} injecoes em {PLAN_PATH}")


def render_injection(entry: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    p0 = P0()
    path = Path(entry["fits_path"])
    layout = astropy_reader.read_layout(path)
    if round(2.0 / layout.tsamp_s / p0.time_bins) != p0.time_factor:
        raise ValueError(f"{path}: amostragem diferente da dos negativos do artigo.")
    block = astropy_reader.read_block(
        path,
        start_sample=entry["start_sample"],
        stop_sample=entry["start_sample"] + p0.window_samples,
        layout=layout,
    )
    if not (np.isfinite(block.data).all() and np.abs(block.data).max() <= MAX_ABS_DECODED):
        raise ValueError(f"{entry['injection_id']}: fundo com dado corrompido; refaca o plano.")
    zscored, keep = zscore(block, p0)
    background = process_window(zscored, keep, block.frequencies_mhz, layout.tsamp_s, p0)
    pulse = Pulse(
        t_top_s=entry["t_top_s"],
        dm=entry["dm"],
        fwhm_s=entry["fwhm_s"],
        band_center_mhz=entry["band_center_mhz"],
        band_fwhm_mhz=entry["band_fwhm_mhz"],
        tau_1250_s=entry["tau_1250_s"],
    )
    unit = pulse_unit(pulse, block.frequencies_mhz, p0.window_samples, layout.tsamp_s)
    pulse_averaged, _, _ = block_average(
        unit * keep[:, None].astype(np.float32),
        keep=keep,
        time_factor=p0.time_factor,
        freq_bins=p0.freq_bins,
    )
    sigma_per_unit = expected_sigma_per_unit(background, pulse_averaged, pulse.dm)
    amplitude = entry["target_sigma"] / sigma_per_unit
    # escala de ruido do perfil dedispersado do fundo (~1 em ruido normal), so como diagnostico
    profile = dedispersed_profile(
        background.image, background.frequencies_mhz, background.bin_seconds, pulse.dm
    )
    background_scale = 1.4826 * float(np.median(np.abs(profile - np.median(profile))))
    injected = process_window(
        zscored, keep, block.frequencies_mhz, layout.tsamp_s, p0, injection=amplitude * unit
    )

    records = []
    for role, window, sample_id in (
        ("injected", injected, entry["injected_sample_id"]),
        ("control", background, entry["control_sample_id"]),
    ):
        image_path = dataset_dir / "images" / "samples" / f"{sample_id}.png"
        render(window, image_path, p0)
        morphology = burst_morphology(
            window.image, window.frequencies_mhz, window.bin_seconds, pulse.dm
        )
        records.append(
            {
                "sample_id": sample_id,
                "image_path": image_path.as_posix(),
                "true_label": "FRB" if role == "injected" else "RFI",
                "role": role,
                "injection_id": entry["injection_id"],
                "fits_path": entry["fits_path"],
                "window_start_seconds": entry["start_sample"] * layout.tsamp_s,
                "measured_sigma": morphology["peak_sigma"],
                "measured_width_ms": morphology["width_ms"],
                "measured_band_fraction": morphology["band_fraction"],
                "channels_zapped": window.channels_zapped,
                "background_profile_scale": background_scale,
            }
        )
    records[0].update(
        {"amplitude": float(amplitude), "expected_sigma_per_unit_amplitude": sigma_per_unit}
    )
    return {"injection_id": entry["injection_id"], "records": records}


def _render(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    plan = read_jsonl(PLAN_PATH)
    if args.limit:
        plan = plan[: args.limit]
    dataset_dir = args.dataset_dir
    if (dataset_dir / "metadata" / "image_manifest.jsonl").exists() and not args.overwrite:
        raise SystemExit(f"{dataset_dir} ja existe; use --overwrite.")
    by_id = {entry["injection_id"]: entry for entry in plan}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = pool.map(render_injection, plan, itertools.repeat(dataset_dir), chunksize=1)
        for result in tqdm(futures, total=len(plan), desc="Injetando", unit="janela"):
            results.append(result)
    manifest = []
    for result in results:
        entry = by_id[result["injection_id"]]
        cell = {key: entry[key] for key in ("target_sigma", "width_ms", "band", "scattering", "dm")}
        for record in result["records"]:
            # o classify exige simulation_parameters em todo registro
            manifest.append(
                {**record, "injection": cell, "simulation_parameters": {"origin": record["role"], **cell}}
            )
    manifest.sort(key=lambda record: record["sample_id"])
    write_jsonl(dataset_dir / "metadata" / "image_manifest.jsonl", manifest)
    write_jsonl(OUTPUT_DIR / "image_manifest.jsonl", manifest)
    write_json(
        OUTPUT_DIR / "render_manifest.json",
        run_manifest(
            experiment="E2-render",
            project_root=project_root,
            p0=vars(P0()) | {"window_samples": P0().window_samples},
            plan={"path": PLAN_PATH.as_posix(), "sha256": sha256_file(PLAN_PATH)},
            n_images=len(manifest),
            dataset_dir=dataset_dir.as_posix(),
        ),
    )
    measured = np.array(
        [[r["injection"]["target_sigma"], r["measured_sigma"]] for r in manifest if r["role"] == "injected"]
    )
    for target in TARGET_SIGMAS:
        values = measured[measured[:, 0] == target, 1]
        print(f"alvo {target:>2}: sigma medido mediano {np.median(values):6.2f} (n = {values.size})")
    redrawn = sorted(entry["injection_id"] for entry in plan if "redrawn" in entry)
    print(f"{len(redrawn)} janelas sorteadas de novo por dado corrompido: {redrawn}")


def _check_one(record: dict[str, Any], output_dir: Path) -> bool:
    """Renderiza a janela do artigo sem pulso e compara com o PNG do artigo."""
    p0 = P0()
    if int(record["preprocessing"]["time_factor"]) != p0.time_factor:
        raise ValueError(f"{record['sample_id']}: fator de tempo diferente do P0 dos negativos.")
    path = Path(record["fits_path"])
    layout = astropy_reader.read_layout(path)
    start = int(round(record["preprocessing"]["window_start_seconds"] / layout.tsamp_s))
    block = astropy_reader.read_block(
        path, start_sample=start, stop_sample=start + p0.window_samples, layout=layout
    )
    zscored, keep = zscore(block, p0)
    window = process_window(zscored, keep, block.frequencies_mhz, layout.tsamp_s, p0)
    image_path = output_dir / f"{record['sample_id']}.png"
    render(window, image_path, p0)
    return image_path.read_bytes() == Path(record["image_path"]).read_bytes()


def _check_control(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    negatives = [r for r in read_jsonl(PAPER_MANIFEST) if r["true_label"] != "FRB"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        same = list(
            tqdm(
                pool.map(_check_one, negatives, itertools.repeat(args.workdir), chunksize=4),
                total=len(negatives),
                desc="Conferindo",
            )
        )
    mismatches = [r["sample_id"] for r, ok in zip(negatives, same) if not ok]
    write_json(
        OUTPUT_DIR / "control_check.json",
        run_manifest(
            experiment="E2-control-check",
            project_root=project_root,
            paper_manifest={"path": PAPER_MANIFEST.as_posix(), "sha256": sha256_file(PAPER_MANIFEST)},
            n=len(negatives),
            identical_png=len(negatives) - len(mismatches),
            mismatches=mismatches[:50],
            passed=not mismatches,
        ),
    )
    print(
        f"{len(negatives) - len(mismatches)}/{len(negatives)} PNGs identicos; "
        f"{'PASSOU' if not mismatches else 'FALHOU'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Sortear as janelas e a grade.")
    plan.add_argument("--workers", type=int, default=16)
    plan.set_defaults(func=_plan)
    render_parser = subparsers.add_parser("render", help="Renderizar injecoes e controles.")
    render_parser.add_argument("--workers", type=int, default=16)
    render_parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    render_parser.add_argument("--limit", type=int, default=0, help="So as primeiras N (teste).")
    render_parser.add_argument("--overwrite", action="store_true")
    render_parser.set_defaults(func=_render)
    check = subparsers.add_parser("check-control", help="Controle sem pulso = P0 do artigo.")
    check.add_argument("--workers", type=int, default=16)
    check.add_argument("--workdir", type=Path, required=True)
    check.set_defaults(func=_check_control)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
