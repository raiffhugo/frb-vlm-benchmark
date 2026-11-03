"""Amostras sorteadas para o E4, o E5 e o E6 (plano, secoes 10.6 a 10.8; regras na 10.12).

Dados reais (seed 20261013, fluxos independentes por amostra), com os ids de
dataset_real_full e as faixas de sigma do diagnostico do artigo:
  - ``e6_r200``: 100 positivos (20 por faixa) e 100 negativos;
  - ``e5_humans``: 200 positivos nas cotas do protocolo e 100 negativos;
  - ``e4_audit``: 200 negativos, disjuntos dos negativos do E5, porque os mesmos avaliadores
    julgam as duas amostras na mesma sessao.
Simulado:
  - ``e6_s200``: as 200 imagens da Tabela 10 (``prompt_timing_experiment.choose_subsample``,
    seed 42), conferidas contra as predicoes da Tabela 10.

Cada amostra sai em results_v4/selections/<nome>.jsonl, como subconjunto do manifesto de
imagens, pronta para ``classify --manifest``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

import prompt_timing_experiment
from revision_v4.common import REAL_SEED, RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json, write_jsonl


OUTPUT_DIR = RESULTS_ROOT / "selections"
REAL_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")
VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
BINARY_MANIFEST = Path("dataset_binary/metadata/image_manifest.jsonl")
TABLE10_PREDICTIONS = Path("results_prompt_timing_2b/predictions.jsonl")

STRATA = ((0.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, float("inf")))
E5_QUOTAS = (25, 45, 60, 45, 25)
R200_QUOTAS = (20, 20, 20, 20, 20)
_STREAMS = {"e6_r200": 0, "e5_humans": 1, "e4_audit": 2}


def stratum_of(sigma: float) -> str:
    for low, high in STRATA:
        if low <= sigma < high:
            return f"[{low:g}, {high:g})" if np.isfinite(high) else f">= {low:g}"
    raise ValueError(f"sigma fora das faixas: {sigma}")


def _draw(ids: list[str], k: int, rng: np.random.Generator) -> list[str]:
    if k > len(ids):
        raise ValueError(f"Faixa com {len(ids)} amostras para uma cota de {k}.")
    return [ids[i] for i in sorted(rng.choice(len(ids), size=k, replace=False))]


def stratified_positives(
    sigma_by_id: dict[str, float], quotas: tuple[int, ...], rng: np.random.Generator
) -> list[str]:
    chosen = []
    for (low, high), quota in zip(STRATA, quotas):
        members = sorted(i for i, s in sigma_by_id.items() if low <= s < high)
        chosen += _draw(members, quota, rng)
    return chosen


def real_selections(
    positives: dict[str, float], negatives: list[str]
) -> dict[str, list[str]]:
    """Ids (do manifesto do artigo) de cada amostra real."""
    rngs = {name: np.random.default_rng([REAL_SEED, stream]) for name, stream in _STREAMS.items()}
    negatives = sorted(negatives)
    e5_negatives = _draw(negatives, 100, rngs["e5_humans"])
    remaining = [i for i in negatives if i not in set(e5_negatives)]
    return {
        "e6_r200": stratified_positives(positives, R200_QUOTAS, rngs["e6_r200"])
        + _draw(negatives, 100, rngs["e6_r200"]),
        "e5_humans": stratified_positives(positives, E5_QUOTAS, rngs["e5_humans"]) + e5_negatives,
        "e4_audit": _draw(remaining, 200, rngs["e4_audit"]),
    }


def _write(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    manifest = {r["sample_id"]: r for r in read_jsonl(REAL_MANIFEST)}
    with VISIBILITY.open(encoding="utf-8") as handle:
        sigma = {row["sample_id"]: float(row["peak_sigma"]) for row in csv.DictReader(handle)}
    positives = {i: sigma[i] for i, r in manifest.items() if r["true_label"] == "FRB"}
    negatives = [i for i, r in manifest.items() if r["true_label"] != "FRB"]

    summary: dict[str, Any] = {}
    for name, ids in real_selections(positives, negatives).items():
        records = []
        for sample_id in sorted(ids):
            record = dict(manifest[sample_id])
            if record["true_label"] == "FRB":
                record["selection"] = {"peak_sigma": sigma[sample_id], "stratum": stratum_of(sigma[sample_id])}
            records.append(record)
        path = OUTPUT_DIR / f"{name}.jsonl"
        write_jsonl(path, records)
        strata: dict[str, int] = {}
        for record in records:
            key = record.get("selection", {}).get("stratum", "negative")
            strata[key] = strata.get(key, 0) + 1
        summary[name] = {"path": path.as_posix(), "sha256": sha256_file(path), "n": len(records), "composition": strata}

    binary = prompt_timing_experiment.load_manifest(BINARY_MANIFEST)
    s200, _ = prompt_timing_experiment.choose_subsample(
        binary, n_frb=100, per_rfi_subtype=10, n_noise=50, seed=42
    )
    table10_ids = sorted(r["sample_id"] for r in read_jsonl(TABLE10_PREDICTIONS))
    if sorted(r["sample_id"] for r in s200) != table10_ids:
        raise SystemExit("O sorteio da Tabela 10 nao reproduz as 200 amostras das predicoes.")
    path = OUTPUT_DIR / "e6_s200.jsonl"
    write_jsonl(path, s200)
    summary["e6_s200"] = {"path": path.as_posix(), "sha256": sha256_file(path), "n": len(s200), "matches_table10": True}

    write_json(
        OUTPUT_DIR / "selections_manifest.json",
        run_manifest(
            experiment="selections-E4-E5-E6",
            project_root=project_root,
            seed=REAL_SEED,
            streams=_STREAMS,
            strata=[[low, None if not np.isfinite(high) else high] for low, high in STRATA],
            quotas={"e5_humans": E5_QUOTAS, "e6_r200": R200_QUOTAS},
            inputs={
                "real_manifest": {"path": REAL_MANIFEST.as_posix(), "sha256": sha256_file(REAL_MANIFEST)},
                "visibility": {"path": VISIBILITY.as_posix(), "sha256": sha256_file(VISIBILITY)},
                "binary_manifest": {"path": BINARY_MANIFEST.as_posix(), "sha256": sha256_file(BINARY_MANIFEST)},
            },
            selections=summary,
        ),
    )
    for name, entry in summary.items():
        print(f"{name}: {entry['n']} {entry.get('composition', '')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write", help="Sortear e gravar as amostras.").set_defaults(func=_write)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
