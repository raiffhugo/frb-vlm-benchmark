#!/usr/bin/env python3
"""Consolidate the visibility diagnostics and print the numbers of Section VII D.

Reads three artifacts and produces, in one place, every value quoted in the text:

* `--real`   CSV from `check_burst_visibility.py --negative-dm both`, carrying
             the column `peak_sigma` (blind floor, 24 DMs x 6 boxcars) and
             `peak_sigma_matched` (symmetric floor, 1 DM x 6 boxcars, the same
             number of trials as the positives);
* `--sim`    CSV from `check_burst_visibility_sim.py`, measured on the same
             time-frequency grid as the real one;
* predictions `results_real_binary_full_{2b,4b}/predictions.jsonl`.

The asymmetry in the number of trials is why there are two negative columns: the
maximum over 144 noise samples is systematically larger than the maximum over 6,
so the blind floor inflates the 99th percentile and, with it, the fraction of
positives declared "below the noise". The text should quote the symmetric number
and mention the blind one as an upper bound.

    uv run python analyze_visibility.py \
      --real fast_frex/burst_visibility_full.csv \
      --sim dataset/metadata/burst_visibility_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = [("2b", "Gemma 4 E2B"), ("4b", "Gemma 4 E4B")]
BANDS = [(0.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, np.inf)]
BAND_NAMES = ["<5", "5-8", "8-15", ">15"]


def load_predictions() -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for key, _ in MODELS:
        path = ROOT / f"results_real_binary_full_{key}" / "predictions.jsonl"
        with path.open() as handle:
            predictions[key] = {
                (record := json.loads(line))["sample_id"]: record["predicted_label"]
                for line in handle
                if line.strip()
            }
    return predictions


def rows_of(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def band_of(value: float) -> int:
    for index, (low, high) in enumerate(BANDS):
        if low <= value < high:
            return index
    return len(BANDS) - 1


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def report_real(real_rows: list[dict[str, str]], predictions) -> dict:
    positives = [r for r in real_rows if r["true_label"] == "FRB"]
    negatives = [r for r in real_rows if r["true_label"] != "FRB"]
    pos_sigma = np.array([float(r["peak_sigma"]) for r in positives])
    neg_blind = np.array([float(r["peak_sigma"]) for r in negatives])
    has_matched = "peak_sigma_matched" in real_rows[0]
    neg_matched = (
        np.array([float(r["peak_sigma_matched"]) for r in negatives]) if has_matched else None
    )

    section("REAL — dedispersed peak distributions")
    print(f"positives n={pos_sigma.size}  median={np.median(pos_sigma):.1f}σ")
    print(f"negatives n={neg_blind.size}  median(blind)={np.median(neg_blind):.1f}σ  p99={np.percentile(neg_blind, 99):.1f}σ")
    if neg_matched is not None:
        print(f"negatives            median(matched)={np.median(neg_matched):.1f}σ  p99={np.percentile(neg_matched, 99):.1f}σ")
        print(f"  inflation of the blind floor over the matched one: p99 {np.percentile(neg_blind, 99) / np.percentile(neg_matched, 99):.2f}x")

    out = {}
    for name, floor in [("blind", neg_blind)] + ([("matched", neg_matched)] if neg_matched is not None else []):
        p99 = float(np.percentile(floor, 99))
        below = int(np.count_nonzero(pos_sigma < p99))
        print(f"  positives below the {name} p99: {below}/{pos_sigma.size} = {below / pos_sigma.size:.1%}")
        out[name] = {"p99": p99, "below": below}

    section("REAL — recall per sigma band")
    header = f"{'band':>8} {'n':>5}" + "".join(f"{label:>22}" for _, label in MODELS)
    print(header)
    bands = {}
    for index, (low, high) in enumerate(BANDS):
        ids = [r["sample_id"] for r in positives if low <= float(r["peak_sigma"]) < high]
        line = f"{BAND_NAMES[index]:>8} {len(ids):>5}"
        bands[BAND_NAMES[index]] = {"n": len(ids)}
        for key, _ in MODELS:
            hits = sum(1 for s in ids if predictions[key][s] == "FRB")
            rate = hits / len(ids) if ids else 0.0
            line += f"{f'{hits}/{len(ids)} = {rate:.3f}':>22}"
            bands[BAND_NAMES[index]][key] = rate
        print(line)

    section("REAL — per source")
    sources = sorted({r["source"] for r in positives if r["source"]})
    for source in sources:
        subset = [r for r in positives if r["source"] == source]
        sigma = np.array([float(r["peak_sigma"]) for r in subset])
        dm = np.array([float(r["dm"]) for r in subset])
        line = f"{source:>14} n={len(subset):>4} DM_med={np.median(dm):7.1f} peak_med={np.median(sigma):6.1f}σ"
        for key, _ in MODELS:
            hits = sum(1 for r in subset if predictions[key][r["sample_id"]] == "FRB")
            line += f"  {key}={hits / len(subset):.3f}"
        print(line)

    print("\n  controlling for brightness (only > 15σ):")
    for source in sources:
        subset = [r for r in positives if r["source"] == source and float(r["peak_sigma"]) >= 15.0]
        if not subset:
            continue
        line = f"{source:>14} n={len(subset):>4}"
        for key, _ in MODELS:
            hits = sum(1 for r in subset if predictions[key][r["sample_id"]] == "FRB")
            line += f"  {key}={hits}/{len(subset)} = {hits / len(subset):.3f}"
        print(line)

    return {"pos_sigma": pos_sigma, "bands": bands, "floors": out}


def report_sim(sim_rows: list[dict[str, str]], real_pos: np.ndarray) -> None:
    frb = [r for r in sim_rows if r["true_label"] == "FRB"]
    contained = [r for r in frb if r.get("sweep_contained") == "1"]
    sigma_all = np.array([float(r["peak_sigma"]) for r in frb])
    sigma = np.array([float(r["peak_sigma"]) for r in contained])

    section("SIMULATED — dedispersed peak of the injected FRBs (same grid as the real one)")
    print(f"all simulated FRBs             n={sigma_all.size}  median={np.median(sigma_all):.0f}σ")
    print(f"whole sweep inside the file    n={sigma.size}  median={np.median(sigma):.0f}σ")
    if sigma.size:
        print(f"  quartiles: {np.percentile(sigma, 25):.0f}σ / {np.median(sigma):.0f}σ / {np.percentile(sigma, 75):.0f}σ")
        print(f"  min={sigma.min():.0f}σ  max={sigma.max():.0f}σ")
        print(f"  fraction below 15σ: {np.count_nonzero(sigma < 15.0)}/{sigma.size}")

    section("CONTROLLED COMPARISON — real vs simulated")
    print(f"real median (600 catalogued):          {np.median(real_pos):8.1f}σ")
    print(f"simulated median (sweep contained):    {np.median(sigma):8.1f}σ")
    print(f"ratio of the medians:                  {np.median(sigma) / np.median(real_pos):8.1f}x")
    print()
    print("distribution of the simulated set over the SAME bands as the real one:")
    for index, (low, high) in enumerate(BANDS):
        n = int(np.count_nonzero((sigma >= low) & (sigma < high)))
        print(f"  {BAND_NAMES[index]:>6}σ : {n:5d}  ({n / sigma.size:5.1%})")
    print()
    n_real_bright = int(np.count_nonzero(real_pos >= 15.0))
    print(f"real  >15σ: {n_real_bright}/{real_pos.size} = {n_real_bright / real_pos.size:.1%} of the positives")
    n_sim_bright = int(np.count_nonzero(sigma >= 15.0))
    print(f"simul >15σ: {n_sim_bright}/{sigma.size} = {n_sim_bright / sigma.size:.1%} of the positives")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, default=ROOT / "fast_frex" / "burst_visibility_full.csv")
    parser.add_argument("--sim", type=Path, default=ROOT / "dataset" / "metadata" / "burst_visibility_sim.csv")
    args = parser.parse_args()

    predictions = load_predictions()
    real_rows = rows_of(args.real)
    summary = report_real(real_rows, predictions)

    if args.sim.exists():
        report_sim(rows_of(args.sim), summary["pos_sigma"])
    else:
        print(f"\n[warning] {args.sim} does not exist yet; simulated section omitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
