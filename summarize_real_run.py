#!/usr/bin/env python3
"""Summary of the real experiment: accuracy per model, per source and per visibility.

Cross-references the predictions of both models with the diagnostic of
`check_burst_visibility.py` to separate a model error from the absence of signal
in the image. It neither modifies nor re-runs the pipeline; it only reads the
artifacts already produced.

    uv run python summarize_real_run.py \
      --predictions-2b results_real_binary_n100_2b/predictions.jsonl \
      --predictions-4b results_real_binary_n100_4b/predictions.jsonl \
      --visibility fast_frex/burst_visibility_n100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def load_predictions(path: Path) -> dict[str, dict]:
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["sample_id"]] = record
    return records


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval: unlike the normal approximation, it does not degenerate at 0/n and n/n."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def binary_truth(label: str) -> str:
    return "FRB" if label.upper() == "FRB" else "NON_FRB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-2b", type=Path, required=True)
    parser.add_argument("--predictions-4b", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    args = parser.parse_args()

    models = {"E2B": load_predictions(args.predictions_2b), "E4B": load_predictions(args.predictions_4b)}
    visibility = {row["sample_id"]: row for row in csv.DictReader(args.visibility.open())}

    print("=" * 74)
    print("OVERALL PERFORMANCE (threshold 0.5)")
    print("=" * 74)
    print(f"{'model':<8}{'class':<10}{'hits':>9}{'n':>5}{'rate':>8}   95% CI")
    for name, preds in models.items():
        for target in ("FRB", "NON_FRB"):
            subset = [p for p in preds.values() if binary_truth(p["true_label"]) == target]
            hits = sum(1 for p in subset if p["predicted_label"] == target)
            lo, hi = wilson(hits, len(subset))
            metric = "recall" if target == "FRB" else "specificity"
            print(f"{name:<8}{target:<10}{hits:>9}{len(subset):>5}{hits/len(subset):>8.3f}   [{lo:.3f}, {hi:.3f}]  ({metric})")

    print()
    print("=" * 74)
    print("RECALL PER BURST-VISIBILITY BAND (dedispersed peak)")
    print("=" * 74)
    bands = [(0, 5, "invisible  (<5 sigma)"), (5, 8, "marginal   (5-8)"), (8, 15, "clear      (8-15)"), (15, 1e9, "obvious    (>15)")]
    print(f"{'band':<24}{'n':>4}   " + "".join(f"{name:>14}" for name in models))
    for low, high, label in bands:
        ids = [
            sid for sid, row in visibility.items()
            if row["true_label"] == "FRB" and low <= float(row["peak_sigma"]) < high
        ]
        if not ids:
            continue
        cells = ""
        for preds in models.values():
            hits = sum(1 for sid in ids if preds[sid]["predicted_label"] == "FRB")
            cells += f"{hits:>7}/{len(ids):<7}"
        print(f"{label:<24}{len(ids):>4}   {cells}")

    print()
    print("=" * 74)
    print("RECALL PER SOURCE")
    print("=" * 74)
    sources = sorted({row["source"] for row in visibility.values() if row["source"]})
    print(f"{'source':<16}{'n':>4}   " + "".join(f"{name:>14}" for name in models))
    for source in sources:
        ids = [sid for sid, row in visibility.items() if row["source"] == source]
        cells = ""
        for preds in models.values():
            hits = sum(1 for sid in ids if preds[sid]["predicted_label"] == "FRB")
            cells += f"{hits:>7}/{len(ids):<7}"
        print(f"{source:<16}{len(ids):>4}   {cells}")

    print()
    print("=" * 74)
    print("AGREEMENT BETWEEN THE MODELS")
    print("=" * 74)
    pairs = Counter(
        (models["E2B"][sid]["predicted_label"], models["E4B"][sid]["predicted_label"])
        for sid in models["E2B"]
    )
    agree = sum(count for (a, b), count in pairs.items() if a == b)
    print(f"agree on {agree}/{len(models['E2B'])} images")
    for (a, b), count in sorted(pairs.items()):
        print(f"  E2B={a:<8} E4B={b:<8} {count:>4}")

    print()
    print("=" * 74)
    print("SANITY: signal measured in the negatives (best DM of a blind scan)")
    print("=" * 74)
    for target in ("FRB", "RFI"):
        values = sorted(float(r["peak_sigma"]) for r in visibility.values() if r["true_label"] == target)
        if values:
            mid = values[len(values) // 2]
            print(f"{target:<6} n={len(values):<4} median {mid:6.2f} sigma   min {values[0]:6.2f}   max {values[-1]:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
