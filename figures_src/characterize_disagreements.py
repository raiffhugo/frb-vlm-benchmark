"""Characterize the VLM-vs-SwinYNet binary disagreements (for Sect. 6.6).

Read-only analysis. For each model (2b, 4b) it reads
``results_comparison_{key}_0t/disagreement_samples.csv`` and the dataset image
manifest and prints:
  * the split into VLM-exclusive-correct and SwinYNet-exclusive-correct;
  * the breakdown by source class (and RFI subtype) of each split;
  * the flux-density statistics of the FRBs the VLM missed (faintness test).
These numbers back the qualitative characterization written in the article.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The binary run that produced disagreement_samples.csv was fed
# dataset_binary/metadata/image_manifest.jsonl (its image_path points into
# dataset/images/). Joining the disagreements against any other render would
# attach the wrong rfi_category to each sample_id, since the binary selection
# draws different negatives.
MANIFEST = ROOT / "dataset_binary" / "metadata" / "image_manifest.jsonl"


def manifest_index() -> dict[str, dict]:
    idx = {}
    with MANIFEST.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                idx[rec["sample_id"]] = rec
    return idx


def percentile(values, p: float) -> float:
    s = sorted(values)
    i = max(0, min(len(s) - 1, round(p * (len(s) - 1))))
    return s[i]


def main() -> None:
    man = manifest_index()
    frb_flux = [
        r["simulation_parameters"]["flux_density"]
        for r in man.values()
        if r["true_label"] == "FRB"
    ]
    p25 = percentile(frb_flux, 0.25)
    print(
        f"All FRB flux_density: n={len(frb_flux)} "
        f"median={statistics.median(frb_flux):.2f} p25={p25:.2f} max={max(frb_flux):.2f}"
    )

    for key in ("2b", "4b"):
        csv_path = ROOT / f"results_comparison_{key}_0t" / "disagreement_samples.csv"
        rows = list(csv.DictReader(csv_path.open()))
        vlm_only = [r for r in rows if r["vlm_correct"] == "True" and r["external_correct"] == "False"]
        ext_only = [r for r in rows if r["external_correct"] == "True" and r["vlm_correct"] == "False"]
        print(
            f"\n=== Gemma 4 E{key.upper()} ===  disagreements={len(rows)}  "
            f"VLM-only-correct={len(vlm_only)}  SwinYNet-only-correct={len(ext_only)}"
        )

        def by_class(group):
            return dict(Counter(r["source_true_label"] for r in group))

        def rfi_subtypes(group):
            return dict(
                Counter(
                    man[r["sample_id"]]["simulation_parameters"].get("rfi_category", "?")
                    for r in group
                    if r["source_true_label"] == "RFI"
                )
            )

        print(f"  VLM-only-correct by class     : {by_class(vlm_only)}")
        print(f"    RFI subtypes                : {rfi_subtypes(vlm_only)}")
        print(f"  SwinYNet-only-correct by class: {by_class(ext_only)}")

        missed = [
            man[r["sample_id"]]["simulation_parameters"]["flux_density"]
            for r in ext_only
            if r["source_true_label"] == "FRB"
        ]
        if missed:
            below = sum(1 for f in missed if f <= p25)
            print(
                f"  FRBs missed by VLM: n={len(missed)} "
                f"median_flux={statistics.median(missed):.2f} max={max(missed):.2f} "
                f"below_global_p25={below}/{len(missed)}"
            )


if __name__ == "__main__":
    main()
