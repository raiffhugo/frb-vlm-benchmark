"""R0, item 8: largura e ocupacao espectral dos 600 bursts reais (plano, secao 10.2).

Reconstroi o array decimado de cada burst com a janela gravada no manifesto do artigo
(`check_burst_visibility.rebuild_image`), mede `burst_morphology` no DM do catalogo e
confere que o pico reproduz o `peak_sigma` de fast_frex/burst_visibility_full.csv. Depois
estratifica o recall dos dois VLMs por largura e por ocupacao, dentro dos bins de sigma.
"""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from check_burst_visibility import burst_morphology, rebuild_image
from revision_v4.common import RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json
from revision_v4.r0_reanalysis import load_binary_run
from revision_v4.stats import wilson


OUTPUT_DIR = RESULTS_ROOT / "r0_morphology"
REAL_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")
VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
PREDICTIONS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
TABLE = OUTPUT_DIR / "burst_morphology_real.csv"
# Classes fixadas antes de ver o resultado. Bins de 1,97 ms: boxcars de 1-2 bins (<= 4 ms),
# 4-8 (8-16 ms) e 16-32 (>= 31 ms). A ocupacao so e usada acima de 10 sigma.
WIDTH_CLASSES = (("<=4 ms", 1, 2), ("8-16 ms", 4, 8), (">=31 ms", 16, 32))
BAND_SIGMA_FLOOR = 10.0
BAND_CLASSES = (("band-limited (<0.5)", 0.0, 0.5), ("broad (>=0.5)", 0.5, 1.01))
SIGMA_STRATA = ((0.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, math.inf))


def _measure(record: dict[str, Any]) -> dict[str, Any]:
    image, freqs, bin_seconds = rebuild_image(record)
    dm = float(record["real_parameters"]["dm"])
    return {
        "sample_id": record["sample_id"],
        "source_sample_id": record["source_sample_id"],
        "source": record["real_parameters"].get("source") or "",
        "dm": dm,
        **burst_morphology(image, freqs, bin_seconds, dm),
    }


def _stratify(rows: list[dict[str, Any]], hits: dict[str, dict[str, bool]]) -> dict[str, Any]:
    def summary(members: list[dict[str, Any]]) -> dict[str, Any]:
        entry: dict[str, Any] = {"n": len(members)}
        for vlm, by_id in hits.items():
            k = sum(by_id[row["sample_id"]] for row in members)
            entry[vlm] = wilson(k, len(members)).as_dict() if members else None
        return entry

    result: dict[str, Any] = {"width": {}, "band": {}}
    for low, high in SIGMA_STRATA:
        stratum = [row for row in rows if low <= row["peak_sigma"] < high]
        key = f"[{low:g}, {high:g})" if math.isfinite(high) else f">= {low:g}"
        result["width"][key] = {
            name: summary([r for r in stratum if lo <= r["width_bins"] <= hi])
            for name, lo, hi in WIDTH_CLASSES
        }
        if low >= BAND_SIGMA_FLOOR or high > BAND_SIGMA_FLOOR:
            eligible = [r for r in stratum if r["peak_sigma"] >= BAND_SIGMA_FLOOR]
            result["band"][key] = {
                name: summary([r for r in eligible if lo <= r["band_fraction"] < hi])
                for name, lo, hi in BAND_CLASSES
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    project_root = Path.cwd().resolve()

    records = [r for r in read_jsonl(REAL_MANIFEST) if r["true_label"] == "FRB"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_measure, records, chunksize=4))

    reference = {
        row["sample_id"]: float(row["peak_sigma"])
        for row in csv.DictReader(VISIBILITY.open(encoding="utf-8"))
    }
    mismatches = [
        row["sample_id"] for row in rows if round(row["peak_sigma"], 2) != reference[row["sample_id"]]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            )

    hits = {}
    for vlm, path in PREDICTIONS.items():
        run = load_binary_run(path)
        hits[vlm] = {sid: bool(score >= 0.5) for sid, score in zip(run.sample_ids, run.scores)}
    write_json(
        OUTPUT_DIR / "morphology_summary.json",
        run_manifest(
            experiment="R0-item8-morphology",
            project_root=project_root,
            inputs={
                "manifest": {"path": REAL_MANIFEST.as_posix(), "sha256": sha256_file(REAL_MANIFEST)},
                "visibility": {"path": VISIBILITY.as_posix(), "sha256": sha256_file(VISIBILITY)},
            },
            table_sha256=sha256_file(TABLE),
            n_bursts=len(rows),
            peak_sigma_reproduced=len(rows) - len(mismatches),
            peak_sigma_mismatches=mismatches[:20],
            classes={
                "width": WIDTH_CLASSES,
                "band": BAND_CLASSES,
                "band_sigma_floor": BAND_SIGMA_FLOOR,
            },
            recall=_stratify(rows, hits),
        ),
    )
    print(f"{len(rows)} bursts; peak_sigma reproduzido em {len(rows) - len(mismatches)}")


if __name__ == "__main__":
    main()
