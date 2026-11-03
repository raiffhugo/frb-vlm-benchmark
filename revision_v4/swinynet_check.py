"""Verificacao do SwinYNet reinstalado (plano, secao 10.3).

  1. ``export``: sorteia 100 amostras do benchmark binario (seed 42) e exporta os FITS
     pelo mesmo BenchmarkExporter do artigo, para um diretorio de trabalho em /var/tmp.
  2. SwinYNet (commit e comando em results_v4/swinynet/README.md).
  3. ``compare``: importa a saida (limiar 0,5, agregacao max) e compara cada
     frb_probability com results_external_binary; criterio |delta| < 1e-6.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from benchmark_export.converter import BenchmarkExportConfig, BenchmarkExporter
from benchmark_predictions.importer import (
    ExternalPredictionImportConfig,
    ExternalPredictionImporter,
)
from revision_v4.common import (
    GENERATION_SEED,
    RESULTS_ROOT,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)


OUTPUT_DIR = RESULTS_ROOT / "swinynet"
BINARY_LABELS = Path("dataset_binary/metadata/labels.jsonl")
BINARY_MANIFEST = Path("dataset_binary/metadata/image_manifest.jsonl")
REFERENCE_PREDICTIONS = Path("results_external_binary/predictions.jsonl")
SUBSET_LABELS = OUTPUT_DIR / "check100_labels.jsonl"
CHECK_PREDICTIONS = OUTPUT_DIR / "check100_predictions.jsonl"
CHECK_COMPARISON = OUTPUT_DIR / "check100_comparison.json"
TOLERANCE = 1e-6


def select_subset(records: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    """Sorteio estavel: independe da ordem de entrada e preserva a ordem por sample_id."""
    ordered = sorted(records, key=lambda record: str(record["sample_id"]))
    chosen = {str(r["sample_id"]) for r in random.Random(seed).sample(ordered, n)}
    return [record for record in ordered if str(record["sample_id"]) in chosen]


def compare_scores(
    reference: dict[str, dict[str, Any]],
    rerun: dict[str, dict[str, Any]],
    *,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    rows = []
    for sample_id, record in sorted(rerun.items()):
        old = reference[sample_id]
        new_p = float(record["parsed_response"]["frb_probability"])
        old_p = float(old["parsed_response"]["frb_probability"])
        rows.append(
            {
                "sample_id": sample_id,
                "true_label": record.get("true_label"),
                "reference_probability": old_p,
                "rerun_probability": new_p,
                "abs_delta": abs(new_p - old_p),
                "reference_candidates": old["parsed_response"].get("n_candidates"),
                "rerun_candidates": record["parsed_response"].get("n_candidates"),
                "label_identical": record.get("predicted_label") == old.get("predicted_label"),
            }
        )
    within = [row for row in rows if row["abs_delta"] < tolerance]
    return {
        "n": len(rows),
        "tolerance": tolerance,
        "within_tolerance": len(within),
        "max_abs_delta": max((row["abs_delta"] for row in rows), default=None),
        "candidate_count_identical": sum(
            row["reference_candidates"] == row["rerun_candidates"] for row in rows
        ),
        "label_identical": sum(row["label_identical"] for row in rows),
        "passed": bool(rows) and len(within) == len(rows),
        "outside_tolerance": [row for row in rows if row["abs_delta"] >= tolerance],
    }


def _export(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    subset = select_subset(read_jsonl(BINARY_LABELS), n=args.n, seed=args.seed)
    write_jsonl(SUBSET_LABELS, subset)
    exporter = BenchmarkExporter(
        config=BenchmarkExportConfig(labels_path=SUBSET_LABELS, output_dir=args.workdir),
        project_root=project_root,
    )
    exported = exporter.export(overwrite=False)
    write_json(
        OUTPUT_DIR / "export_manifest.json",
        run_manifest(
            experiment="SwinYNet-check100-export",
            project_root=project_root,
            subset={"n": args.n, "seed": args.seed, "sha256": sha256_file(SUBSET_LABELS)},
            workdir=str(args.workdir),
            exported=len(exported),
        ),
    )
    print(f"{len(exported)} FITS exportados em {args.workdir}")


def _compare(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    importer = ExternalPredictionImporter(
        config=ExternalPredictionImportConfig(
            benchmark_manifest=args.workdir / "metadata" / "benchmark_manifest.jsonl",
            external_predictions=args.external,
            output_path=CHECK_PREDICTIONS,
            threshold=0.5,
            aggregation="max",
        ),
        project_root=project_root,
    )
    imported = importer.import_predictions(overwrite=True)

    # sample_id do export (sample_000000..099) -> fonte (frb_00000) -> id do benchmark
    export_manifest = read_jsonl(args.workdir / "metadata" / "benchmark_manifest.jsonl")
    source_by_export = {r["sample_id"]: r["source_sample_id"] for r in export_manifest}
    binary_by_source = {
        r["source_sample_id"]: r["sample_id"] for r in read_jsonl(BINARY_MANIFEST)
    }
    rerun = {
        binary_by_source[source_by_export[record["sample_id"]]]: record
        for record in imported
    }
    reference = {r["sample_id"]: r for r in read_jsonl(REFERENCE_PREDICTIONS)}
    result = compare_scores(reference, rerun)
    write_json(
        CHECK_COMPARISON,
        run_manifest(
            experiment="SwinYNet-check100",
            project_root=project_root,
            reference_predictions={
                "path": REFERENCE_PREDICTIONS.as_posix(),
                "sha256": sha256_file(REFERENCE_PREDICTIONS),
            },
            external_output_sha256=sha256_file(args.external),
            **result,
        ),
    )
    print(
        f"{result['within_tolerance']}/{result['n']} dentro de {TOLERANCE:g}; "
        f"max |delta| = {result['max_abs_delta']:.3g}; "
        f"{'PASSOU' if result['passed'] else 'FALHOU'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Exportar o subconjunto de checagem.")
    export.add_argument("--n", type=int, default=100)
    export.add_argument("--seed", type=int, default=GENERATION_SEED)
    export.add_argument("--workdir", type=Path, required=True)
    export.set_defaults(func=_export)
    compare = subparsers.add_parser("compare", help="Comparar com a rodada do artigo.")
    compare.add_argument("--workdir", type=Path, required=True)
    compare.add_argument("--external", type=Path, required=True)
    compare.set_defaults(func=_compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
