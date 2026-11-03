"""E0: reconstroi os PNGs do benchmark simulado e verifica a pipeline ponta a ponta.

Protocolo (plano, secao 10.3):
  1. ``render``: re-renderiza os 3000 PNGs de dataset/fits com os parametros do
     image_manifest.jsonl de 23/06 (percentil 1-99, viridis, 1024x768, dpi 192 padrao,
     sem titulo). O manifesto original nao e tocado; o reconstruido vai para
     results_v4/e0/ e tem de ser identico a ele, registro a registro.
  2. ``select-check``: sorteia 50 imagens do benchmark binario (seed 42).
  3. ``run_pipeline.py classify`` com o E2B nessas 50 (comando em results_v4/e0/README.md).
  4. ``compare``: cada raw_model_response tem de ser identico ao gravado em
     results_vlm2b_0t_binary/predictions.jsonl. Se falhar, E6-E8 esperam.
  ``revisions`` audita as revisoes HF em cache: pesos e entradas tokenizadas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from plot_dataset.plotter import DatasetPlotter, PlotConfig
from revision_v4.common import (
    GENERATION_SEED,
    MODEL_REVISIONS,
    RESULTS_ROOT,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)
from vlm_classifier.classifier import read_git_provenance


OUTPUT_DIR = RESULTS_ROOT / "e0"
REFERENCE_MANIFEST = Path("dataset/metadata/image_manifest.jsonl")
LABELS = Path("dataset/metadata/labels.jsonl")
IMAGES_DIR = Path("dataset/images")
BINARY_MANIFEST = Path("dataset_binary/metadata/image_manifest.jsonl")
REFERENCE_PREDICTIONS = Path("results_vlm2b_0t_binary/predictions.jsonl")

REBUILT_MANIFEST = OUTPUT_DIR / "image_manifest_rebuilt.jsonl"
PNG_HASHES = OUTPUT_DIR / "png_sha256.csv"
CHECK_MANIFEST = OUTPUT_DIR / "check50_manifest.jsonl"
CHECK_PREDICTIONS = OUTPUT_DIR / "check50_predictions.jsonl"
CHECK_COMPARISON = OUTPUT_DIR / "check50_comparison.json"

# Parametros gravados no manifesto de 23/06 (o dpi nao era gravado; o comando do
# historico nao passava --dpi, logo valia o padrao 192).
EXPECTED_PLOT = {
    "anonymized_image_path": True,
    "backend": "astropy",
    "cmap": "viridis",
    "height": 768,
    "normalization": "percentile",
    "title": False,
    "width": 1024,
}
RENDER_DPI = 192
PERCENTILES = (1.0, 99.0)


def compare_manifests(
    reference: list[dict[str, Any]],
    rebuilt: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatched = [
        old.get("sample_id")
        for old, new in zip(reference, rebuilt)
        if old != new
    ]
    return {
        "reference_records": len(reference),
        "rebuilt_records": len(rebuilt),
        "mismatched_records": len(mismatched),
        "first_mismatches": mismatched[:20],
        "identical": len(reference) == len(rebuilt) and not mismatched,
    }


def select_check_records(
    records: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: str(record["sample_id"]))
    chosen = {
        str(record["sample_id"])
        for record in random.Random(seed).sample(ordered, n)
    }
    return [record for record in ordered if str(record["sample_id"]) in chosen]


def compare_responses(
    reference: list[dict[str, Any]],
    rerun: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(record["sample_id"]): record for record in reference}
    rows = []
    for record in rerun:
        old = by_id[str(record["sample_id"])]
        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_true_label": record.get("source_true_label"),
                "raw_identical": record.get("raw_model_response")
                == old.get("raw_model_response"),
                "label_identical": record.get("predicted_label")
                == old.get("predicted_label"),
                "reference_raw": old.get("raw_model_response"),
                "rerun_raw": record.get("raw_model_response"),
                "rerun_error": record.get("error"),
            }
        )
    n_raw = sum(row["raw_identical"] for row in rows)
    n_label = sum(row["label_identical"] for row in rows)
    return {
        "n": len(rows),
        "raw_identical": n_raw,
        "label_identical": n_label,
        "passed": bool(rows) and n_raw == len(rows),
        "mismatches": [row for row in rows if not row["raw_identical"]],
    }


def _render(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    git = read_git_provenance(project_root)
    reference = read_jsonl(REFERENCE_MANIFEST)
    plots = {json.dumps(record["plot"], sort_keys=True) for record in reference}
    if plots != {json.dumps(EXPECTED_PLOT, sort_keys=True)}:
        raise ValueError(f"Parametros de plot inesperados no manifesto: {plots}")
    if REBUILT_MANIFEST.exists():
        raise FileExistsError(f"{REBUILT_MANIFEST} ja existe; apague para refazer.")

    plotter = DatasetPlotter(
        config=PlotConfig(
            labels_path=LABELS,
            output_dir=IMAGES_DIR,
            manifest_path=REBUILT_MANIFEST,
            cmap=EXPECTED_PLOT["cmap"],
            normalization=EXPECTED_PLOT["normalization"],
            percentile_low=PERCENTILES[0],
            percentile_high=PERCENTILES[1],
            width=EXPECTED_PLOT["width"],
            height=EXPECTED_PLOT["height"],
            dpi=RENDER_DPI,
            include_title=EXPECTED_PLOT["title"],
        ),
        project_root=project_root,
    )
    rebuilt = plotter.plot_all(workers=args.workers)

    with PNG_HASHES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_sample_id", "true_label", "bytes", "sha256"])
        for record in rebuilt:
            path = project_root / record["image_path"]
            writer.writerow(
                [
                    record["sample_id"],
                    record["source_sample_id"],
                    record["true_label"],
                    path.stat().st_size,
                    sha256_file(path),
                ]
            )

    check = compare_manifests(reference, rebuilt)
    write_json(
        OUTPUT_DIR / "render_manifest.json",
        run_manifest(
            experiment="E0-render",
            project_root=project_root,
            git_provenance=git,
            labels={"path": LABELS.as_posix(), "sha256": sha256_file(LABELS)},
            reference_manifest={
                "path": REFERENCE_MANIFEST.as_posix(),
                "sha256": sha256_file(REFERENCE_MANIFEST),
            },
            rebuilt_manifest_sha256=sha256_file(REBUILT_MANIFEST),
            png_hashes_sha256=sha256_file(PNG_HASHES),
            render={**EXPECTED_PLOT, "dpi": RENDER_DPI, "percentiles": PERCENTILES},
            workers=args.workers,
            manifest_check=check,
        ),
    )
    print(json.dumps(check, indent=2))


def _select_check(args: argparse.Namespace) -> None:
    records = select_check_records(read_jsonl(BINARY_MANIFEST), n=args.n, seed=args.seed)
    write_jsonl(CHECK_MANIFEST, records)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["true_label"]] = counts.get(record["true_label"], 0) + 1
    print(f"{len(records)} imagens em {CHECK_MANIFEST}: {counts}")


def _compare(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    result = compare_responses(read_jsonl(REFERENCE_PREDICTIONS), read_jsonl(CHECK_PREDICTIONS))
    write_json(
        CHECK_COMPARISON,
        run_manifest(
            experiment="E0-check50",
            project_root=project_root,
            reference_predictions={
                "path": REFERENCE_PREDICTIONS.as_posix(),
                "sha256": sha256_file(REFERENCE_PREDICTIONS),
            },
            rerun_predictions={
                "path": CHECK_PREDICTIONS.as_posix(),
                "sha256": sha256_file(CHECK_PREDICTIONS),
            },
            **result,
        ),
    )
    print(
        f"raw identico: {result['raw_identical']}/{result['n']}; "
        f"rotulo identico: {result['label_identical']}/{result['n']}; "
        f"{'PASSOU' if result['passed'] else 'FALHOU'}"
    )


def _revisions(args: argparse.Namespace) -> None:
    """Compara pesos (blob) e entradas tokenizadas entre as revisoes HF em cache."""
    from huggingface_hub import scan_cache_dir
    from PIL import Image
    from transformers import AutoProcessor

    from vlm_classifier.prompt import prompt_for_task

    project_root = Path.cwd().resolve()
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    cache = {repo.repo_id: repo for repo in scan_cache_dir().repos}
    report: dict[str, Any] = {}
    for model_id, pinned in MODEL_REVISIONS.items():
        rows = []
        for revision in sorted(cache[model_id].revisions, key=lambda rev: rev.last_modified):
            blobs = {file.file_name: file.blob_path.name for file in revision.files}
            processor = AutoProcessor.from_pretrained(
                model_id,
                revision=revision.commit_hash,
                padding_side="left",
                local_files_only=True,
            )
            inputs = {}
            for task in ("frb-binary", "multiclass"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_for_task(task)},
                        ],
                    }
                ]
                encoded = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="np",
                    add_generation_prompt=True,
                )
                digest = hashlib.sha256()
                for key in sorted(encoded):
                    digest.update(key.encode())
                    digest.update(encoded[key].tobytes())
                inputs[task] = digest.hexdigest()
            rows.append(
                {
                    "revision": revision.commit_hash,
                    "first_cached_utc": revision.last_modified,
                    "pinned_for_v4": revision.commit_hash == pinned,
                    "blobs": blobs,
                    "input_sha256": inputs,
                }
            )
        report[model_id] = rows
    write_json(
        OUTPUT_DIR / "model_revisions.json",
        run_manifest(
            experiment="E0-model-revisions",
            project_root=project_root,
            image={"path": str(args.image), "sha256": sha256_file(Path(args.image))},
            revisions=report,
        ),
    )
    for model_id, rows in report.items():
        weights = {row["blobs"].get("model.safetensors") for row in rows}
        binary = {row["input_sha256"]["frb-binary"] for row in rows}
        print(f"{model_id}: {len(rows)} revisoes, {len(weights)} pesos, {len(binary)} entradas binarias")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="Re-renderizar os 3000 PNGs.")
    render.add_argument("--workers", type=int, default=32)
    render.set_defaults(func=_render)

    select = subparsers.add_parser("select-check", help="Sortear as imagens de checagem.")
    select.add_argument("--n", type=int, default=50)
    select.add_argument("--seed", type=int, default=GENERATION_SEED)
    select.set_defaults(func=_select_check)

    compare = subparsers.add_parser("compare", help="Comparar com a rodada do artigo.")
    compare.set_defaults(func=_compare)

    revisions = subparsers.add_parser("revisions", help="Auditar revisoes HF em cache.")
    revisions.add_argument(
        "--image",
        type=Path,
        default=IMAGES_DIR / "samples" / "sample_000000.png",
    )
    revisions.set_defaults(func=_revisions)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
