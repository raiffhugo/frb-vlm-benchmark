"""E1: conjunto simulado held-out (plano, secao 10.3). Checagens e proveniencia.

A geracao usa os comandos normais da pipeline (results_v4/e1/README.md):
``simulate`` com config_e1.yaml (seed 20260927, 1000 por classe), ``plot`` com os
parametros do benchmark e ``select-binary-benchmark`` com seed 20260927. Este script
confere o resultado (contagens, sementes disjuntas das do benchmark, determinismo) e
grava o manifesto de proveniencia, os hashes dos PNGs e copias versionadas dos
manifestos das 3000 amostras.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from revision_v4.common import (
    RESULTS_ROOT,
    SIM_SEED,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
)


OUTPUT_DIR = RESULTS_ROOT / "e1"
CONFIG = Path("config_e1.yaml")
LABELS = Path("dataset_e1/metadata/labels.jsonl")
IMAGE_MANIFEST = Path("dataset_e1/metadata/image_manifest.jsonl")
BINARY_MANIFEST = Path("dataset_e1_binary/metadata/image_manifest.jsonl")
BENCHMARK_LABELS = Path("dataset/metadata/labels.jsonl")


def replay_sample_seeds(master_seed: int, count: int) -> list[int]:
    """As sementes por amostra que o simulador sorteia, na ordem frb, rfi, noise."""
    rng = random.Random(master_seed)
    return [rng.randrange(1, 2**31 - 1) for _ in range(count)]


def seed_checks(e1_labels: list[dict[str, Any]], benchmark_labels: list[dict[str, Any]]) -> dict[str, Any]:
    e1_seeds = [int(record["seed"]) for record in e1_labels]
    benchmark_seeds = {int(record["seed"]) for record in benchmark_labels}
    return {
        "n": len(e1_seeds),
        "internal_duplicates": len(e1_seeds) - len(set(e1_seeds)),
        "collisions_with_benchmark": len(set(e1_seeds) & benchmark_seeds),
        "replays_master_seed": e1_seeds == replay_sample_seeds(SIM_SEED, len(e1_seeds)),
    }


def composition(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(record["true_label"]) for record in records)
    subtypes = Counter(
        "+".join(record["simulation_parameters"].get("rfi_types", []))
        for record in records
        if record["true_label"] == "RFI"
    )
    return {"by_class": dict(sorted(labels.items())), "rfi_subtypes": dict(sorted(subtypes.items()))}


def _check(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    e1_labels = read_jsonl(LABELS)
    images = read_jsonl(IMAGE_MANIFEST)
    binary = read_jsonl(BINARY_MANIFEST)
    seeds = seed_checks(e1_labels, read_jsonl(BENCHMARK_LABELS))

    with (OUTPUT_DIR / "png_sha256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "source_sample_id", "true_label", "bytes", "sha256"])
        for record in images:
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
    for source in (LABELS, IMAGE_MANIFEST):
        shutil.copyfile(source, OUTPUT_DIR / source.name)

    checks = {
        "seeds": seeds,
        "multiclass_composition": composition(images),
        "binary_composition": composition(binary),
        "passed": (
            seeds["internal_duplicates"] == 0
            and seeds["collisions_with_benchmark"] == 0
            and seeds["replays_master_seed"]
            and len(images) == 3000
            and len(binary) == 2000
        ),
    }
    write_json(
        OUTPUT_DIR / "e1_manifest.json",
        run_manifest(
            experiment="E1-heldout-set",
            project_root=project_root,
            config={"path": CONFIG.as_posix(), "sha256": sha256_file(CONFIG), "text": CONFIG.read_text()},
            files={
                path.as_posix(): sha256_file(path)
                for path in (LABELS, IMAGE_MANIFEST, BINARY_MANIFEST, OUTPUT_DIR / "png_sha256.csv")
            },
            **checks,
        ),
    )
    print(
        f"sementes: {seeds}; multiclasse {checks['multiclass_composition']['by_class']}; "
        f"binario {checks['binary_composition']['by_class']}; "
        f"{'PASSOU' if checks['passed'] else 'FALHOU'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Conferir o conjunto e gravar a proveniencia.")
    check.set_defaults(func=_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
