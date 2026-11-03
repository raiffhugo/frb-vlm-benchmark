"""Rodadas de GPU dos VLMs: E1, E2, E3, E4, E6, E7 e E8 (plano, secoes 10.3 a 10.10).

Cada rodada chama ``run_pipeline.py classify`` com as flags das rodadas do artigo
(historico do shell de 11-12/07, reproduzidas no E0), a revisao HF fixada em
``common.MODEL_REVISIONS`` e um prompt congelado de ``prompts_v4/``. As rodadas de um modelo
rodam em sequencia na GPU dada. Uma rodada com predictions.jsonl completo e pulada, o que
permite retomar a sequencia; uma incompleta continua de onde parou (``classify --resume``, que
recusa a retomada se a rodada gravada nao for a mesma).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from revision_v4.common import GENERATION_SEED, MODEL_REVISIONS, RESULTS_ROOT


MODELS = {"e2b": "google/gemma-4-E2B-it", "e4b": "google/gemma-4-E4B-it"}

# Flags das rodadas do artigo (results_v4/e0/README.md).
PAPER_FLAGS = (
    "--model", "gemma4",
    "--device-map", "cuda",
    "--dtype", "auto",
    "--max-new-tokens", "1024",
    "--cache-implementation", "static",
    "--generation-seed", str(GENERATION_SEED),
)


@dataclass(frozen=True)
class Run:
    experiment: str
    arm: str
    task: str
    prompt: Path | tuple[tuple[str, Path], ...]  # o mesmo para os dois modelos, ou um por modelo
    manifest: Path
    config: Path

    def prompt_for(self, model: str) -> Path:
        return self.prompt if isinstance(self.prompt, Path) else dict(self.prompt)[model]


_PROMPTS = Path("prompts_v4")
_FULL = _PROMPTS / "binary_full.txt"
_E1_BINARY = Path("dataset_e1_binary/metadata/image_manifest.jsonl")
_E1_MULTICLASS = Path("dataset_e1/metadata/image_manifest.jsonl")
_E3_WINDOWS = Path("dataset_real_e3/metadata/image_manifest.jsonl")
_SELECTIONS = RESULTS_ROOT / "selections"
_E6_PROMPTS = {
    "p0": _FULL,
    "pa": _PROMPTS / "e6_pa_paraphrase.txt",
    "pb": _PROMPTS / "e6_pb_non_frb_first.txt",
    "pc": _PROMPTS / "e6_pc_short.txt",
    "pd": _PROMPTS / "e6_pd_generic_non_frb.txt",
    "pe": _PROMPTS / "e6_pe_no_low_snr.txt",
    "r": _PROMPTS / "e6_r_relaxed_morphology.txt",
}
_CONFIG = Path("config.yaml")
# E7: V1 e V2 sao o lean e o lean+reason da Tabela 10; o filler do V4 tem o tamanho medio da
# justificativa de cada modelo (revision_v4/e7_prompts.py)
_E7_PROMPTS = {
    "v1_lean": _PROMPTS / "binary_lean.txt",
    "v1o_no_reason": _PROMPTS / "e7_v1o_no_reason.txt",
    "v2_reason_before": _PROMPTS / "binary_lean_reason.txt",
    "v3_reason_after": _PROMPTS / "e7_v3_reason_after.txt",
    "v4_filler": tuple((model, _PROMPTS / f"e7_v4_filler_{model}.txt") for model in MODELS),
    "v5_reason_reworded": _PROMPTS / "e7_v5_reason_reworded.txt",
}

RUNS: dict[str, tuple[Run, ...]] = {
    "e1": tuple(
        Run("e1", arm, "frb-binary", _PROMPTS / f"{arm}.txt", _E1_BINARY, Path("config_e1_binary.yaml"))
        for arm in ("binary_full", "binary_lean", "binary_lean_reason")
    )
    + (
        Run("e1", "multiclass", "multiclass", _PROMPTS / "multiclass.txt", _E1_MULTICLASS, Path("config_e1.yaml")),
    ),
    "e3": (Run("e3", "binary_full", "frb-binary", _FULL, _E3_WINDOWS, _CONFIG),),
    "e2": (
        Run("e2", "binary_full", "frb-binary", _FULL, Path("dataset_real_e2/metadata/image_manifest.jsonl"), _CONFIG),
    ),
    "e4": tuple(
        Run("e4", f"{variant}_binary_full", "frb-binary", _FULL,
            Path(f"dataset_real_e4_{variant}/metadata/image_manifest.jsonl"), _CONFIG)
        for variant in ("p1", "p2")
    ),
    "e6": tuple(
        Run("e6", f"{subset}_{name}", "frb-binary", prompt, _SELECTIONS / f"e6_{subset}.jsonl", _CONFIG)
        for subset in ("s200", "r200")
        for name, prompt in _E6_PROMPTS.items()
    )
    + (
        Run("e6", "real1600_r", "frb-binary", _E6_PROMPTS["r"],
            Path("dataset_real_full/metadata/image_manifest.jsonl"), _CONFIG),
        Run("e6", "sim2000_r", "frb-binary", _E6_PROMPTS["r"],
            Path("dataset_binary/metadata/image_manifest.jsonl"), _CONFIG),
    ),
    "e7": tuple(
        Run("e7", arm, "frb-binary", prompt, _SELECTIONS / "e7_heldout400.jsonl", Path("config_e1_binary.yaml"))
        for arm, prompt in _E7_PROMPTS.items()
    ),
    "e8": (
        Run("e8", "binary_full", "frb-binary", _FULL, Path("dataset_e8_decimated/metadata/image_manifest.jsonl"), _CONFIG),
    ),
}


def output_path(run: Run, model: str) -> Path:
    return RESULTS_ROOT / run.experiment / f"{model}_{run.arm}" / "predictions.jsonl"


def count_records(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def is_complete(run: Run, model: str) -> bool:
    output = output_path(run, model)
    return output.exists() and count_records(output) == count_records(run.manifest)


def classify_command(run: Run, model: str, *, resume: bool = False) -> list[str]:
    model_id = MODELS[model]
    return [
        sys.executable, "run_pipeline.py", "classify",
        *PAPER_FLAGS,
        "--model-id", model_id,
        "--model-revision", MODEL_REVISIONS[model_id],
        "--task", run.task,
        "--prompt-file", run.prompt_for(model).as_posix(),
        "--config", run.config.as_posix(),
        "--manifest", run.manifest.as_posix(),
        "--output", output_path(run, model).as_posix(),
        "--resume" if resume else "--overwrite",
    ]


def select_runs(experiment: str, arms: list[str] | None) -> tuple[Run, ...]:
    runs = RUNS[experiment]
    if not arms:
        return runs
    by_arm = {run.arm: run for run in runs}
    return tuple(by_arm[arm] for arm in arms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiment", choices=sorted(RUNS))
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--gpu", required=True, help="Valor de CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--arms", nargs="+", help="Rodadas do experimento, na ordem dada.")
    parser.add_argument("--dry-run", action="store_true", help="So imprimir os comandos.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "HF_HUB_OFFLINE": "1"}
    for run in select_runs(args.experiment, args.arms):
        output = output_path(run, args.model)
        command = classify_command(run, args.model, resume=output.exists())
        if args.dry_run:
            print(" ".join(command))
            continue
        if is_complete(run, args.model):
            print(f"{output}: completo, pulando", flush=True)
            continue
        print(f"{output}: {'retomando' if output.exists() else 'comecando'}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent / "classify.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{output}: {count_records(output)}/{count_records(run.manifest)}", flush=True)


if __name__ == "__main__":
    main()
