"""Filas de GPU da revisao: o que cada GPU roda, em ordem (montadas em 19/10/2026).

Depois de uma queda do servidor, basta rodar, uma vez por GPU e em segundo plano,

    uv run python -m revision_v4.queues run --gpu N

Cada etapa e idempotente: uma rodada de VLM completa e pulada e uma interrompida continua de
onde parou (``classify --resume``); SwinYNet, importacao, renders e CNN so rodam se o resultado
ainda nao existe (o SwinYNet e o treino da CNN recomecam do zero, porque nao retomam). ``show``
lista as filas e o estado de cada etapa.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from revision_v4.common import RESULTS_ROOT
from revision_v4.gpu_runs import count_records, is_complete, select_runs


SWINYNET_DIR = Path.home() / "SwinYNet"
SWINYNET_WORKDIR = Path("/var/tmp/frb_svlm_e1_swinynet")
E5_WORKDIR = Path("/var/tmp/frb_svlm_e5_cnn")
# as chaves que o classify exige em cada registro do manifesto
REQUIRED_MANIFEST_KEYS = {"sample_id", "image_path", "true_label", "simulation_parameters"}
R200_ARMS = tuple(f"r200_{name}" for name in ("p0", "pa", "pb", "pc", "pd", "pe", "r")) + ("real1600_r",)
S200_ARMS = tuple(f"s200_{name}" for name in ("p0", "pa", "pb", "pc", "pd", "pe", "r")) + ("sim2000_r",)


@dataclass(frozen=True)
class Step:
    kind: str  # vlm, swinynet, import-swinynet, cnn-render, cnn-train, cnn-score
    experiment: str = ""
    model: str = ""
    arms: tuple[str, ...] = ()
    fold: int = 0

    def label(self) -> str:
        if self.kind == "vlm":
            return f"{self.experiment} {self.model}" + (f" ({len(self.arms)} bracos)" if self.arms else "")
        return f"{self.kind}" + (f" fold {self.fold}" if self.kind.startswith("cnn") else "")


QUEUES: dict[int, tuple[Step, ...]] = {
    0: (
        Step("vlm", "e1", "e4b"),
        Step("vlm", "e2", "e4b"),
        Step("vlm", "e4", "e4b"),
        Step("vlm", "e8", "e4b"),
    ),
    1: (
        Step("vlm", "e3", "e4b"),
        Step("vlm", "e6", "e4b", R200_ARMS),
    ),
    # o E2 do E2B foi para o fim em 19/10 (a primeira tentativa falhou no manifesto) e o E7
    # entrou antes dele
    2: (
        Step("vlm", "e1", "e2b"),
        Step("vlm", "e4", "e2b"),
        Step("vlm", "e8", "e2b"),
        Step("vlm", "e6", "e2b"),
        Step("vlm", "e7", "e2b"),
        Step("vlm", "e7", "e4b"),
        Step("vlm", "e2", "e2b"),
    ),
    3: (
        Step("swinynet"),
        Step("import-swinynet"),
        Step("vlm", "e3", "e2b"),
        Step("cnn-render", fold=0),
        Step("cnn-render", fold=1),
        Step("cnn-train", fold=0),
        Step("cnn-score", fold=0),
        Step("cnn-train", fold=1),
        Step("cnn-score", fold=1),
        Step("vlm", "e6", "e4b", S200_ARMS),
    ),
}


def is_done(step: Step) -> bool:
    if step.kind == "vlm":
        runs = select_runs(step.experiment, list(step.arms) or None)
        return all(run.manifest.exists() and is_complete(run, step.model) for run in runs)
    if step.kind == "swinynet":
        return (SWINYNET_WORKDIR / "output" / "manifest.json").exists()
    if step.kind == "import-swinynet":
        path = RESULTS_ROOT / "e1" / "swinynet" / "predictions.jsonl"
        return path.exists() and count_records(path) == 2000
    if step.kind == "cnn-render":
        return (E5_WORKDIR / f"fold{step.fold}" / "metadata" / "image_manifest.jsonl").exists()
    if step.kind == "cnn-train":
        return (RESULTS_ROOT / "e5" / f"cnn_train_fold{step.fold}.json").exists()
    if step.kind == "cnn-score":
        return (RESULTS_ROOT / "e5" / f"cnn_scores_fold{step.fold}.jsonl").exists()
    raise ValueError(f"Etapa desconhecida: {step.kind}")


def command(step: Step, gpu: int) -> tuple[list[str], Path | None, Path | None]:
    """(comando, diretorio de trabalho, log); None = repositorio e saida no terminal."""
    python = sys.executable
    if step.kind == "vlm":
        arms = ["--arms", *step.arms] if step.arms else []
        return [python, "-m", "revision_v4.gpu_runs", step.experiment, "--model", step.model,
                "--gpu", str(gpu), *arms], None, None
    if step.kind == "swinynet":
        return (
            ["uv", "run", "--no-sync", "swinynet", "-c", "frbd/cfgs/te8hjj4j.yaml",
             "-p", str(SWINYNET_WORKDIR / "fits"), "-o", str(SWINYNET_WORKDIR / "output" / "manifest.json"),
             "-x", "run.num_postprocessor=0", "-I", "none"],
            SWINYNET_DIR,
            SWINYNET_WORKDIR / "swinynet.log",
        )
    if step.kind == "import-swinynet":
        return ([python, "-m", "revision_v4.e1_analysis", "import-swinynet", "--workdir", str(SWINYNET_WORKDIR),
                 "--external", str(SWINYNET_WORKDIR / "output" / "manifest.json")],
                None, RESULTS_ROOT / "e1" / "swinynet" / "import.log")
    if step.kind == "cnn-render":
        return ([python, "-m", "revision_v4.e5_cnn", "render", "--fold", str(step.fold), "--workers", "40"],
                None, RESULTS_ROOT / "e5" / f"cnn_render_fold{step.fold}.log")
    name = step.kind.removeprefix("cnn-")
    return ([python, "-m", "revision_v4.e5_cnn", name, "--fold", str(step.fold)],
            None, RESULTS_ROOT / "e5" / f"cnn_{name}_fold{step.fold}.log")


def _run(args: argparse.Namespace) -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1"}
    for step in QUEUES[args.gpu]:
        if is_done(step):
            print(f"GPU {args.gpu}: {step.label()} ja feito", flush=True)
            continue
        argv, cwd, log = command(step, args.gpu)
        print(f"GPU {args.gpu}: {step.label()} comecando", flush=True)
        if log is None:
            result = subprocess.run(argv, cwd=cwd, env=env)
        else:
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(argv, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"GPU {args.gpu}: {step.label()} FALHOU (codigo {result.returncode})", flush=True)


def manifest_problems(manifest: Path) -> list[str]:
    """O que impediria o classify de rodar: arquivo ausente, chave obrigatoria ou imagem faltando."""
    if not manifest.exists():
        return [f"{manifest}: nao existe"]
    problems = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = sorted(REQUIRED_MANIFEST_KEYS - record.keys())
        if missing:
            problems.append(f"{manifest}:{number}: sem {', '.join(missing)}")
        elif not Path(record["image_path"]).exists():
            problems.append(f"{manifest}:{number}: imagem ausente {record['image_path']}")
        if len(problems) >= 3:
            break
    return problems


def _check(_args: argparse.Namespace) -> None:
    seen: set[Path] = set()
    failures = 0
    for steps in QUEUES.values():
        for step in steps:
            if step.kind != "vlm":
                continue
            for run in select_runs(step.experiment, list(step.arms) or None):
                if run.manifest in seen:
                    continue
                seen.add(run.manifest)
                problems = manifest_problems(run.manifest)
                failures += bool(problems)
                print(f"{'FALHA' if problems else 'ok   '} {run.manifest}")
                for problem in problems:
                    print(f"      {problem}")
    raise SystemExit(1 if failures else 0)


def _show(_args: argparse.Namespace) -> None:
    for gpu, steps in QUEUES.items():
        print(f"GPU {gpu}:")
        for step in steps:
            print(f"  [{'x' if is_done(step) else ' '}] {step.label()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Rodar (ou retomar) a fila de uma GPU.")
    run.add_argument("--gpu", type=int, choices=sorted(QUEUES), required=True)
    run.set_defaults(func=_run)
    subparsers.add_parser("show", help="Mostrar as filas e o que ja foi feito.").set_defaults(func=_show)
    subparsers.add_parser("check", help="Conferir os manifestos de todas as rodadas.").set_defaults(func=_check)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
