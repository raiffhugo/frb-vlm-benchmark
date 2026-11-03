from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from vlm_classifier.classifier import read_git_provenance


# Protocolo congelado (plano, secao 10.1).
SIM_SEED = 20260927
REAL_SEED = 20261013
GENERATION_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERATIONS = 10_000
SIGMA_BINS = (0.0, 5.0, 8.0, 15.0, 30.0, 60.0, 100.0, 300.0, float("inf"))

# Revisoes HF fixadas para as rodadas novas. Os pesos sao os mesmos de todas as
# revisoes usadas no artigo (ver results_v4/e0/model_revisions.json).
MODEL_REVISIONS = {
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
}

RESULTS_ROOT = Path("results_v4")

# Lido quando o modulo e importado, isto e, no inicio do comando: as etapas que reescrevem um
# arquivo versionado (um plano refeito, por exemplo) nao podem se marcar como arvore suja.
GIT_AT_START = read_git_provenance(Path.cwd())

_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "astropy",
    "matplotlib",
    "pillow",
    "torch",
    "transformers",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def software_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run_manifest(
    *,
    experiment: str,
    project_root: Path,
    git_provenance: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Proveniencia de uma etapa: commit, comando, versoes e campos do protocolo."""
    return {
        "experiment": experiment,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "git": git_provenance or GIT_AT_START,
        "software_versions": software_versions(),
        **fields,
    }
