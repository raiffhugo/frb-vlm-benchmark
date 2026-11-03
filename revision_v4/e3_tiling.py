"""E3: tiling com agregacao por arquivo (plano, secao 10.5; definicoes na 10.12, 19/10).

  1. ``check-p0``: o P0 refeito nas 1600 janelas do artigo contra ``dataset_real_full``
     (PNG byte a byte e os campos que o manifesto do artigo grava).
  2. ``analyze``: metricas por arquivo e por janela das rodadas do E2B e do E4B, com o
     casamento temporal pelo sweep do catalogo e a comparacao pareada com a janela centrada
     do artigo; tambem sem os dois arquivos com DAT_SCL corrompido (secao 10.12).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.window_aggregator import window_probability
from prepare_real.preprocess import dispersion_delay_s
from prepare_real.reader import read_layout
from revision_v4.common import (
    RESULTS_ROOT,
    SIGMA_BINS,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
)
from revision_v4.stats import PairedCounts, mcnemar_exact, newcombe_paired_difference, wilson


OUTPUT_DIR = RESULTS_ROOT / "e3"
PAPER_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")
E3_MANIFEST = Path("dataset_real_e3/metadata/image_manifest.jsonl")
VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
PAPER_PREDICTIONS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
VLMS = ("e2b", "e4b")
THRESHOLD = 0.5
# Arquivos com sub-integracoes de DAT_SCL corrompido (plano, secao 10.12): o E3 mantem as janelas
# deles, como uma busca cega veria, e da tambem o resultado sem eles.
CORRUPTED_FILES = ("00324_neg_sample", "00853_neg_sample")
# Campos que o manifesto do artigo grava; o codigo atual acrescenta os das janelas.
PAPER_RECORD_FIELDS = ("fits_path", "source_sample_id", "true_label", "real_parameters")


# ---------------------------------------------------------------------------
# check-p0


def compare_p0(
    paper: list[dict[str, Any]],
    rebuilt: list[dict[str, Any]],
    *,
    project_root: Path,
) -> dict[str, Any]:
    if len(paper) != len(rebuilt):
        raise ValueError(f"{len(paper)} registros no artigo e {len(rebuilt)} refeitos.")
    mismatches = []
    identical_png = 0
    for old, new in zip(paper, rebuilt):
        problems = [field for field in PAPER_RECORD_FIELDS if old.get(field) != new.get(field)]
        problems += [
            f"preprocessing.{key}"
            for key, value in old["preprocessing"].items()
            if new["preprocessing"].get(key) != value
        ]
        if sha256_file(project_root / old["image_path"]) == sha256_file(
            project_root / new["image_path"]
        ):
            identical_png += 1
        else:
            problems.append("png")
        if problems:
            mismatches.append({"sample_id": old["sample_id"], "problems": problems})
    return {
        "n": len(paper),
        "identical_png": identical_png,
        "records_with_differences": len(mismatches),
        "mismatches": mismatches[:50],
        "passed": not mismatches,
    }


def _check_p0(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    rebuilt_manifest = args.workdir / "metadata" / "image_manifest.jsonl"
    result = compare_p0(
        read_jsonl(PAPER_MANIFEST), read_jsonl(rebuilt_manifest), project_root=project_root
    )
    write_json(
        OUTPUT_DIR / "p0_repro.json",
        run_manifest(
            experiment="E3-P0-reproduction",
            project_root=project_root,
            paper_manifest={"path": PAPER_MANIFEST.as_posix(), "sha256": sha256_file(PAPER_MANIFEST)},
            rebuilt_manifest={"path": str(rebuilt_manifest), "sha256": sha256_file(rebuilt_manifest)},
            **result,
        ),
    )
    print(
        f"{result['identical_png']}/{result['n']} PNGs identicos; "
        f"{result['records_with_differences']} registros com diferenca; "
        f"{'PASSOU' if result['passed'] else 'FALHOU'}"
    )


# ---------------------------------------------------------------------------
# analyze


@dataclass(frozen=True)
class Sweep:
    """Intervalo de chegada do burst [inicio, fim], em segundos dentro do arquivo."""

    start: float
    stop: float


def catalog_sweeps(
    *, toa_s: float, dm: float, frequencies_mhz: np.ndarray, edge_channels: int
) -> dict[str, Sweep]:
    """Sweep na banda inteira (primario) e so nos canais visiveis (sensibilidade).

    O ToA do catalogo e a chegada no canal mais alto, como na centragem do artigo.
    """
    f_low, f_high = float(frequencies_mhz[0]), float(frequencies_mhz[-1])
    visible_low = float(frequencies_mhz[edge_channels])
    visible_high = float(frequencies_mhz[len(frequencies_mhz) - 1 - edge_channels])
    return {
        "full_band": Sweep(toa_s, toa_s + dispersion_delay_s(dm, f_low, f_high)),
        "visible_band": Sweep(
            toa_s + dispersion_delay_s(dm, visible_high, f_high),
            toa_s + dispersion_delay_s(dm, visible_low, f_high),
        ),
    }


def window_contains(record: dict[str, Any], sweep: Sweep) -> bool:
    return (
        float(record["window_start_seconds"]) <= sweep.start
        and sweep.stop <= float(record["window_stop_seconds"])
    )


@dataclass(frozen=True)
class FileWindows:
    """As janelas de um arquivo, com o score de cada uma (None = invalida)."""

    file_group_id: str
    is_frb: bool
    records: tuple[dict[str, Any], ...]
    scores: tuple[float | None, ...]

    @property
    def valid_scores(self) -> list[float]:
        return [score for score in self.scores if score is not None]

    def detected(self) -> bool | None:
        valid = self.valid_scores
        return max(valid) >= THRESHOLD if valid else None

    def detected_in(self, sweep: Sweep) -> bool:
        return any(
            score is not None and score >= THRESHOLD
            for record, score in zip(self.records, self.scores)
            if window_contains(record, sweep)
        )


def group_windows(
    manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, FileWindows]:
    by_sample = {record["sample_id"]: record for record in predictions}
    if set(by_sample) != {record["sample_id"] for record in manifest}:
        raise ValueError("Predicoes e manifesto do E3 nao cobrem as mesmas janelas.")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        groups[record["source_sample_id"]].append(record)
    files = {}
    for source_id, records in groups.items():
        ordered = tuple(sorted(records, key=lambda record: int(record["window_index"])))
        files[source_id] = FileWindows(
            file_group_id=source_id,
            is_frb=ordered[0]["true_label"] == "FRB",
            records=ordered,
            scores=tuple(window_probability(by_sample[r["sample_id"]])[0] for r in ordered),
        )
    return files


def _rate(hits: list[bool]) -> dict[str, Any]:
    return {"k": int(sum(hits)), "n": len(hits), **wilson(int(sum(hits)), len(hits)).as_dict()}


def _paired(first: list[bool], second: list[bool]) -> dict[str, Any]:
    counts = PairedCounts.from_outcomes(first, second)
    return {
        "difference_newcombe": newcombe_paired_difference(counts).as_dict(),
        "mcnemar": mcnemar_exact(counts),
    }


def tiling_metrics(
    files: dict[str, FileWindows],
    sweeps: dict[str, dict[str, Sweep]],
    centered: dict[str, bool],
) -> dict[str, Any]:
    """Metricas do protocolo para um modelo. ``centered``: acerto do artigo por arquivo."""
    excluded = sorted(fid for fid, f in files.items() if f.detected() is None)
    positives = [f for fid, f in sorted(files.items()) if f.is_frb and fid not in excluded]
    negatives = [f for fid, f in sorted(files.items()) if not f.is_frb and fid not in excluded]
    windows_neg = [score for f in negatives for score in f.valid_scores]
    result: dict[str, Any] = {
        "files_without_valid_window": excluded,
        "invalid_windows": int(sum(score is None for f in files.values() for score in f.scores)),
        "recall_any_window": _rate([bool(f.detected()) for f in positives]),
        "fpr_per_file": _rate([bool(f.detected()) for f in negatives]),
        # janelas do mesmo arquivo nao sao independentes: o intervalo e so indicativo
        "fpr_per_window": _rate([score >= THRESHOLD for score in windows_neg]),
    }
    for band in ("full_band", "visible_band"):
        has_window = [
            any(window_contains(r, sweeps[f.file_group_id][band]) for r in f.records)
            for f in positives
        ]
        matched = [f.detected_in(sweeps[f.file_group_id][band]) for f in positives]
        result[band] = {
            "fraction_with_full_sweep_window": _rate(has_window),
            "recall_time_matched": _rate(matched),
            "recall_time_matched_given_window": _rate(
                [hit for hit, ok in zip(matched, has_window) if ok]
            ),
        }
    matched = [f.detected_in(sweeps[f.file_group_id]["full_band"]) for f in positives]
    paper_hits = [centered[f.file_group_id] for f in positives]
    result["paired_vs_centered"] = {
        "recall_centered": _rate(paper_hits),
        "time_matched_minus_centered": _paired(matched, paper_hits),
        "any_window_minus_centered": _paired(
            [bool(f.detected()) for f in positives], paper_hits
        ),
        "fpr_file_minus_centered": _paired(
            [bool(f.detected()) for f in negatives],
            [not centered[f.file_group_id] for f in negatives],
        ),
    }
    return result


def stratified(
    files: dict[str, FileWindows],
    sweeps: dict[str, dict[str, Sweep]],
    centered: dict[str, bool],
    key_of: dict[str, str],
) -> dict[str, Any]:
    groups: dict[str, list[FileWindows]] = defaultdict(list)
    for fid, f in sorted(files.items()):
        if f.is_frb and f.detected() is not None:
            groups[key_of[fid]].append(f)
    return {
        key: {
            "recall_any_window": _rate([bool(f.detected()) for f in members]),
            "recall_time_matched": _rate(
                [f.detected_in(sweeps[f.file_group_id]["full_band"]) for f in members]
            ),
            "recall_centered": _rate([centered[f.file_group_id] for f in members]),
        }
        for key, members in sorted(groups.items())
    }


def sigma_bin_of(sigma: float) -> str:
    for low, high in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
        if low <= sigma < high:
            return f"[{low:g}, {high:g})" if np.isfinite(high) else f">= {low:g}"
    raise ValueError(f"sigma fora dos bins: {sigma}")


def centered_hits(paper_manifest: list[dict[str, Any]], path: Path) -> dict[str, bool]:
    """Acerto da rodada do artigo por arquivo (FRB detectado; negativo rejeitado)."""
    source_of = {r["sample_id"]: r["source_sample_id"] for r in paper_manifest}
    hits = {}
    for record in read_jsonl(path):
        probability, reason = window_probability(record)
        if reason is not None:
            raise ValueError(f"{record['sample_id']}: {reason}")
        predicted_frb = probability >= THRESHOLD
        hits[source_of[record["sample_id"]]] = predicted_frb == (record["true_label"] == "FRB")
    return hits


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    manifest = read_jsonl(E3_MANIFEST)
    paper_manifest = read_jsonl(PAPER_MANIFEST)

    sweeps: dict[str, dict[str, Sweep]] = {}
    for record in manifest:
        source_id = record["source_sample_id"]
        if record["true_label"] != "FRB" or source_id in sweeps:
            continue
        layout = read_layout(project_root / record["fits_path"])
        parameters = record["real_parameters"]
        sweeps[source_id] = catalog_sweeps(
            toa_s=float(parameters["toa_s"]),
            dm=float(parameters["dm"]),
            frequencies_mhz=np.asarray(layout.frequencies_mhz),
            edge_channels=int(record["preprocessing"]["edge_channels"]),
        )

    with VISIBILITY.open(encoding="utf-8") as handle:
        sigma_of = {row["source_sample_id"]: float(row["peak_sigma"]) for row in csv.DictReader(handle)}
    source_of = {r["source_sample_id"]: r["real_parameters"]["source"] for r in manifest}
    sigma_bin = {fid: sigma_bin_of(sigma) for fid, sigma in sigma_of.items()}

    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "manifest": {"path": E3_MANIFEST.as_posix(), "sha256": sha256_file(E3_MANIFEST)},
    }
    for vlm in args.models:
        path = OUTPUT_DIR / f"{vlm}_binary_full" / "predictions.jsonl"
        files = group_windows(manifest, read_jsonl(path))
        if not set(CORRUPTED_FILES) <= set(files):
            raise ValueError("Os arquivos corrompidos deveriam estar no E3.")
        centered = centered_hits(paper_manifest, PAPER_PREDICTIONS[vlm])
        results[vlm] = {
            **tiling_metrics(files, sweeps, centered),
            "without_corrupted_files": tiling_metrics(
                {fid: f for fid, f in files.items() if fid not in CORRUPTED_FILES}, sweeps, centered
            ),
            "by_source": stratified(files, sweeps, centered, source_of),
            "by_sigma_bin": stratified(files, sweeps, centered, sigma_bin),
        }
        inputs[vlm] = {"path": path.as_posix(), "sha256": sha256_file(path)}

    write_json(
        OUTPUT_DIR / ("e3_results.json" if len(args.models) == len(VLMS) else f"e3_results_{'_'.join(args.models)}.json"),
        run_manifest(
            experiment="E3-tiling",
            project_root=project_root,
            threshold=THRESHOLD,
            corrupted_files=list(CORRUPTED_FILES),
            inputs=inputs,
            results=results,
        ),
    )
    for vlm in args.models:
        r = results[vlm]
        print(
            f"{vlm}: recall qualquer janela {r['recall_any_window']['estimate']:.3f}, "
            f"casado {r['full_band']['recall_time_matched']['estimate']:.3f}, "
            f"centrado {r['paired_vs_centered']['recall_centered']['estimate']:.3f}; "
            f"FPR por arquivo {r['fpr_per_file']['estimate']:.3f}, "
            f"por janela {r['fpr_per_window']['estimate']:.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check-p0", help="Comparar o P0 refeito com o do artigo.")
    check.add_argument("--workdir", type=Path, required=True)
    check.set_defaults(func=_check_p0)
    analyze = subparsers.add_parser("analyze", help="Metricas do E3.")
    analyze.add_argument("--models", nargs="+", choices=VLMS, default=list(VLMS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
