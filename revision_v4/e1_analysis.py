"""E1: rodadas no conjunto simulado held-out (plano, secoes 10.2 e 10.3).

  1. ``import-swinynet``: normaliza a saida do SwinYNet como no artigo (limiar 0,5, maximo
     entre candidatos) e troca os ids do export pelos do binario do E1.
  2. ``analyze``: le as rodadas de results_v4/e1/ e as do benchmark do artigo e grava
     e1_results.json com
       - o binario a 0,5 e a diferenca pareada VLM - SwinYNet (C2);
       - benchmark x E1, com a diferenca entre amostras independentes;
       - o item 1 do R0 com os limiares escolhidos num conjunto e aplicados no outro (C1);
       - os hibridos AND/OR nesses limiares (C1);
       - as variantes de prompt da Tabela 10 nas 2000 imagens (C9);
       - o multiclasse nas 3000 (C7);
       - o recall por bin de sigma, benchmark x E1 (C3/C10).

Uma resposta sem frb_probability valida (ou sem rotulo, no multiclasse) conta como resposta
errada em todas as metricas (plano, secao 10.12, 19/10).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_predictions.importer import (
    ExternalPredictionImportConfig,
    ExternalPredictionImporter,
)
from evaluation.window_aggregator import window_probability
from revision_v4.common import (
    RESULTS_ROOT,
    SIGMA_BINS,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)
from revision_v4.r0_reanalysis import (
    multiclass_rates,
    paired_comparison,
    rates,
    threshold_matching_fpr,
    threshold_matching_recall,
    threshold_max_accuracy,
)
from revision_v4.stats import Interval, newcombe_unpaired_difference, wilson


E1_DIR = RESULTS_ROOT / "e1"
VLMS = ("e2b", "e4b")
BINARY_ARMS = ("binary_full", "binary_lean", "binary_lean_reason")
# Abaixo disso o burst e "fraco" no recorte descritivo do C9 (bins [0,5), [5,8) e [8,15)).
FAINT_SIGMA = 15.0

SETS: dict[str, dict[str, Path]] = {
    "benchmark": {
        "binary_manifest": Path("dataset_binary/metadata/image_manifest.jsonl"),
        "visibility": Path("dataset/metadata/burst_visibility_sim.csv"),
        "e2b_binary_full": Path("results_vlm2b_0t_binary/predictions.jsonl"),
        "e4b_binary_full": Path("results_vlm_0t_binary/predictions.jsonl"),
        "swinynet": Path("results_external_binary/predictions.jsonl"),
        # results_vlm2b_0t_new and results_vlm_0t_new in the working repository: renamed results_vlm2b_0t
        # and results_vlm_0t in this release (same SHA-256); the manifests in results_v4 keep the old names.
        "e2b_multiclass": Path("results_vlm2b_0t/predictions.jsonl"),
        "e4b_multiclass": Path("results_vlm_0t/predictions.jsonl"),
    },
    "e1": {
        "binary_manifest": Path("dataset_e1_binary/metadata/image_manifest.jsonl"),
        "visibility": E1_DIR / "burst_visibility_e1.csv",
        "swinynet": E1_DIR / "swinynet" / "predictions.jsonl",
        **{
            f"{vlm}_{arm}": E1_DIR / f"{vlm}_{arm}" / "predictions.jsonl"
            for vlm in VLMS
            for arm in (*BINARY_ARMS, "multiclass")
        },
    },
}


# ---------------------------------------------------------------------------
# Leitura


@dataclass(frozen=True)
class Run:
    sample_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    y_true: np.ndarray
    scores: np.ndarray  # frb_probability; NaN quando a resposta falhou
    source_labels: tuple[str, ...]

    @property
    def failed(self) -> np.ndarray:
        return np.isnan(self.scores)

    def predict(self, threshold: float) -> np.ndarray:
        """Decisao FRB no limiar; uma resposta que falhou vira a decisao errada."""
        predicted = np.where(self.failed, -np.inf, self.scores) >= threshold
        return np.where(self.failed, ~self.y_true, predicted)


def load_run(path: Path, source_of: dict[str, str]) -> Run:
    records = sorted(read_jsonl(path), key=lambda record: record["sample_id"])
    scores = []
    for record in records:
        probability, reason = window_probability(record)
        scores.append(np.nan if reason is not None else probability)
    return Run(
        sample_ids=tuple(r["sample_id"] for r in records),
        source_ids=tuple(source_of[r["sample_id"]] for r in records),
        y_true=np.array([r["true_label"] == "FRB" for r in records]),
        scores=np.array(scores, dtype=float),
        source_labels=tuple(str(r["source_true_label"]).upper() for r in records),
    )


def check_aligned(runs: dict[str, Run]) -> None:
    reference = next(iter(runs.values()))
    for name, run in runs.items():
        if run.sample_ids != reference.sample_ids or not np.array_equal(run.y_true, reference.y_true):
            raise ValueError(f"{name}: amostras ou rotulos diferentes das outras rodadas.")


def read_sigma(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        return {
            row["source_sample_id"]: float(row["peak_sigma"])
            for row in csv.DictReader(handle)
            if row["true_label"] == "FRB"
        }


# ---------------------------------------------------------------------------
# Tabelas


def binary_summary(run: Run, subtype_of: dict[str, str]) -> dict[str, Any]:
    predicted = run.predict(0.5)
    by_source = {}
    for source in ("RFI", "NOISE"):
        mask = np.array([label == source for label in run.source_labels])
        by_source[source] = wilson(int(np.sum(predicted[mask])), int(mask.sum())).as_dict()
    subtypes = np.array([subtype_of.get(sample_id, "") for sample_id in run.sample_ids])
    by_subtype = {
        subtype: wilson(int(np.sum(predicted[subtypes == subtype])), int(np.sum(subtypes == subtype))).as_dict()
        for subtype in sorted(set(subtype_of.values()))
    }
    return {
        "threshold_0.5": rates(run.y_true, predicted),
        "fpr_by_source": by_source,
        "fpr_by_rfi_subtype": by_subtype,
        "failures": int(run.failed.sum()),
    }


def interval_from(entry: dict[str, float]) -> Interval:
    return Interval(entry["estimate"], entry["low"], entry["high"])


def unpaired(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Primeiro - segundo para cada taxa de ``rates``, entre amostras independentes."""
    return {
        key: newcombe_unpaired_difference(interval_from(first[key]), interval_from(second[key])).as_dict()
        for key in ("accuracy", "recall", "fpr")
    }


def operating_points_transfer(
    vlm_select: Run, swin_select: Run, vlm_apply: Run, swin_apply: Run
) -> list[dict[str, Any]]:
    """Item 1 do R0: o limiar do SwinYNet que casa a FPR (ou o recall) do VLM em cada
    nivel e escolhido no conjunto de selecao e avaliado no de aplicacao."""
    y = vlm_apply.y_true
    levels = np.unique(
        np.concatenate([run.scores[~run.failed] for run in (vlm_select, vlm_apply)])
    )
    rows = []
    for level in levels[1:]:
        selected = vlm_select.predict(level)
        targets = {
            "fpr_matched": float(np.mean(selected[~vlm_select.y_true])),
            "recall_matched": float(np.mean(selected[vlm_select.y_true])),
        }
        thresholds = {
            "fpr_matched": threshold_matching_fpr(
                swin_select.y_true, swin_select.scores, targets["fpr_matched"]
            ),
            "recall_matched": threshold_matching_recall(
                swin_select.y_true, swin_select.scores, targets["recall_matched"]
            ),
        }
        vlm_predicted = vlm_apply.predict(level)
        row: dict[str, Any] = {"vlm_level": float(level), "vlm": rates(y, vlm_predicted)}
        for name, threshold in thresholds.items():
            swin_predicted = swin_apply.predict(threshold)
            row[name] = {
                "target_on_selection": targets[name],
                "swinynet_threshold": threshold,
                "swinynet": rates(y, swin_predicted),
                "vlm_minus_swinynet": paired_comparison(y, vlm_predicted, swin_predicted),
            }
        rows.append(row)
    return rows


def hybrids_transfer(
    vlm_select: Run, swin_select: Run, vlm_apply: Run, swin_apply: Run
) -> dict[str, Any]:
    """Item 2 do R0 com o limiar do SwinYNet escolhido no conjunto de selecao."""
    y = vlm_apply.y_true
    selected = vlm_select.predict(0.5)
    thresholds = {
        "fixed_0.5": 0.5,
        "fpr_matched": threshold_matching_fpr(
            swin_select.y_true, swin_select.scores, float(np.mean(selected[~vlm_select.y_true]))
        ),
        "recall_matched": threshold_matching_recall(
            swin_select.y_true, swin_select.scores, float(np.mean(selected[vlm_select.y_true]))
        ),
        "max_accuracy": threshold_max_accuracy(swin_select.y_true, swin_select.scores),
    }
    vlm_predicted = vlm_apply.predict(0.5)
    result = {}
    for name, threshold in thresholds.items():
        swin_predicted = swin_apply.predict(threshold)
        entry: dict[str, Any] = {
            "swinynet_threshold": threshold,
            "swinynet_alone": rates(y, swin_predicted),
            "vlm_minus_swinynet": paired_comparison(y, vlm_predicted, swin_predicted, bootstrap=False),
        }
        for rule, combined in (
            ("and", vlm_predicted & swin_predicted),
            ("or", vlm_predicted | swin_predicted),
        ):
            entry[rule] = {
                "hybrid": rates(y, combined),
                "hybrid_minus_swinynet": paired_comparison(y, combined, swin_predicted, bootstrap=False),
            }
        result[name] = entry
    return result


def sigma_bin_label(sigma: float) -> str:
    for low, high in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
        if low <= sigma < high:
            return f"[{low:g}, {high:g})" if np.isfinite(high) else f">= {low:g}"
    raise ValueError(f"sigma fora dos bins: {sigma}")


def recall_by_sigma_bin(run: Run, sigma_of: dict[str, float], threshold: float = 0.5) -> dict[str, Any]:
    predicted = run.predict(threshold)
    bins: dict[str, list[bool]] = {}
    for source_id, is_frb, hit in zip(run.source_ids, run.y_true, predicted):
        if is_frb:
            bins.setdefault(sigma_bin_label(sigma_of[source_id]), []).append(bool(hit))
    ordered = [sigma_bin_label(low) for low in SIGMA_BINS[:-1]]
    return {
        label: {"n": len(bins[label]), **wilson(sum(bins[label]), len(bins[label])).as_dict()}
        for label in ordered
        if label in bins
    }


def prompt_variants(
    arms: dict[str, Run], multiclass_frb: np.ndarray, sigma_of: dict[str, float]
) -> dict[str, Any]:
    """C9: as variantes da Tabela 10 nas mesmas imagens. ``multiclass_frb`` e a decisao
    FRB do multiclasse nas mesmas amostras, como o braco multiclasse da Tabela 10."""
    full = arms["binary_full"]
    y = full.y_true
    predicted = {arm: run.predict(0.5) for arm, run in arms.items()}
    predicted["multiclass"] = multiclass_frb
    faint = np.array(
        [is_frb and sigma_of[source] < FAINT_SIGMA for source, is_frb in zip(full.source_ids, y)]
    )
    result: dict[str, Any] = {"arms": {}, "pairs": {}}
    for arm, decisions in predicted.items():
        result["arms"][arm] = {
            "binary": rates(y, decisions),
            "failures": int(arms[arm].failed.sum()) if arm in arms else None,
            "agreement_with_full": wilson(
                int(np.sum(decisions == predicted["binary_full"])), len(y)
            ).as_dict(),
        }
    for first, second in (
        ("binary_lean", "binary_full"),
        ("binary_lean_reason", "binary_full"),
        ("binary_lean_reason", "binary_lean"),
        ("multiclass", "binary_full"),
    ):
        result["pairs"][f"{first}_minus_{second}"] = {
            "accuracy": paired_comparison(y, predicted[first], predicted[second]),
            "frb_recall": paired_comparison(y[y], predicted[first][y], predicted[second][y]),
            "frb_recall_faint_descriptive": paired_comparison(
                y[faint], predicted[first][faint], predicted[second][faint], bootstrap=False
            ),
        }
    result["faint_definition"] = f"FRB com peak_sigma < {FAINT_SIGMA:g} (n = {int(faint.sum())})"
    return result


def multiclass_as_binary(multiclass_path: Path, binary: Run, multiclass_source_of: dict[str, str]) -> np.ndarray:
    """Decisao FRB do multiclasse nas amostras do binario (pareadas pelo id de origem)."""
    label_by_source = {
        multiclass_source_of[r["sample_id"]]: str(r.get("predicted_label")).upper()
        for r in read_jsonl(multiclass_path)
    }
    labels = [label_by_source[source] for source in binary.source_ids]
    predicted = np.array([label == "FRB" for label in labels])
    failed = np.array([label in ("NONE", "") for label in labels])
    return np.where(failed, ~binary.y_true, predicted)


# ---------------------------------------------------------------------------


def _source_map(path: Path) -> dict[str, str]:
    return {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(path)}


def _subtype_map(path: Path) -> dict[str, str]:
    """Subtipo de RFI por amostra, como no R0."""
    return {
        r["sample_id"]: "+".join(r["simulation_parameters"].get("rfi_types", []))
        for r in read_jsonl(path)
        if r["true_label"] == "RFI"
    }


def _load_set(name: str) -> dict[str, Any]:
    paths = SETS[name]
    source_of = _source_map(paths["binary_manifest"])
    arms = ("binary_full",) if name == "benchmark" else BINARY_ARMS
    runs = {f"{vlm}_{arm}": load_run(paths[f"{vlm}_{arm}"], source_of) for vlm in VLMS for arm in arms}
    runs["swinynet"] = load_run(paths["swinynet"], source_of)
    check_aligned(runs)
    if runs["swinynet"].failed.any():
        raise ValueError(f"{name}: SwinYNet sem score em alguma amostra.")
    return {
        "runs": runs,
        "subtypes": _subtype_map(paths["binary_manifest"]),
        "sigma": read_sigma(paths["visibility"]),
    }


def _analyze(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    data = {name: _load_set(name) for name in SETS}
    multiclass_source = {
        "benchmark": _source_map(Path("dataset/metadata/image_manifest.jsonl")),
        "e1": _source_map(Path("dataset_e1/metadata/image_manifest.jsonl")),
    }

    results: dict[str, Any] = {"binary_0.5": {}, "generalization": {}, "c2_vlm_minus_swinynet": {}}
    for name, entry in data.items():
        runs = entry["runs"]
        results["binary_0.5"][name] = {
            run_name: binary_summary(run, entry["subtypes"]) for run_name, run in runs.items()
        }
        results["c2_vlm_minus_swinynet"][name] = {
            vlm: paired_comparison(
                runs["swinynet"].y_true,
                runs[f"{vlm}_binary_full"].predict(0.5),
                runs["swinynet"].predict(0.5),
            )
            for vlm in VLMS
        }
    for run_name in ("e2b_binary_full", "e4b_binary_full", "swinynet"):
        results["generalization"][run_name] = unpaired(
            results["binary_0.5"]["e1"][run_name]["threshold_0.5"],
            results["binary_0.5"]["benchmark"][run_name]["threshold_0.5"],
        )

    results["c1_transfer"] = {}
    for select, apply in (("e1", "benchmark"), ("benchmark", "e1")):
        s, a = data[select]["runs"], data[apply]["runs"]
        results["c1_transfer"][f"chosen_on_{select}_applied_to_{apply}"] = {
            vlm: {
                "operating_points": operating_points_transfer(
                    s[f"{vlm}_binary_full"], s["swinynet"], a[f"{vlm}_binary_full"], a["swinynet"]
                ),
                "hybrids": hybrids_transfer(
                    s[f"{vlm}_binary_full"], s["swinynet"], a[f"{vlm}_binary_full"], a["swinynet"]
                ),
            }
            for vlm in VLMS
        }

    e1_runs = data["e1"]["runs"]
    results["c9_prompt_variants"] = {
        vlm: prompt_variants(
            {arm: e1_runs[f"{vlm}_{arm}"] for arm in BINARY_ARMS},
            multiclass_as_binary(
                SETS["e1"][f"{vlm}_multiclass"], e1_runs[f"{vlm}_binary_full"], multiclass_source["e1"]
            ),
            data["e1"]["sigma"],
        )
        for vlm in VLMS
    }

    results["c7_multiclass"] = {}
    for vlm in VLMS:
        e1 = multiclass_rates(read_jsonl(SETS["e1"][f"{vlm}_multiclass"]))
        benchmark = multiclass_rates(read_jsonl(SETS["benchmark"][f"{vlm}_multiclass"]))
        results["c7_multiclass"][vlm] = {
            "e1": e1,
            "benchmark_all_3000": benchmark,
            "e1_minus_benchmark_accuracy": newcombe_unpaired_difference(
                interval_from(e1["accuracy"]), interval_from(benchmark["accuracy"])
            ).as_dict(),
        }

    results["sigma_bins"] = {
        name: {
            run_name: recall_by_sigma_bin(run, entry["sigma"])
            for run_name, run in entry["runs"].items()
        }
        for name, entry in data.items()
    }

    inputs = {
        name: {key: {"path": path.as_posix(), "sha256": sha256_file(path)} for key, path in paths.items()}
        for name, paths in SETS.items()
    }
    write_json(
        E1_DIR / "e1_results.json",
        run_manifest(experiment="E1-analysis", project_root=project_root, inputs=inputs, results=results),
    )
    for name in SETS:
        for run_name in ("e2b_binary_full", "e4b_binary_full", "swinynet"):
            acc = results["binary_0.5"][name][run_name]["threshold_0.5"]["accuracy"]
            print(f"{name:9s} {run_name:16s} acuracia {acc['estimate']:.4f} [{acc['low']:.4f}; {acc['high']:.4f}]")


def _import_swinynet(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    export_manifest = args.workdir / "metadata" / "benchmark_manifest.jsonl"
    raw_output = E1_DIR / "swinynet" / "predictions_export_ids.jsonl"
    imported = ExternalPredictionImporter(
        config=ExternalPredictionImportConfig(
            benchmark_manifest=export_manifest,
            external_predictions=args.external,
            output_path=raw_output,
            threshold=0.5,
            aggregation="max",
        ),
        project_root=project_root,
    ).import_predictions(overwrite=True)

    # id do export (sample_000000...) -> origem (frb_00000) -> id do binario do E1
    source_by_export = {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(export_manifest)}
    binary_by_source = {
        r["source_sample_id"]: r["sample_id"] for r in read_jsonl(SETS["e1"]["binary_manifest"])
    }
    remapped = sorted(
        (
            {
                **record,
                "export_sample_id": record["sample_id"],
                "sample_id": binary_by_source[source_by_export[record["sample_id"]]],
            }
            for record in imported
        ),
        key=lambda record: record["sample_id"],
    )
    if len(remapped) != len(binary_by_source):
        raise ValueError(f"{len(remapped)} predicoes para {len(binary_by_source)} amostras.")
    write_jsonl(SETS["e1"]["swinynet"], remapped)
    raw_output.unlink()
    write_json(
        E1_DIR / "swinynet" / "import_manifest.json",
        run_manifest(
            experiment="E1-SwinYNet-import",
            project_root=project_root,
            export_manifest={"path": str(export_manifest), "sha256": sha256_file(export_manifest)},
            external_output={"path": str(args.external), "sha256": sha256_file(args.external)},
            threshold=0.5,
            aggregation="max",
            n=len(remapped),
        ),
    )
    print(f"{len(remapped)} predicoes do SwinYNet em {SETS['e1']['swinynet']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import-swinynet", help="Importar a saida do SwinYNet.")
    importer.add_argument("--workdir", type=Path, required=True)
    importer.add_argument("--external", type=Path, required=True)
    importer.set_defaults(func=_import_swinynet)
    analyze = subparsers.add_parser("analyze", help="Tabelas do E1.")
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
