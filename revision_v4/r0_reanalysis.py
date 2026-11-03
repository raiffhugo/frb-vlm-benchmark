"""R0: reanalises sem GPU (plano, secao 10.2), itens 1 a 7.

O item 8 (largura e ocupacao espectral dos bursts reais) le os 600 FITS e fica em
r0_burst_morphology.py. Os limiares casados usam, ate o E1 ficar pronto, 2 folds
estratificados do proprio benchmark (seed 42): o limiar do SwinYNet e escolhido num fold
e avaliado no outro, e as duas metades avaliadas fora do fold sao somadas.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from revision_v4.common import (
    BOOTSTRAP_SEED,
    RESULTS_ROOT,
    SIGMA_BINS,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
)
from revision_v4.stats import (
    PairedCounts,
    bootstrap_paired_difference,
    mcnemar_chi2_corrected,
    mcnemar_exact,
    newcombe_paired_difference,
    wilson,
)


OUTPUT_DIR = RESULTS_ROOT / "r0"
FOLD_SEED = 42

INPUTS = {
    "e2b_binary": Path("results_vlm2b_0t_binary/predictions.jsonl"),
    "e4b_binary": Path("results_vlm_0t_binary/predictions.jsonl"),
    "swinynet_binary": Path("results_external_binary/predictions.jsonl"),
    "binary_manifest": Path("dataset_binary/metadata/image_manifest.jsonl"),
    "e2b_real": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b_real": Path("results_real_binary_full_4b/predictions.jsonl"),
    "real_manifest": Path("dataset_real_full/metadata/image_manifest.jsonl"),
    "real_visibility": Path("fast_frex/burst_visibility_full.csv"),
    "sim_visibility": Path("dataset/metadata/burst_visibility_sim.csv"),
    "timing_e2b": Path("results_prompt_timing_2b/predictions.jsonl"),
    "timing_e4b": Path("results_prompt_timing_4b/predictions.jsonl"),
    # results_vlm2b_0t_new and results_vlm_0t_new in the working repository: renamed results_vlm2b_0t
    # and results_vlm_0t in this release (same SHA-256); the manifests in results_v4 keep the old names.
    "e2b_multiclass": Path("results_vlm2b_0t/predictions.jsonl"),
    "e4b_multiclass": Path("results_vlm_0t/predictions.jsonl"),
    "multiclass_dev": Path("results_multiclass_subset/subset_manifest.jsonl"),
}
VLMS = ("e2b", "e4b")
TIMING_ARMS = ("binary_full", "binary_lean", "binary_lean_reason", "multiclass")


# ---------------------------------------------------------------------------
# Leitura


def _parsed(record: dict[str, Any], key: str = "parsed_response") -> dict[str, Any]:
    value = record.get(key)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError(f"{record.get('sample_id')}: {key} ausente.")
    return value


@dataclass(frozen=True)
class BinaryRun:
    sample_ids: tuple[str, ...]
    y_true: np.ndarray
    scores: np.ndarray
    source_labels: tuple[str, ...]


def load_binary_run(path: Path) -> BinaryRun:
    records = sorted(read_jsonl(path), key=lambda record: record["sample_id"])
    return BinaryRun(
        sample_ids=tuple(record["sample_id"] for record in records),
        y_true=np.array([record["true_label"] == "FRB" for record in records]),
        scores=np.array([float(_parsed(record)["frb_probability"]) for record in records]),
        source_labels=tuple(str(record["source_true_label"]).upper() for record in records),
    )


# ---------------------------------------------------------------------------
# Metricas


def rates(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(predicted, dtype=bool)
    tp, fn = int(np.sum(y & p)), int(np.sum(y & ~p))
    fp, tn = int(np.sum(~y & p)), int(np.sum(~y & ~p))
    result: dict[str, Any] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    result["accuracy"] = wilson(tp + tn, tp + fn + fp + tn).as_dict()
    result["recall"] = wilson(tp, tp + fn).as_dict() if tp + fn else None
    result["fpr"] = wilson(fp, fp + tn).as_dict() if fp + tn else None
    result["specificity"] = wilson(tn, fp + tn).as_dict() if fp + tn else None
    result["precision_frb"] = wilson(tp, tp + fp).as_dict() if tp + fp else None
    result["precision_non_frb"] = wilson(tn, tn + fn).as_dict() if tn + fn else None
    return result


def paired_comparison(
    y_true: np.ndarray,
    first_predicted: np.ndarray,
    second_predicted: np.ndarray,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Diferenca de acuracia (primeiro - segundo) com Newcombe, bootstrap e McNemar."""
    first_correct = np.asarray(first_predicted, dtype=bool) == np.asarray(y_true, dtype=bool)
    second_correct = np.asarray(second_predicted, dtype=bool) == np.asarray(y_true, dtype=bool)
    counts = PairedCounts.from_outcomes(first_correct, second_correct)
    result: dict[str, Any] = {
        "accuracy_difference_newcombe": newcombe_paired_difference(counts).as_dict(),
        "mcnemar": mcnemar_exact(counts),
        "mcnemar_chi2_corrected_p_v3": mcnemar_chi2_corrected(counts),
    }
    if bootstrap:
        result["accuracy_difference_bootstrap"] = bootstrap_paired_difference(
            first_correct, second_correct, seed=BOOTSTRAP_SEED
        ).as_dict()
    return result


# ---------------------------------------------------------------------------
# Item 1: pontos de operacao casados com folds estratificados


def stratified_folds(
    sample_ids: Sequence[str],
    strata: Sequence[str],
    *,
    seed: int = FOLD_SEED,
) -> np.ndarray:
    """Fold 0/1 por amostra, alternando dentro de cada estrato embaralhado."""
    fold = np.zeros(len(sample_ids), dtype=int)
    by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(index)
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        indices = sorted(by_stratum[stratum], key=lambda i: sample_ids[i])
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold[index] = position % 2
    return fold


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    return np.append(np.unique(scores), np.inf)


def threshold_matching_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Menor limiar (maior recall) cuja FPR nao passa da FPR alvo."""
    negatives = scores[~y_true]
    for threshold in _candidate_thresholds(scores):
        if np.mean(negatives >= threshold) <= target_fpr + 1e-12:
            return float(threshold)
    return float("inf")


def threshold_matching_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float
) -> float:
    """Maior limiar (menor FPR) cujo recall alcanca o recall alvo."""
    positives = scores[y_true]
    best = float(np.min(scores))
    for threshold in _candidate_thresholds(scores):
        if np.mean(positives >= threshold) >= target_recall - 1e-12:
            best = float(threshold)
    return best


def threshold_max_accuracy(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Limiar de maior acuracia; no empate, o menor (favorece o recall)."""
    best_threshold, best_accuracy = 0.5, -1.0
    for threshold in _candidate_thresholds(scores):
        accuracy = float(np.mean((scores >= threshold) == y_true))
        if accuracy > best_accuracy:
            best_threshold, best_accuracy = float(threshold), accuracy
    return best_threshold


def cross_fitted(
    y_true: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    choose: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> tuple[np.ndarray, list[float]]:
    """Predicoes fora do fold: o limiar escolhido no fold k vale para o fold 1-k.

    ``choose(y, scores, mask)`` recebe o fold de selecao e devolve o limiar."""
    predicted = np.zeros_like(y_true, dtype=bool)
    thresholds = []
    for selection in (0, 1):
        mask = folds == selection
        threshold = choose(y_true[mask], scores[mask], mask)
        thresholds.append(threshold)
        predicted[~mask] = scores[~mask] >= threshold
    return predicted, thresholds


def operating_points(
    vlm: BinaryRun,
    swinynet: BinaryRun,
    folds: np.ndarray,
) -> list[dict[str, Any]]:
    y = vlm.y_true
    rows = []
    levels = np.unique(vlm.scores)
    for level in levels[1:]:
        vlm_predicted = vlm.scores >= level

        def fpr_target(y_sel: np.ndarray, s_sel: np.ndarray, mask: np.ndarray) -> float:
            target = float(np.mean(vlm.scores[mask][~y_sel] >= level))
            return threshold_matching_fpr(y_sel, s_sel, target)

        def recall_target(y_sel: np.ndarray, s_sel: np.ndarray, mask: np.ndarray) -> float:
            target = float(np.mean(vlm.scores[mask][y_sel] >= level))
            return threshold_matching_recall(y_sel, s_sel, target)

        row: dict[str, Any] = {"vlm_level": float(level), "vlm": rates(y, vlm_predicted)}
        for name, chooser in (("fpr_matched", fpr_target), ("recall_matched", recall_target)):
            predicted, thresholds = cross_fitted(y, swinynet.scores, folds, chooser)
            row[name] = {
                "swinynet_thresholds_by_selection_fold": thresholds,
                "swinynet": rates(y, predicted),
                "vlm_minus_swinynet": paired_comparison(y, vlm_predicted, predicted),
            }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Item 2: hibridos


def hybrids(vlm: BinaryRun, swinynet: BinaryRun, folds: np.ndarray) -> dict[str, Any]:
    y = vlm.y_true
    vlm_predicted = vlm.scores >= 0.5
    vlm_level = float(np.min(vlm.scores[vlm_predicted])) if vlm_predicted.any() else 0.5

    def fpr_target(y_sel: np.ndarray, s_sel: np.ndarray, mask: np.ndarray) -> float:
        return threshold_matching_fpr(y_sel, s_sel, float(np.mean(vlm_predicted[mask][~y_sel])))

    def recall_target(y_sel: np.ndarray, s_sel: np.ndarray, mask: np.ndarray) -> float:
        return threshold_matching_recall(y_sel, s_sel, float(np.mean(vlm_predicted[mask][y_sel])))

    def max_accuracy(y_sel: np.ndarray, s_sel: np.ndarray, _mask: np.ndarray) -> float:
        return threshold_max_accuracy(y_sel, s_sel)

    result: dict[str, Any] = {"vlm_operating_level": vlm_level}
    choosers = {
        "fixed_0.5": None,
        "fpr_matched": fpr_target,
        "recall_matched": recall_target,
        "max_accuracy": max_accuracy,
    }
    for name, chooser in choosers.items():
        if chooser is None:
            swin_predicted, thresholds = swinynet.scores >= 0.5, [0.5, 0.5]
        else:
            swin_predicted, thresholds = cross_fitted(y, swinynet.scores, folds, chooser)
        entry: dict[str, Any] = {
            "swinynet_thresholds_by_selection_fold": thresholds,
            "swinynet_alone": rates(y, swin_predicted),
        }
        for rule, combined in (
            ("and", vlm_predicted & swin_predicted),
            ("or", vlm_predicted | swin_predicted),
        ):
            entry[rule] = {
                "hybrid": rates(y, combined),
                "hybrid_minus_swinynet": paired_comparison(
                    y, combined, swin_predicted, bootstrap=False
                ),
            }
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# Item 4: Wilson nas Tabelas 5, 9-12


def table5_and_9(runs: dict[str, BinaryRun], rfi_subtypes: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, run in runs.items():
        predicted = run.scores >= 0.5
        by_source: dict[str, Any] = {}
        for source in ("RFI", "NOISE"):
            mask = np.array([label == source for label in run.source_labels])
            by_source[source] = wilson(int(np.sum(predicted[mask])), int(mask.sum())).as_dict()
        by_subtype: dict[str, Any] = {}
        subtype_of = np.array([rfi_subtypes.get(sample_id, "") for sample_id in run.sample_ids])
        for subtype in sorted(set(rfi_subtypes.values())):
            mask = subtype_of == subtype
            by_subtype[subtype] = wilson(int(np.sum(predicted[mask])), int(mask.sum())).as_dict()
        result[name] = {
            "table5_threshold_0.5": rates(run.y_true, predicted),
            "table9_fpr_by_source": by_source,
            "fpr_by_rfi_subtype": by_subtype,
        }
    return result


def _timing_prediction(record: dict[str, Any], arm: str) -> bool:
    parsed = _parsed(record[arm], key="parsed")
    if arm == "multiclass":
        return str(parsed["label"]).upper() == "FRB"
    return float(parsed["frb_probability"]) >= 0.5


def table10_and_item6(path: Path) -> dict[str, Any]:
    records = sorted(read_jsonl(path), key=lambda record: record["sample_id"])
    y = np.array([record["source_true_label"] == "FRB" for record in records])
    predicted = {arm: np.array([_timing_prediction(r, arm) for r in records]) for arm in TIMING_ARMS}
    arms = {}
    for arm in TIMING_ARMS:
        agreement = int(np.sum(predicted[arm] == predicted["binary_full"]))
        arms[arm] = {
            "binary": rates(y, predicted[arm]),
            "agreement_with_full": wilson(agreement, len(records)).as_dict(),
        }
    pairs = {}
    for first, second in (
        ("binary_lean", "binary_full"),
        ("binary_lean_reason", "binary_full"),
        ("multiclass", "binary_full"),
        ("binary_lean_reason", "binary_lean"),
    ):
        pairs[f"{first}_minus_{second}"] = paired_comparison(y, predicted[first], predicted[second])
    return {"n": len(records), "arms": arms, "mcnemar_between_variants": pairs}


def multiclass_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Acuracia e recall/precisao por classe; um rotulo ausente conta como erro."""
    truth = np.array([str(r["true_label"]).upper() for r in rows])
    predicted = np.array([str(r["predicted_label"]).upper() for r in rows])
    per_class = {}
    for label in ("FRB", "RFI", "NOISE"):
        tp = int(np.sum((truth == label) & (predicted == label)))
        per_class[label] = {
            "support": int(np.sum(truth == label)),
            "recall": (
                wilson(tp, int(np.sum(truth == label))).as_dict()
                if np.any(truth == label)
                else None
            ),
            "precision": (
                wilson(tp, int(np.sum(predicted == label))).as_dict()
                if np.any(predicted == label)
                else None
            ),
        }
    return {
        "n": len(rows),
        "accuracy": wilson(int(np.sum(truth == predicted)), len(rows)).as_dict(),
        "per_class": per_class,
        "parse_failures": int(np.sum(predicted == "NONE")),
    }


def multiclass_tables(path: Path, dev_ids: set[str]) -> dict[str, Any]:
    records = read_jsonl(path)
    result = {}
    for subset_name, keep in (
        ("all_3000", lambda sample_id: True),
        ("dev_400", lambda sample_id: sample_id in dev_ids),
        ("outside_dev_2600", lambda sample_id: sample_id not in dev_ids),
    ):
        rows = [record for record in records if keep(record["sample_id"])]
        result[subset_name] = multiclass_rates(rows)
    return result


# ---------------------------------------------------------------------------
# Item 5: recall por bin de sigma, simulado x real


def _bin_label(low: float, high: float) -> str:
    return f"[{low:g}, {high:g})" if np.isfinite(high) else f">= {low:g}"


def sigma_bins(
    sim_rows: list[dict[str, str]],
    sim_hits: dict[str, dict[str, bool]],
    real_rows: list[dict[str, str]],
    real_hits: dict[str, dict[str, bool]],
) -> list[dict[str, Any]]:
    table = []
    for low, high in zip(SIGMA_BINS[:-1], SIGMA_BINS[1:]):
        row: dict[str, Any] = {"bin": _bin_label(low, high), "low": low, "high": high}
        for population, rows, hits in (
            ("simulated", sim_rows, sim_hits),
            ("real", real_rows, real_hits),
        ):
            members = [r["key"] for r in rows if low <= float(r["peak_sigma"]) < high]
            entry: dict[str, Any] = {"n": len(members)}
            for vlm in VLMS:
                k = sum(hits[vlm][key] for key in members)
                entry[vlm] = wilson(k, len(members)).as_dict() if members else None
            row[population] = entry
        table.append(row)
    return table


# ---------------------------------------------------------------------------


def run(project_root: Path) -> dict[str, Any]:
    binary = {
        "e2b": load_binary_run(INPUTS["e2b_binary"]),
        "e4b": load_binary_run(INPUTS["e4b_binary"]),
        "swinynet": load_binary_run(INPUTS["swinynet_binary"]),
    }
    reference = binary["swinynet"]
    for name, item in binary.items():
        if item.sample_ids != reference.sample_ids or not np.array_equal(item.y_true, reference.y_true):
            raise ValueError(f"{name}: amostras ou rotulos diferentes do SwinYNet.")
    folds = stratified_folds(reference.sample_ids, reference.source_labels)

    manifest = read_jsonl(INPUTS["binary_manifest"])
    rfi_subtypes = {
        r["sample_id"]: "+".join(r["simulation_parameters"].get("rfi_types", []))
        for r in manifest
        if r["true_label"] == "RFI"
    }

    results: dict[str, Any] = {
        "folds": {"seed": FOLD_SEED, "sizes": [int(np.sum(folds == k)) for k in (0, 1)]},
        "item1_operating_points": {
            vlm: operating_points(binary[vlm], reference, folds) for vlm in VLMS
        },
        "item2_hybrids": {vlm: hybrids(binary[vlm], reference, folds) for vlm in VLMS},
        "item3_accuracy_difference_at_0.5": {
            vlm: paired_comparison(
                reference.y_true, binary[vlm].scores >= 0.5, reference.scores >= 0.5
            )
            for vlm in VLMS
        },
        "item4_wilson": {
            "tables_5_and_9": table5_and_9(binary, rfi_subtypes),
            "table_10": {
                vlm: table10_and_item6(INPUTS[f"timing_{vlm}"])["arms"] for vlm in VLMS
            },
        },
    }

    # Item 5
    sim_rows = [
        {**row, "key": row["source_sample_id"]}
        for row in csv.DictReader(INPUTS["sim_visibility"].open(encoding="utf-8"))
        if row["true_label"] == "FRB" and row["sweep_contained"] == "1"
    ]
    source_by_binary_id = {r["sample_id"]: r["source_sample_id"] for r in manifest}
    sim_hits = {
        vlm: {
            source_by_binary_id[sample_id]: bool(score >= 0.5)
            for sample_id, score in zip(binary[vlm].sample_ids, binary[vlm].scores)
        }
        for vlm in VLMS
    }
    real_rows = [
        {**row, "key": row["source_sample_id"]}
        for row in csv.DictReader(INPUTS["real_visibility"].open(encoding="utf-8"))
        if row["true_label"] == "FRB"
    ]
    real_source = {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(INPUTS["real_manifest"])}
    real_binary = {vlm: load_binary_run(INPUTS[f"{vlm}_real"]) for vlm in VLMS}
    real_hits = {
        vlm: {
            real_source[sample_id]: bool(score >= 0.5)
            for sample_id, score in zip(real_binary[vlm].sample_ids, real_binary[vlm].scores)
        }
        for vlm in VLMS
    }
    results["item4_wilson"]["table_12_real"] = {
        vlm: rates(real_binary[vlm].y_true, real_binary[vlm].scores >= 0.5) for vlm in VLMS
    }
    results["item5_sigma_bins"] = sigma_bins(sim_rows, sim_hits, real_rows, real_hits)

    # Item 6
    results["item6_prompt_variants"] = {
        vlm: table10_and_item6(INPUTS[f"timing_{vlm}"])["mcnemar_between_variants"] for vlm in VLMS
    }

    # Item 7
    dev_ids = {r["sample_id"] for r in read_jsonl(INPUTS["multiclass_dev"])}
    results["item7_multiclass"] = {
        vlm: multiclass_tables(INPUTS[f"{vlm}_multiclass"], dev_ids) for vlm in VLMS
    }
    results["item4_wilson"]["table_11"] = {
        vlm: results["item7_multiclass"][vlm]["all_3000"] for vlm in VLMS
    }
    return results


def _write_tables(results: dict[str, Any]) -> None:
    def fmt(interval: dict[str, float] | None) -> str:
        if interval is None:
            return ""
        return f"{interval['estimate']:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]"

    with (OUTPUT_DIR / "operating_points.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "vlm", "vlm_level", "vlm_recall", "vlm_fpr",
                "swin_fpr_matched_thresholds", "swin_recall_at_fpr_match", "swin_fpr_at_fpr_match",
                "acc_diff_fpr_match_newcombe", "mcnemar_p_fpr_match",
                "swin_recall_matched_thresholds", "swin_recall_at_recall_match",
                "swin_fpr_at_recall_match", "acc_diff_recall_match_newcombe",
                "mcnemar_p_recall_match",
            ]
        )
        for vlm, rows in results["item1_operating_points"].items():
            for row in rows:
                f, r = row["fpr_matched"], row["recall_matched"]
                writer.writerow(
                    [
                        vlm, row["vlm_level"], fmt(row["vlm"]["recall"]), fmt(row["vlm"]["fpr"]),
                        ";".join(f"{t:.7g}" for t in f["swinynet_thresholds_by_selection_fold"]),
                        fmt(f["swinynet"]["recall"]), fmt(f["swinynet"]["fpr"]),
                        fmt(f["vlm_minus_swinynet"]["accuracy_difference_newcombe"]),
                        f"{f['vlm_minus_swinynet']['mcnemar']['p_value_exact']:.3g}",
                        ";".join(f"{t:.7g}" for t in r["swinynet_thresholds_by_selection_fold"]),
                        fmt(r["swinynet"]["recall"]), fmt(r["swinynet"]["fpr"]),
                        fmt(r["vlm_minus_swinynet"]["accuracy_difference_newcombe"]),
                        f"{r['vlm_minus_swinynet']['mcnemar']['p_value_exact']:.3g}",
                    ]
                )

    with (OUTPUT_DIR / "sigma_bins.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin", "sim_n", "sim_e2b", "sim_e4b", "real_n", "real_e2b", "real_e4b"])
        for row in results["item5_sigma_bins"]:
            sim, real = row["simulated"], row["real"]
            writer.writerow(
                [
                    row["bin"], sim["n"], fmt(sim["e2b"]), fmt(sim["e4b"]),
                    real["n"], fmt(real["e2b"]), fmt(real["e4b"]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    project_root = Path.cwd().resolve()
    results = run(project_root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "r0_results.json", results)
    _write_tables(results)
    write_json(
        OUTPUT_DIR / "run_manifest.json",
        run_manifest(
            experiment="R0",
            project_root=project_root,
            inputs={name: {"path": p.as_posix(), "sha256": sha256_file(p)} for name, p in INPUTS.items()},
            fold_seed=FOLD_SEED,
            bootstrap={"seed": BOOTSTRAP_SEED},
            sigma_bins=[b if np.isfinite(b) else "inf" for b in SIGMA_BINS],
        ),
    )
    print(f"R0 gravado em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
