"""Comparacoes pareadas entre rodadas binarias nas mesmas imagens (E4, E6 e E8).

Uma rodada e lida como no E1 (``e1_analysis.load_run``): decisao FRB com frb_probability
>= 0,5, e uma resposta que falhou vira a decisao errada.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from revision_v4.common import SIGMA_BINS, read_jsonl
from revision_v4.e1_analysis import Run, load_run, sigma_bin_label
from revision_v4.r0_reanalysis import paired_comparison, rates
from revision_v4.stats import newcombe_unpaired_difference, wilson


ALPHA = 0.05


def load(path: Path, keys: dict[str, str] | None = None) -> Run:
    """``keys`` leva o sample_id a chave do sigma (o proprio id se None)."""
    if keys is None:
        keys = {r["sample_id"]: r["sample_id"] for r in read_jsonl(path)}
    return load_run(path, keys)


def restrict(run: Run, sample_ids: set[str]) -> Run:
    mask = np.array([sample_id in sample_ids for sample_id in run.sample_ids])
    return Run(
        sample_ids=tuple(s for s, keep in zip(run.sample_ids, mask) if keep),
        source_ids=tuple(s for s, keep in zip(run.source_ids, mask) if keep),
        y_true=run.y_true[mask],
        scores=run.scores[mask],
        source_labels=tuple(s for s, keep in zip(run.source_labels, mask) if keep),
    )


def check_same_samples(first: Run, second: Run, name: str) -> None:
    if first.sample_ids != second.sample_ids or not np.array_equal(first.y_true, second.y_true):
        raise ValueError(f"{name}: amostras ou rotulos diferentes da rodada de referencia.")


def summary(run: Run) -> dict[str, Any]:
    return {"binary_0.5": rates(run.y_true, run.predict(0.5)), "failures": int(run.failed.sum())}


def versus(first: Run, second: Run, *, bootstrap: bool = True) -> dict[str, Any]:
    """Primeiro - segundo: acuracia, recall de FRB e especificidade (1 - FPR), pareados."""
    check_same_samples(first, second, "versus")
    y = first.y_true
    a, b = first.predict(0.5), second.predict(0.5)
    result: dict[str, Any] = {"accuracy": paired_comparison(y, a, b, bootstrap=bootstrap)}
    if y.any():
        result["frb_recall"] = paired_comparison(y[y], a[y], b[y], bootstrap=bootstrap)
    if (~y).any():
        result["specificity"] = paired_comparison(y[~y], a[~y], b[~y], bootstrap=bootstrap)
        result["false_positives"] = {"first": int(a[~y].sum()), "second": int(b[~y].sum()), "n": int((~y).sum())}
    return result


def significant(comparison: dict[str, Any]) -> bool:
    return comparison["mcnemar"]["p_value_exact"] < ALPHA


def agreement(first_path: Path, second_path: Path, sample_ids: set[str] | None = None) -> dict[str, int]:
    """Mesmas imagens, prompt e flags: quantas respostas brutas e rotulos coincidem."""
    second = {r["sample_id"]: r for r in read_jsonl(second_path)}
    records = [r for r in read_jsonl(first_path) if sample_ids is None or r["sample_id"] in sample_ids]
    return {
        "n": len(records),
        "identical_raw_response": sum(r["raw_model_response"] == second[r["sample_id"]]["raw_model_response"] for r in records),
        "same_label": sum(r["predicted_label"] == second[r["sample_id"]]["predicted_label"] for r in records),
    }


def read_visibility(path: Path, key: str = "sample_id") -> dict[str, dict[str, Any]]:
    """Linhas FRB de um CSV de visibilidade, pela coluna ``key``."""
    with path.open(encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle) if row["true_label"] == "FRB"}


def bins_of(run: Run, sigma_of: dict[str, float]) -> np.ndarray:
    """Bin de sigma de cada FRB da rodada ('' nos NON_FRB)."""
    return np.array(
        [sigma_bin_label(sigma_of[key]) if is_frb else "" for key, is_frb in zip(run.source_ids, run.y_true)]
    )


def ordered_bins() -> list[str]:
    return [sigma_bin_label(low) for low in SIGMA_BINS[:-1]]


def recall_by_bin_paired(first: Run, second: Run, sigma_of: dict[str, float]) -> dict[str, Any]:
    """Recall por bin das duas rodadas e o McNemar exato (primeiro - segundo) em cada bin."""
    check_same_samples(first, second, "recall_by_bin_paired")
    labels = bins_of(first, sigma_of)
    a, b = first.predict(0.5), second.predict(0.5)
    result = {}
    for label in ordered_bins():
        mask = labels == label
        if not mask.any():
            continue
        n = int(mask.sum())
        result[label] = {
            "n": n,
            "first": wilson(int(a[mask].sum()), n).as_dict(),
            "second": wilson(int(b[mask].sum()), n).as_dict(),
            "first_minus_second": paired_comparison(first.y_true[mask], a[mask], b[mask], bootstrap=False),
        }
    return result


def recall_by_bin(run: Run, sigma_of: dict[str, float], mask: np.ndarray | None = None) -> dict[str, Any]:
    labels = bins_of(run, sigma_of)
    hits = run.predict(0.5)
    keep = np.ones(len(labels), dtype=bool) if mask is None else mask
    result = {}
    for label in ordered_bins():
        selected = (labels == label) & keep
        if selected.any():
            n = int(selected.sum())
            result[label] = {"n": n, **wilson(int(hits[selected].sum()), n).as_dict()}
    return result


def unpaired_by_bin(first: dict[str, Any], second: dict[str, Any], min_n: int) -> dict[str, Any]:
    """Diferenca de recall primeiro - segundo por bin (Newcombe), nos bins com >= min_n nos dois."""
    from revision_v4.stats import Interval

    result = {}
    for label in ordered_bins():
        if label in first and label in second and first[label]["n"] >= min_n and second[label]["n"] >= min_n:
            a, b = first[label], second[label]
            difference = newcombe_unpaired_difference(
                Interval(a["estimate"], a["low"], a["high"]), Interval(b["estimate"], b["low"], b["high"])
            ).as_dict()
            result[label] = {"n_first": a["n"], "n_second": b["n"], "difference": difference,
                             "significant": not difference["low"] <= 0.0 <= difference["high"]}
    return result
