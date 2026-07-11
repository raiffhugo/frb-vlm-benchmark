"""Regenerate the article's ROC and score-histogram figures from saved scores.

Reads ``results_comparison_{2b,4b}_0t/paired_scores.csv`` (no inference needed)
and writes a vector PDF (for the LaTeX article) plus a 300-dpi PNG preview (for
the Markdown draft) into ``figures/``. The external detector is
labelled ``SwinYNet`` (instead of the generic "External") and the VLM curve is
labelled with the specific Gemma model name. The recipe (bins, AUC in legend,
colours) mirrors ``benchmark_predictions/compare.py`` so the figures stay
faithful; an assertion checks that the recomputed ROC-AUC matches the article.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = [ROOT / "figures"]

EXTERNAL_LABEL = "SwinYNet"
VLM_COLOR = "tab:blue"
EXTERNAL_COLOR = "tab:orange"
HIST_BINS = np.linspace(0.0, 1.0, 21)

MODELS = [
    {"key": "2b", "vlm_label": "Gemma 4 E2B", "expected_auc": 0.9520},
    {"key": "4b", "vlm_label": "Gemma 4 E4B", "expected_auc": 0.9248},
]

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def _load(csv_path: Path):
    y_true, score_vlm, score_external = [], [], []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            y_true.append(int(row["y_true"]))
            score_vlm.append(float(row["score_vlm"]))
            score_external.append(float(row["score_external"]))
    return (
        np.asarray(y_true, dtype=int),
        np.asarray(score_vlm, dtype=float),
        np.asarray(score_external, dtype=float),
    )


def _save(fig, stem: str) -> None:
    for out_dir in FIG_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.pdf")  # vector, used by the LaTeX article
        fig.savefig(out_dir / f"{stem}.png", dpi=300)  # raster preview for the .md
    plt.close(fig)


def _roc(y_true, score_vlm, score_external, vlm_label: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for label, scores, color in (
        (vlm_label, score_vlm, VLM_COLOR),
        (EXTERNAL_LABEL, score_external, EXTERNAL_COLOR),
    ):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, label=f"{label} AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, stem)


def _hist(score_vlm, score_external, vlm_label: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hist(score_vlm, bins=HIST_BINS, alpha=0.55, label=vlm_label, color=VLM_COLOR)
    ax.hist(
        score_external,
        bins=HIST_BINS,
        alpha=0.55,
        label=EXTERNAL_LABEL,
        color=EXTERNAL_COLOR,
    )
    ax.set_xlabel("Predicted FRB probability")
    ax.set_ylabel("Samples")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    for model in MODELS:
        csv_path = ROOT / f"results_comparison_{model['key']}_0t" / "paired_scores.csv"
        y_true, score_vlm, score_external = _load(csv_path)
        auc = roc_auc_score(y_true, score_vlm)
        assert abs(auc - model["expected_auc"]) < 5e-4, (
            f"{model['key']}: recomputed ROC-AUC {auc:.4f} "
            f"!= article value {model['expected_auc']}"
        )
        _roc(y_true, score_vlm, score_external, model["vlm_label"], f"roc_{model['key']}")
        _hist(score_vlm, score_external, model["vlm_label"], f"score_hist_{model['key']}")
        print(
            f"{model['key']}: ROC-AUC(VLM)={auc:.4f} OK "
            f"-> roc_{model['key']}.pdf, score_hist_{model['key']}.pdf (+ .png)"
        )


if __name__ == "__main__":
    main()
