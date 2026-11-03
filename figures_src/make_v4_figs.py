"""New figures of the v4 revision (plan, section 7 item 3 and section 13.2 item 4).

Reads the revision results (``results_v4/``) and the runs behind them, checks every plotted
number against the analysis JSONs, and writes vector PDFs plus 300-dpi PNG previews into
``figures/`` (the paper itself is not part of this repository).

``recall_vs_snr_injection``
    Replaces Fig. 7. Recall of each VLM on the 600 real bursts in the sigma bins of section 10.1,
    with the E2 injection-recovery overlaid: recall per bin of measured sigma, the logistic fit
    with its bootstrap band, and sigma50.
``roc_operating_points``
    C1. The benchmark ROC of Fig. 3 with the operating point of each level of the quantized VLM
    score and SwinYNet at the threshold that matches the FPR of each level, chosen on the E1
    held-out set and applied to the benchmark.
``recall_sim_real_e8``
    C3/C10. Recall per sigma bin of the simulated bursts with a contained sweep, through the
    original and the decimated rendering (E8), against the real bursts.

The style (serif fonts, sizes, colours, hatching-free secondary encoding by marker and line
style) follows ``make_article_figs.py`` and ``make_real_figs.py``: E2B blue, E4B orange, the VLM
blue and SwinYNet orange in the ROC, real bursts in neutral grey.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# the paths in revision_v4 are relative to the repository root
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from revision_v4.common import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, read_jsonl  # noqa: E402
from revision_v4.e1_analysis import sigma_bin_label  # noqa: E402
from revision_v4.e2_analysis import MANIFEST as E2_MANIFEST  # noqa: E402
from revision_v4.e2_analysis import fit_logistic  # noqa: E402
from revision_v4.e2_analysis import output as e2_output  # noqa: E402
from revision_v4.e8_decimated import BINARY_MANIFEST, MIN_BURSTS_PER_BIN, PAPER_RUNS, SIM_VISIBILITY  # noqa: E402
from revision_v4.e8_decimated import output as e8_output  # noqa: E402
from revision_v4.paired_runs import Run, load, ordered_bins, read_visibility  # noqa: E402
from revision_v4.stats import wilson  # noqa: E402

OUT_DIR = ROOT / "figures"
REAL_VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
REAL_RUNS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
MODELS = [
    {"key": "e2b", "short": "2b", "label": "Gemma 4 E2B", "color": "tab:blue", "expected_auc": 0.9520,
     "roc_label_offset": (8, -14)},
    {"key": "e4b", "short": "4b", "label": "Gemma 4 E4B", "color": "tab:orange", "expected_auc": 0.9248,
     "roc_label_offset": (8, 2)},
]
REAL_COLOR = "0.25"
VLM_COLOR = "tab:blue"
SWIN_COLOR = "tab:orange"
SIGMA_LIMITS = (1.5, 1500.0)
BIN_TEXT = {
    "[0, 5)": r"$<5$", "[5, 8)": r"$5\!-\!8$", "[8, 15)": r"$8\!-\!15$", "[15, 30)": r"$15\!-\!30$",
    "[30, 60)": r"$30\!-\!60$", "[60, 100)": r"$60\!-\!100$", "[100, 300)": r"$100\!-\!300$", ">= 300": r"$\geq 300$",
}
ROC_FPR_LIMIT = 0.16

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def _results(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))["results"]


def _check(label: str, computed: float, expected: float, tolerance: float = 1e-9) -> None:
    assert abs(computed - expected) <= tolerance, f"{label}: {computed} != {expected} (JSON da analise)"


def binned(run: Run, sigma_of: dict[str, float], mask: np.ndarray, min_n: int = 1) -> list[dict]:
    """Recall por bin de sigma (secao 10.1) entre as amostras de ``mask``, com Wilson 95% e a
    mediana do sigma dos membros (a posicao do ponto no eixo log)."""
    hits = run.predict(0.5)
    sigma = np.array([sigma_of.get(key, np.nan) for key in run.source_ids])
    labels = np.array([sigma_bin_label(s) if keep else "" for s, keep in zip(sigma, mask)])
    rows = []
    for label in ordered_bins():
        members = labels == label
        n = int(members.sum())
        if n < min_n:
            continue
        interval = wilson(int(hits[members].sum()), n)
        rows.append({"bin": label, "n": n, "sigma": float(np.median(sigma[members])), **interval.as_dict()})
    return rows


def _check_bins(name: str, rows: list[dict], expected: dict) -> None:
    for row in rows:
        entry = expected[row["bin"]]
        _check(f"{name} {row['bin']} n", row["n"], entry["n"])
        _check(f"{name} {row['bin']} recall", row["estimate"], entry["estimate"])


def _errorbars(ax, rows: list[dict], x=None, **style) -> None:
    """Recall com o IC de Wilson; no eixo de sigma, o ponto fica na mediana do bin."""
    x = np.array([r["sigma"] for r in rows]) if x is None else np.asarray(x)
    y = np.array([r["estimate"] for r in rows])
    yerr = np.array([[r["estimate"] - r["low"] for r in rows], [r["high"] - r["estimate"] for r in rows]])
    ax.errorbar(x, y, yerr=yerr, linestyle="none", capsize=2.5, elinewidth=1.0, capthick=1.0, **style)


def _sigma_axes(ax, title: str) -> None:
    ax.set_xscale("log")
    ax.set_xlim(*SIGMA_LIMITS)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"Dedispersed peak significance ($\sigma$)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3, linewidth=0.6)


def _real_bins(model: dict) -> list[dict]:
    real = load(REAL_RUNS[model["key"]])
    sigma = {key: float(row["peak_sigma"]) for key, row in read_visibility(REAL_VISIBILITY).items()}
    return binned(real, sigma, real.y_true)


# ---------------------------------------------------------------------------
# Fig. 7 of v4: injection-recovery (E2) over the real recall


def logistic_band(sigma: np.ndarray, detected: np.ndarray, grid: np.ndarray) -> dict:
    """Curva logistica em log10(sigma) e banda de 95% por bootstrap das injecoes, com as mesmas
    reamostras de ``e2_analysis.sigma50_with_bootstrap`` (10 000, seed 42), para que o IC do
    sigma50 da figura seja o do JSON."""
    x = np.log10(sigma)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    curves, sigma50 = [], []
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, sigma.size, sigma.size)
        fit = fit_logistic(x[index], detected[index].astype(float))
        if fit is not None:
            curves.append(expit(fit[0] + fit[1] * np.log10(grid)))
            sigma50.append(10 ** (-fit[0] / fit[1]))
    a, b = fit_logistic(x, detected.astype(float))
    curves = np.array(curves)
    return {
        "a_b": (a, b),
        "curve": expit(a + b * np.log10(grid)),
        "low": np.percentile(curves, 2.5, axis=0),
        "high": np.percentile(curves, 97.5, axis=0),
        "sigma50": 10 ** (-a / b),
        "sigma50_low": float(np.percentile(sigma50, 2.5)),
        "sigma50_high": float(np.percentile(sigma50, 97.5)),
    }


def figure_injection() -> None:
    results = _results("results_v4/e2/e2_results.json")
    manifest = {r["sample_id"]: r for r in read_jsonl(E2_MANIFEST)}
    measured = {sid: float(r["measured_sigma"]) for sid, r in manifest.items() if r["role"] == "injected"}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        expected = results[model["key"]]
        real_rows = _real_bins(model)
        _check_bins(f"{model['key']} real", real_rows, expected["real_recall_by_sigma_bin"])

        run = load(e2_output(model["key"]))
        injection_rows = binned(run, measured, run.y_true)
        _check_bins(f"{model['key']} E2", injection_rows, expected["recall_by_measured_sigma_bin"])
        sigma = np.array([measured[s] for s, injected in zip(run.sample_ids, run.y_true) if injected])
        detected = run.predict(0.5)[run.y_true]
        grid = np.logspace(np.log10(sigma.min()), np.log10(sigma.max()), 300)
        fit = logistic_band(sigma, detected, grid)
        reference = expected["sigma50"]
        _check(f"{model['key']} a", fit["a_b"][0], reference["logistic_a_b"][0], 1e-8)
        _check(f"{model['key']} sigma50", fit["sigma50"], reference["sigma50"], 1e-8)
        _check(f"{model['key']} sigma50 low", fit["sigma50_low"], reference["low"], 1e-8)
        _check(f"{model['key']} sigma50 high", fit["sigma50_high"], reference["high"], 1e-8)

        color = model["color"]
        ax.fill_between(grid, fit["low"], fit["high"], color=color, alpha=0.15, linewidth=0)
        ax.plot(grid, fit["curve"], color=color, linewidth=1.8)
        _errorbars(ax, injection_rows, marker="s", markersize=7, color=color, markeredgecolor="white",
                   markeredgewidth=1.0, zorder=4)
        _errorbars(ax, real_rows, marker="o", markersize=7, color=REAL_COLOR, markeredgecolor="white",
                   markeredgewidth=1.0, zorder=5)
        ax.errorbar([fit["sigma50"]], [0.5], xerr=[[fit["sigma50"] - fit["sigma50_low"]], [fit["sigma50_high"] - fit["sigma50"]]],
                    marker="D", markersize=6, color=color, markeredgecolor="black", markeredgewidth=0.8,
                    elinewidth=1.2, capsize=3, linestyle="none", zorder=6)
        ax.annotate(
            rf"$\sigma_{{50}} = {fit['sigma50']:.1f}$ [{fit['sigma50_low']:.1f}, {fit['sigma50_high']:.1f}]",
            xy=(fit["sigma50_high"], 0.5), xytext=(8, -4), textcoords="offset points", fontsize=11, va="top",
        )
        _sigma_axes(ax, model["label"])
        handles = [
            Line2D([], [], marker="o", color=REAL_COLOR, markeredgecolor="white", markersize=7, linestyle="none",
                   label="Real catalogued FRBs ($n=600$)"),
            Line2D([], [], marker="s", color=color, markeredgecolor="white", markersize=7, linestyle="none",
                   label="Injections into real negatives ($n=1024$)"),
            Line2D([], [], color=color, linewidth=1.8, label="Logistic fit to the injections, 95% band"),
        ]
        ax.legend(handles=handles, loc="lower right")
    axes[0].set_ylabel("Recall")
    fig.tight_layout()
    _save(fig, "recall_vs_snr_injection")


# ---------------------------------------------------------------------------
# C1: benchmark ROC with the operating points


def _paired_scores(short: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, vlm, swin = [], [], []
    with (ROOT / f"results_comparison_{short}_0t" / "paired_scores.csv").open() as handle:
        for row in csv.DictReader(handle):
            y.append(int(row["y_true"]))
            vlm.append(float(row["score_vlm"]))
            swin.append(float(row["score_external"]))
    return np.array(y, dtype=bool), np.array(vlm), np.array(swin)


def _rates(y: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    return float(np.mean(predicted[~y])), float(np.mean(predicted[y]))


def figure_roc() -> None:
    transfer = _results("results_v4/e1/e1_results.json")["c1_transfer"]["chosen_on_e1_applied_to_benchmark"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        y, vlm, swin = _paired_scores(model["short"])
        _check(f"{model['key']} AUC", roc_auc_score(y, vlm), model["expected_auc"], 5e-4)
        for scores, color, style in ((vlm, VLM_COLOR, "-"), (swin, SWIN_COLOR, "--")):
            fpr, tpr, _ = roc_curve(y, scores)
            ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=1.6, zorder=2)

        offscale = []
        for row in transfer[model["key"]]["operating_points"]:
            level = row["vlm_level"]
            vlm_point = _rates(y, vlm >= level)
            _check(f"{model['key']} VLM {level} FPR", vlm_point[0], row["vlm"]["fpr"]["estimate"])
            _check(f"{model['key']} VLM {level} recall", vlm_point[1], row["vlm"]["recall"]["estimate"])
            matched = row["fpr_matched"]
            swin_point = _rates(y, swin >= matched["swinynet_threshold"])
            _check(f"{model['key']} Swin {level} FPR", swin_point[0], matched["swinynet"]["fpr"]["estimate"])
            _check(f"{model['key']} Swin {level} recall", swin_point[1], matched["swinynet"]["recall"]["estimate"])
            if max(vlm_point[0], swin_point[0]) > ROC_FPR_LIMIT:
                offscale.append(level)
                continue
            ax.plot(*vlm_point, marker="o", markersize=7, color=VLM_COLOR, markeredgecolor="white", zorder=4)
            ax.plot(*swin_point, marker="^", markersize=8, color=SWIN_COLOR, markeredgecolor="white", zorder=4)

        # the fixed 0.5 threshold, the comparison of v3
        at_half = {"vlm": _rates(y, vlm >= 0.5), "swin": _rates(y, swin >= 0.5)}
        ax.plot(*at_half["swin"], marker="s", markersize=8, markerfacecolor="white", markeredgecolor=SWIN_COLOR,
                markeredgewidth=1.6, linestyle="none", zorder=5)
        for point, offset in ((at_half["vlm"], model["roc_label_offset"]), (at_half["swin"], (-10, -16))):
            ax.annotate("0.5", xy=point, xytext=offset, textcoords="offset points", fontsize=10, color="0.2")
        ax.set_xlim(-0.003, ROC_FPR_LIMIT)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xlabel("False Positive Rate")
        ax.set_title(model["label"], fontsize=13)
        ax.grid(alpha=0.3, linewidth=0.6)
        handles = [
            Line2D([], [], color=VLM_COLOR, marker="o", markeredgecolor="white", markersize=7, linewidth=1.6,
                   label=f"{model['label']}: ROC and score levels"),
            Line2D([], [], color=SWIN_COLOR, linestyle="--", linewidth=1.6, label="SwinYNet: ROC"),
            Line2D([], [], color=SWIN_COLOR, marker="^", markeredgecolor="white", markersize=8, linestyle="none",
                   label="SwinYNet at the VLM's FPR (threshold from E1)"),
            Line2D([], [], color=SWIN_COLOR, marker="s", markerfacecolor="white", markeredgewidth=1.6, markersize=8,
                   linestyle="none", label="SwinYNet at the fixed 0.5 threshold"),
        ]
        ax.legend(handles=handles, loc="lower right")
        print(f"{model['key']}: niveis fora da escala de FPR (<= {ROC_FPR_LIMIT}): {offscale}")
    axes[0].set_ylabel("True Positive Rate")
    fig.tight_layout()
    _save(fig, "roc_operating_points")


# ---------------------------------------------------------------------------
# C3/C10: simulated vs real through the two rendering paths (E8)


def figure_e8() -> None:
    results = _results("results_v4/e8/e8_results.json")
    sim_rows = read_visibility(SIM_VISIBILITY, key="source_sample_id")
    sim_sigma = {key: float(row["peak_sigma"]) for key, row in sim_rows.items()}
    source_of = {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(BINARY_MANIFEST)}
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, model in zip(axes, MODELS):
        expected = results[model["key"]]
        real_rows = _real_bins(model)
        _check_bins(f"{model['key']} real", real_rows, expected["real_recall_by_sigma_bin"])
        paths = {}
        for path, run in (("original", load(PAPER_RUNS[model["key"]], source_of)),
                          ("decimated", load(e8_output(model["key"]), source_of))):
            contained = np.array([is_frb and sim_rows[key]["sweep_contained"] == "1"
                                  for key, is_frb in zip(run.source_ids, run.y_true)])
            paths[path] = binned(run, sim_sigma, contained, min_n=MIN_BURSTS_PER_BIN)
            for row in paths[path]:
                comparison = expected["sim_contained_minus_real"][path][row["bin"]]
                real = expected["real_recall_by_sigma_bin"][row["bin"]]
                _check(f"{model['key']} {path} {row['bin']} n", row["n"], comparison["n_first"])
                _check(f"{model['key']} {path} {row['bin']} diff", row["estimate"] - real["estimate"],
                       comparison["difference"]["estimate"])
        color = model["color"]
        # each bin is one position, with the real, the original and the decimated side by side
        position = {label: i for i, label in enumerate(ordered_bins())}

        def at(rows: list[dict], offset: float) -> list[float]:
            return [position[r["bin"]] + offset for r in rows]

        _errorbars(ax, real_rows, at(real_rows, -0.22), marker="o", markersize=7, color=REAL_COLOR,
                   markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        _errorbars(ax, paths["original"], at(paths["original"], 0.0), marker="s", markersize=7,
                   markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6, color=color, zorder=4)
        _errorbars(ax, paths["decimated"], at(paths["decimated"], 0.22), marker="s", markersize=7, color=color,
                   markeredgecolor="white", markeredgewidth=1.0, zorder=4)
        n_real = {r["bin"]: r["n"] for r in real_rows}
        n_sim = {r["bin"]: r["n"] for r in paths["original"]}
        ax.set_xticks(range(len(position)))
        ax.set_xticklabels([
            BIN_TEXT[label] + f"\n$n={n_real[label]}$" + (f"\nsim. ${n_sim[label]}$" if label in n_sim else "")
            for label in position
        ])
        ax.set_xlim(-0.6, len(position) - 0.4)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(r"Dedispersed peak significance ($\sigma$)")
        ax.set_title(model["label"], fontsize=13)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)
        n_contained = expected["contained_frb"]
        handles = [
            Line2D([], [], marker="o", color=REAL_COLOR, markeredgecolor="white", markersize=7, linestyle="none",
                   label="Real catalogued FRBs ($n=600$)"),
            Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6,
                   markersize=7, linestyle="none", label=f"Simulated, original rendering ($n={n_contained}$)"),
            Line2D([], [], marker="s", color=color, markeredgecolor="white", markersize=7, linestyle="none",
                   label="Simulated, decimated rendering (E8)"),
        ]
        # closer, so that it does not touch the CI of the 60-100 sigma bin of E4B
        ax.legend(handles=handles, loc="lower right", labelspacing=0.3, borderaxespad=0.4)
    axes[0].set_ylabel("Recall")
    fig.tight_layout()
    _save(fig, "recall_sim_real_e8")


def main() -> None:
    figure_injection()
    figure_roc()
    figure_e8()


if __name__ == "__main__":
    main()
