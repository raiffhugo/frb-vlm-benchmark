"""Regenerate the article's real-data (FAST-FREX) figures.

Reads the artifacts produced by the real-data run --- the image manifest of
``dataset_real_full``, the per-sample predictions of both VLMs, and the
dedispersed-peak diagnostic of ``check_burst_visibility.py`` --- and writes
vector PDFs (used by the LaTeX article) plus 300-dpi PNG previews into
``figures/``.

Three figures are produced:

``examples_real``
    Three representative anonymized spectra exactly as supplied to the model:
    a bright real burst, a catalogued burst with no detectable signal in this
    representation, and a negative sample.
``recall_vs_snr``
    Recall of both VLMs as a function of the dedispersed peak significance of
    the burst, which is the central diagnostic of the section.
``snr_hist_real``
    Distribution of that significance for the 600 positives and the 1000
    negatives, showing how much of the positive set sits at the noise floor.
    The simulated bursts, measured on the same time--frequency grid, are
    overlaid: they occupy a regime the real set barely reaches, which is what
    makes the recall gap between the two benchmarks interpretable.

The style (serif fonts, colours, figure sizes) mirrors ``make_article_figs.py``
so the real-data figures match the simulated ones.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
MANIFEST = ROOT / "dataset_real_full" / "metadata" / "image_manifest.jsonl"
VISIBILITY = ROOT / "fast_frex" / "burst_visibility_full.csv"
VISIBILITY_SIM = ROOT / "dataset" / "metadata" / "burst_visibility_sim.csv"

MODELS = [
    {"key": "2b", "label": "Gemma 4 E2B", "color": "tab:blue"},
    {"key": "4b", "label": "Gemma 4 E4B", "color": "tab:orange"},
]

# Samples chosen for the illustrative figure: a bright burst, a catalogued
# burst with no detectable signal in this representation, and a typical negative.
EXAMPLES = [
    ("sample_000477", "Real FRB, detectable"),
    ("sample_000320", "Real FRB, below noise floor"),
    ("sample_000616", "Negative (RFI/noise)"),
]

BANDS = [(0.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, np.inf)]
BAND_LABELS = [r"$<5$", r"$5\!-\!8$", r"$8\!-\!15$", r"$>15$"]

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


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def load_inputs():
    manifest = {}
    with MANIFEST.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                manifest[record["sample_id"]] = record
    visibility = {row["sample_id"]: row for row in csv.DictReader(VISIBILITY.open())}
    predictions = {}
    for model in MODELS:
        path = ROOT / f"results_real_binary_full_{model['key']}" / "predictions.jsonl"
        with path.open() as handle:
            predictions[model["key"]] = {
                json.loads(line)["sample_id"]: json.loads(line)
                for line in handle
                if line.strip()
            }
    return manifest, visibility, predictions


def figure_examples(manifest, visibility) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
    for ax, (sample_id, caption) in zip(axes, EXAMPLES):
        image = mpimg.imread(ROOT / manifest[sample_id]["image_path"])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        # Always the single-trial measurement: for the positives it coincides with
        # peak_sigma, and for the negative it avoids advertising a maximum over 24
        # DMs next to 1-DM measurements, which is the asymmetry the text corrects.
        row = visibility[sample_id]
        sigma = float(row.get("peak_sigma_matched") or row["peak_sigma"])
        ax.set_title(f"{caption}\n" rf"peak $= {sigma:.1f}\,\sigma$", fontsize=11)
    _save(fig, "examples_real")


def figure_recall_vs_snr(visibility, predictions) -> None:
    positives = [s for s, row in visibility.items() if row["true_label"] == "FRB"]
    counts, rates, errors = [], {}, {}
    for low, high in BANDS:
        ids = [s for s in positives if low <= float(visibility[s]["peak_sigma"]) < high]
        counts.append(len(ids))
        for model in MODELS:
            hits = sum(
                1 for s in ids if predictions[model["key"]][s]["predicted_label"] == "FRB"
            )
            rates.setdefault(model["key"], []).append(hits / len(ids) if ids else 0.0)
            # Plain binomial error bar, only to indicate the support of the band.
            p = hits / len(ids) if ids else 0.0
            errors.setdefault(model["key"], []).append(
                np.sqrt(p * (1 - p) / len(ids)) if ids else 0.0
            )

    x = np.arange(len(BANDS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for offset, model in zip((-width / 2, width / 2), MODELS):
        ax.bar(
            x + offset,
            rates[model["key"]],
            width,
            yerr=errors[model["key"]],
            capsize=3,
            label=model["label"],
            color=model["color"],
            alpha=0.85,
        )
    ax.set_xticks(x)
    # The support of each band goes into the tick label itself: as loose text
    # below the axis it collided with the x-axis label.
    ax.set_xticklabels(
        [f"{label}\n$n={n}$" for label, n in zip(BAND_LABELS, counts)]
    )
    ax.set_xlabel(r"Dedispersed peak significance ($\sigma$)")
    ax.set_ylabel("Recall on real FRBs")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "recall_vs_snr")


def figure_snr_hist(visibility) -> None:
    positives = np.array(
        [float(r["peak_sigma"]) for r in visibility.values() if r["true_label"] == "FRB"]
    )
    negatives = np.array(
        [float(r["peak_sigma"]) for r in visibility.values() if r["true_label"] == "RFI"]
    )
    # The symmetric floor (1 drawn DM, the same number of trials as the positives)
    # is what compares against the positives; the blind one, maximized over 24 DMs,
    # is shown only as a conservative upper bound.
    matched = np.array(
        [
            float(r["peak_sigma_matched"])
            for r in visibility.values()
            if r["true_label"] == "RFI" and r.get("peak_sigma_matched")
        ]
    )

    # Simulated bursts measured on the SAME grid, restricted to those with a
    # contained sweep: they are the population the article's simulated recall
    # was obtained on.
    simulated = np.array([])
    if VISIBILITY_SIM.exists():
        simulated = np.array(
            [
                float(row["peak_sigma"])
                for row in csv.DictReader(VISIBILITY_SIM.open())
                if row["true_label"] == "FRB" and row.get("sweep_contained") == "1"
            ]
        )

    top = max(900.0, simulated.max() * 1.1 if simulated.size else 0.0)
    bins = np.logspace(np.log10(1.5), np.log10(top), 44)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hist(negatives, bins=bins, alpha=0.55, label="Real negatives, $n=1000$", color="tab:orange")
    ax.hist(positives, bins=bins, alpha=0.55, label="Real catalogued FRBs, $n=600$", color="tab:blue")
    if simulated.size:
        ax.hist(
            simulated,
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color="tab:green",
            label=rf"Simulated FRBs, $n={simulated.size}$",
        )

    reference = matched if matched.size else negatives
    p99 = float(np.percentile(reference, 99))
    ax.axvline(p99, color="0.3", linestyle="--", linewidth=1.2)
    ax.text(
        p99 * 1.10,
        ax.get_ylim()[1] * 0.97,
        rf"99th pct. of negatives ({p99:.1f}$\,\sigma$)",
        fontsize=10,
        rotation=90,
        va="top",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Dedispersed peak significance ($\sigma$)")
    ax.set_ylabel("Number of samples")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    _save(fig, "snr_hist_real")


def main() -> None:
    manifest, visibility, predictions = load_inputs()
    figure_examples(manifest, visibility)
    figure_recall_vs_snr(visibility, predictions)
    figure_snr_hist(visibility)


if __name__ == "__main__":
    main()
