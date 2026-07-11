"""Compose a 3-panel example figure (FRB | RFI | NOISE) for the article.

Reads three anonymized dynamic-spectrum PNGs from ``dataset/images/samples``
and writes a side-by-side figure (vector PDF for the LaTeX article + 300-dpi PNG
preview for the Markdown draft) into
``article/latex/figures/examples_classes.{pdf,png}``. The class labels and the
RFI subtype shown in the panel titles are read from -- and checked against -- the
dataset image manifest, so the figure and caption stay faithful to the data.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "dataset" / "images" / "samples"
MANIFEST = ROOT / "dataset" / "metadata" / "image_manifest.jsonl"
# Ver a nota em make_article_figs.py: o artigo compilado hoje le de figures_v2,
# e escrever so em latex/figures fazia a regeracao passar despercebida.
FIG_DIRS = [ROOT / "figures"]

# Panels of the class-example figure, taken from the multiclass dataset that
# steps 1-2 of the README rebuild. The published figure was rendered from a
# separate simulation run that is not part of this release, so these ids are
# not the ones in the paper; the classes, and the RFI subtype named in the
# caption (impulsive_broadband), are the same.
PANELS = [
    {"sample_id": "sample_000003", "expect": "FRB"},
    {"sample_id": "sample_001677", "expect": "RFI"},
    {"sample_id": "sample_002000", "expect": "NOISE"},
]

plt.rcParams.update(
    {"font.family": "serif", "axes.titlesize": 14, "axes.labelsize": 12}
)


def _manifest_index() -> dict[str, dict]:
    index = {}
    with MANIFEST.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                index[rec["sample_id"]] = rec
    return index


def _title(rec: dict, expect: str) -> str:
    label = rec["true_label"]
    assert label == expect, f"{rec['sample_id']}: label {label} != expected {expect}"
    if label == "RFI":
        subtype = rec.get("simulation_parameters", {}).get("rfi_category", "")
        return f"RFI ({subtype.replace('_', ' ')})" if subtype else "RFI"
    return label


def main() -> None:
    index = _manifest_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, panel in zip(axes, PANELS):
        rec = index[panel["sample_id"]]
        img = mpimg.imread(SAMPLES_DIR / f"{panel['sample_id']}.png")
        ax.imshow(img, aspect="auto")
        ax.set_title(_title(rec, panel["expect"]))
        # The source PNGs already carry calibrated "Time (s)"/"Frequency (MHz)"
        # axes, so we only add the class title and drop the outer frame ticks.
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    for out_dir in FIG_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "examples_classes.pdf")  # vector, for the article
        fig.savefig(out_dir / "examples_classes.png", dpi=300)  # preview for the .md
    plt.close(fig)
    titles = [_title(index[p["sample_id"]], p["expect"]) for p in PANELS]
    print("wrote examples_classes.pdf / .png  | panels:", titles)


if __name__ == "__main__":
    main()
