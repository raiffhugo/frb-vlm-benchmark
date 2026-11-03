"""E6: analise da sensibilidade ao prompt (plano, secao 10.8; definicoes na 10.12).

Le as rodadas results_v4/e6/<modelo>_<conjunto>_<prompt> e grava e6_results.json (com --models
so de um modelo, e6_results_<modelo>.json) com, por modelo:
  - S200 e R200: acuracia, recall e FPR (Wilson) de cada prompt; McNemar exato contra o P0 da
    mesma rodada (s200_p0, r200_p0) em acuracia, recall e especificidade; a faixa minimo-maximo
    de cada taxa entre as parafrases P-a, P-b e P-c; e a concordancia do P0 da rodada com o P0
    do artigo nas mesmas imagens;
  - o R nos conjuntos inteiros (1600 reais e 2000 simulados) contra o P0 do artigo, McNemar em
    acuracia, recall e especificidade;
  - o recall do R e do P0 do artigo nos bursts do FRB 20121102A (coluna ``source`` de
    burst_visibility_full.csv), McNemar exato.
Decisao com frb_probability >= 0,5; uma resposta que falhou vira a decisao errada.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from revision_v4.common import RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json
from revision_v4.paired_runs import agreement, load, read_visibility, restrict, summary, versus
from revision_v4.r0_reanalysis import paired_comparison
from revision_v4.stats import wilson


OUTPUT_DIR = RESULTS_ROOT / "e6"
MODELS = ("e2b", "e4b")
SUBSETS = ("s200", "r200")
PROMPTS = ("p0", "pa", "pb", "pc", "pd", "pe", "r")
PARAPHRASES = ("pa", "pb", "pc")
RATES = ("accuracy", "recall", "fpr")
PAPER_RUNS = {
    "sim": {"e2b": Path("results_vlm2b_0t_binary/predictions.jsonl"), "e4b": Path("results_vlm_0t_binary/predictions.jsonl")},
    "real": {"e2b": Path("results_real_binary_full_2b/predictions.jsonl"), "e4b": Path("results_real_binary_full_4b/predictions.jsonl")},
}
SUBSET_DOMAIN = {"s200": "sim", "r200": "real"}
FULL_SETS = {"sim2000_r": "sim", "real1600_r": "real"}
REAL_VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
FRB20121102 = "FRB20121102"


def output(model: str, arm: str) -> Path:
    return OUTPUT_DIR / f"{model}_{arm}" / "predictions.jsonl"


def paraphrase_range(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for rate in RATES:
        values = [summaries[name]["binary_0.5"][rate]["estimate"] for name in PARAPHRASES]
        result[rate] = {"min": min(values), "max": max(values)}
    return result


def _subset(model: str, subset: str) -> dict[str, Any]:
    runs = {prompt: load(output(model, f"{subset}_{prompt}")) for prompt in PROMPTS}
    summaries = {prompt: summary(run) for prompt, run in runs.items()}
    ids = set(runs["p0"].sample_ids)
    return {
        "prompts": summaries,
        "minus_p0": {prompt: versus(runs[prompt], runs["p0"]) for prompt in PROMPTS if prompt != "p0"},
        "paraphrase_range": paraphrase_range(summaries),
        "p0_agreement_with_paper": agreement(output(model, f"{subset}_p0"), PAPER_RUNS[SUBSET_DOMAIN[subset]][model], ids),
    }


def _full_set(model: str, arm: str) -> dict[str, Any]:
    relaxed = load(output(model, arm))
    paper = restrict(load(PAPER_RUNS[FULL_SETS[arm]][model]), set(relaxed.sample_ids))
    return {"r": summary(relaxed), "paper_p0": summary(paper), "r_minus_paper_p0": versus(relaxed, paper)}


def source_recall(model: str, sources: dict[str, str]) -> dict[str, Any]:
    relaxed = load(output(model, "real1600_r"))
    paper = restrict(load(PAPER_RUNS["real"][model]), set(relaxed.sample_ids))
    mask = np.array([sources.get(sample_id) == FRB20121102 for sample_id in relaxed.sample_ids])
    r_hits, p0_hits = relaxed.predict(0.5)[mask], paper.predict(0.5)[mask]
    n = int(mask.sum())
    return {
        "n": n,
        "r": wilson(int(r_hits.sum()), n).as_dict(),
        "paper_p0": wilson(int(p0_hits.sum()), n).as_dict(),
        "r_minus_paper_p0": paired_comparison(np.ones(n, dtype=bool), r_hits, p0_hits, bootstrap=False),
    }


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    sources = {key: row["source"] for key, row in read_visibility(REAL_VISIBILITY).items()}
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {"real_visibility": {"path": REAL_VISIBILITY.as_posix(), "sha256": sha256_file(REAL_VISIBILITY)}}
    for model in args.models:
        results[model] = {
            **{subset: _subset(model, subset) for subset in SUBSETS},
            **{arm: _full_set(model, arm) for arm in FULL_SETS},
            "frb20121102_recall": source_recall(model, sources),
        }
        arms = [f"{subset}_{prompt}" for subset in SUBSETS for prompt in PROMPTS] + list(FULL_SETS)
        inputs[model] = {arm: {"path": output(model, arm).as_posix(), "sha256": sha256_file(output(model, arm))} for arm in arms}
    write_json(
        OUTPUT_DIR / ("e6_results.json" if len(args.models) == len(MODELS) else f"e6_results_{'_'.join(args.models)}.json"),
        run_manifest(experiment="E6-analysis", project_root=project_root, inputs=inputs, results=results),
    )
    for model in args.models:
        for subset in SUBSETS:
            accuracy = {p: results[model][subset]["prompts"][p]["binary_0.5"]["accuracy"]["estimate"] for p in PROMPTS}
            print(model, subset, " ".join(f"{p}={v:.3f}" for p, v in accuracy.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Tabelas do E6.")
    analyze.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
