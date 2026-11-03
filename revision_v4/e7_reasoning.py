"""E7: ablacao do raciocinio no held-out do E1 (plano, secao 10.9; emenda de 19/10 na 10.12).

  1. ``select``: as 400 imagens do binario do E1 (200 FRB, 20 por subtipo de RFI e 100 NOISE),
     pelo sorteador da Tabela 10 (``prompt_timing_experiment.choose_subsample``) com a seed
     20260927, em results_v4/selections/e7_heldout400.jsonl.
  2. ``analyze``: le as seis variantes de cada modelo em results_v4/e7/ e grava e7_results.json
     (com --models so de um modelo, e7_results_<modelo>.json) com as taxas a 0,5, os McNemar no
     recall de FRB, a regra, a copia do filler do V4 e a concordancia de V1 e V2 com as mesmas
     400 amostras nas rodadas do E1.

Regra emendada (decide; avaliada no E2B, o modelo da afirmacao, e so descritiva no E4B): a
afirmacao causal volta so se
  - V2 > V1o e V2 > V1 no recall de FRB, cada um com McNemar exato p < 0,05;
  - o V4 nao reproduz o ganho (nao ha V4 > V1o com p < 0,05) e V2 > V4 com p < 0,05;
  - o V5 mantem o efeito (V5 > V1o com p < 0,05).
Se V3 > V1o com p < 0,05, o efeito e atribuido ao pedido de justificativa, nao a posicao dela.
A regra original da secao 10.9 (com o V1 no lugar do V1o) tambem e calculada e relatada.
Uma resposta sem frb_probability valida conta como decisao errada, como no E1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import prompt_timing_experiment
from revision_v4.common import (
    MODEL_REVISIONS,
    RESULTS_ROOT,
    SIM_SEED,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)
from revision_v4.e1_analysis import FAINT_SIGMA, SETS, Run, load_run, read_sigma, recall_by_sigma_bin
from revision_v4.e7_prompts import FILLER_TOKENS, filler
from revision_v4.gpu_runs import MODELS
from revision_v4.r0_reanalysis import paired_comparison, rates


E7_DIR = RESULTS_ROOT / "e7"
SAMPLE = RESULTS_ROOT / "selections" / "e7_heldout400.jsonl"
E1_BINARY = Path("dataset_e1_binary/metadata/image_manifest.jsonl")
COMPOSITION = {"n_frb": 200, "per_rfi_subtype": 20, "n_noise": 100}
ARMS = ("v1_lean", "v1o_no_reason", "v2_reason_before", "v3_reason_after", "v4_filler", "v5_reason_reworded")
ALPHA = 0.05
RULE_MODEL = "e2b"
# (primeiro, segundo): diferenca primeiro - segundo
PAIRS = (
    ("v2_reason_before", "v1o_no_reason"),
    ("v2_reason_before", "v1_lean"),
    ("v4_filler", "v1o_no_reason"),
    ("v4_filler", "v1_lean"),
    ("v2_reason_before", "v4_filler"),
    ("v5_reason_reworded", "v1o_no_reason"),
    ("v5_reason_reworded", "v1_lean"),
    ("v3_reason_after", "v1o_no_reason"),
    ("v2_reason_before", "v3_reason_after"),
    ("v1o_no_reason", "v1_lean"),
)
# V1 e V2 sao os prompts das rodadas lean e lean+reason do E1
E1_ARM = {"v1_lean": "binary_lean", "v2_reason_before": "binary_lean_reason"}


def select(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen, _ = prompt_timing_experiment.choose_subsample(records, seed=SIM_SEED, **COMPOSITION)
    return chosen


def pair_name(first: str, second: str) -> str:
    return f"{first}_minus_{second}"


def compare(runs: dict[str, Run], mask: np.ndarray | None = None) -> dict[str, Any]:
    """McNemar e Newcombe pareados em acuracia e em recall de FRB para cada par de PAIRS."""
    y = next(iter(runs.values())).y_true
    frb = y if mask is None else y & mask
    predicted = {arm: run.predict(0.5) for arm, run in runs.items()}
    result = {}
    for first, second in PAIRS:
        entry = {"frb_recall": paired_comparison(y[frb], predicted[first][frb], predicted[second][frb])}
        if mask is None:
            entry["accuracy"] = paired_comparison(y, predicted[first], predicted[second])
        result[pair_name(first, second)] = entry
    return result


def significantly_better(pairs: dict[str, Any], first: str, second: str) -> bool:
    entry = pairs[pair_name(first, second)]["frb_recall"]
    return (
        entry["accuracy_difference_newcombe"]["estimate"] > 0
        and entry["mcnemar"]["p_value_exact"] < ALPHA
    )


def evaluate_rule(pairs: dict[str, Any], baseline: str) -> dict[str, Any]:
    """A regra com ``baseline`` como o braco sem justificativa (V1o na emendada, V1 na original)."""
    conditions = {
        f"v2_gt_{baseline}": significantly_better(pairs, "v2_reason_before", baseline),
        f"v4_does_not_reproduce_gain_over_{baseline}": not significantly_better(pairs, "v4_filler", baseline),
        f"v5_gt_{baseline}": significantly_better(pairs, "v5_reason_reworded", baseline),
    }
    if baseline == "v1o_no_reason":
        conditions["v2_gt_v1_lean"] = significantly_better(pairs, "v2_reason_before", "v1_lean")
        conditions["v2_gt_v4_filler"] = significantly_better(pairs, "v2_reason_before", "v4_filler")
    return {"conditions": conditions, "causal_claim_returns": all(conditions.values())}


def filler_compliance(records: list[dict[str, Any]], expected: str, tokenizer: Any) -> dict[str, Any]:
    """Quantas respostas do V4 copiam o filler e quantos tokens o campo tem."""
    exact, tokens = 0, []
    for record in records:
        try:
            raw = record["raw_model_response"]
            value = json.loads(raw[raw.index("{") : raw.rindex("}") + 1]).get("filler", "")
        except (ValueError, AttributeError):
            value = ""
        value = value if isinstance(value, str) else ""
        exact += value == expected
        tokens.append(len(tokenizer(value, add_special_tokens=False)["input_ids"]) if value else 0)
    return {
        "n": len(records),
        "exact_copy": exact,
        "expected_tokens": len(tokenizer(expected, add_special_tokens=False)["input_ids"]),
        "mean_tokens": float(np.mean(tokens)),
        "min_tokens": int(np.min(tokens)),
        "max_tokens": int(np.max(tokens)),
    }


def reason_tokens(records: list[dict[str, Any]], tokenizer: Any) -> float:
    reasons = [str((r.get("parsed_response") or {}).get("reason", "")) for r in records]
    return float(np.mean([len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in reasons]))


def agreement_with_e1(e7_records: list[dict[str, Any]], e1_path: Path) -> dict[str, int]:
    """V1 e V2 rodam de novo no E7; mesmas imagens, prompt e flags que no E1."""
    e1 = {r["sample_id"]: r for r in read_jsonl(e1_path)}
    identical = sum(r["raw_model_response"] == e1[r["sample_id"]]["raw_model_response"] for r in e7_records)
    same = sum(r["predicted_label"] == e1[r["sample_id"]]["predicted_label"] for r in e7_records)
    return {"n": len(e7_records), "identical_raw_response": identical, "same_label": same}


def output(model: str, arm: str) -> Path:
    return E7_DIR / f"{model}_{arm}" / "predictions.jsonl"


# ---------------------------------------------------------------------------


def _select(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    chosen = select(prompt_timing_experiment.load_manifest(E1_BINARY))
    write_jsonl(SAMPLE, chosen)
    composition: dict[str, int] = {}
    for record in chosen:
        key = record["true_label"]
        if key == "RFI":
            key = f"RFI/{prompt_timing_experiment.rfi_subtype(record)}"
        composition[key] = composition.get(key, 0) + 1
    write_json(
        SAMPLE.with_name("e7_selection_manifest.json"),
        run_manifest(
            experiment="E7-selection",
            project_root=project_root,
            seed=SIM_SEED,
            sampler="prompt_timing_experiment.choose_subsample",
            requested=COMPOSITION,
            composition=composition,
            input={"path": E1_BINARY.as_posix(), "sha256": sha256_file(E1_BINARY)},
            output={"path": SAMPLE.as_posix(), "sha256": sha256_file(SAMPLE), "n": len(chosen)},
        ),
    )
    print(f"{len(chosen)} amostras em {SAMPLE}: {composition}")


def _analyze(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    project_root = Path.cwd().resolve()
    source_of = {r["sample_id"]: r["source_sample_id"] for r in read_jsonl(SAMPLE)}
    sigma_of = read_sigma(SETS["e1"]["visibility"])
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {"sample": {"path": SAMPLE.as_posix(), "sha256": sha256_file(SAMPLE)}}
    for model in args.models:
        model_id = MODELS[model]
        runs = {arm: load_run(output(model, arm), source_of) for arm in ARMS}
        reference = runs["v1_lean"]
        for arm, run in runs.items():
            if run.sample_ids != reference.sample_ids:
                raise ValueError(f"{model} {arm}: amostras diferentes das outras variantes.")
        faint = np.array([is_frb and sigma_of[s] < FAINT_SIGMA for s, is_frb in zip(reference.source_ids, reference.y_true)])
        pairs = compare(runs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISIONS[model_id])
        records = {arm: read_jsonl(output(model, arm)) for arm in ARMS}
        results[model] = {
            "arms": {
                arm: {
                    "binary_0.5": rates(run.y_true, run.predict(0.5)),
                    "failures": int(run.failed.sum()),
                    "frb_recall_by_sigma_bin": recall_by_sigma_bin(run, sigma_of),
                }
                for arm, run in runs.items()
            },
            "pairs": pairs,
            "pairs_faint_descriptive": compare(runs, faint),
            "faint_definition": f"FRB com peak_sigma < {FAINT_SIGMA:g} (n = {int(faint.sum())})",
            "rule_amended": evaluate_rule(pairs, "v1o_no_reason"),
            "rule_original": evaluate_rule(pairs, "v1_lean"),
            "v3_reproduces_gain": significantly_better(pairs, "v3_reason_after", "v1o_no_reason"),
            "filler": filler_compliance(records["v4_filler"], filler(FILLER_TOKENS[model]), tokenizer),
            "mean_reason_tokens": {
                arm: reason_tokens(records[arm], tokenizer)
                for arm in ("v2_reason_before", "v3_reason_after", "v5_reason_reworded")
            },
            "agreement_with_e1": {
                arm: agreement_with_e1(records[arm], SETS["e1"][f"{model}_{e1_arm}"])
                for arm, e1_arm in E1_ARM.items()
            },
        }
        inputs[model] = {arm: {"path": output(model, arm).as_posix(), "sha256": sha256_file(output(model, arm))} for arm in ARMS}
    if RULE_MODEL in results:
        results["rule_model"] = RULE_MODEL
        results["decision"] = results[RULE_MODEL]["rule_amended"]["causal_claim_returns"]
    write_json(
        E7_DIR / ("e7_results.json" if len(args.models) == len(MODELS) else f"e7_results_{'_'.join(args.models)}.json"),
        run_manifest(experiment="E7-analysis", project_root=project_root, inputs=inputs, alpha=ALPHA, results=results),
    )
    for model in args.models:
        recalls = {arm: results[model]["arms"][arm]["binary_0.5"]["recall"]["estimate"] for arm in ARMS}
        print(model, " ".join(f"{arm}={value:.3f}" for arm, value in recalls.items()))
    if "decision" in results:
        print(f"Regra emendada no {RULE_MODEL}: afirmacao causal {'volta' if results['decision'] else 'nao volta'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("select", help="Sortear e gravar as 400 imagens.").set_defaults(func=_select)
    analyze = subparsers.add_parser("analyze", help="Tabelas e regra do E7.")
    analyze.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
