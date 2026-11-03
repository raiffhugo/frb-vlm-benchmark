"""E2: analise da injecao-recuperacao (plano, secao 10.4; definicoes na 10.12).

Le as rodadas results_v4/e2/<modelo>_binary_full e o manifesto de dataset_real_e2 e grava
e2_results.json com, por modelo:
  - recall por bin do sigma medido (Wilson), no total e por nivel de cada fator (largura, banda,
    espalhamento); o recall por sigma alvo, so descritivo;
  - sigma de 50% de completeza: logistico por maxima verossimilhanca em log10 do sigma medido,
    sigma50 = 10^(-a/b), com IC 95% por bootstrap percentil das injecoes (10 000 reamostras,
    seed 42); reamostras sem inclinacao positiva finita ficam fora do IC e sao contadas;
  - FPR nos 1024 controles (Wilson);
  - E2 - real por bin (Newcombe, amostras independentes), contra o P0 do artigo nos 600 bursts
    reais, nos bins com >= 10 bursts nos dois.
Deteccao com frb_probability >= 0,5; uma resposta que falhou vira a decisao errada.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from revision_v4.common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    RESULTS_ROOT,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
)
from revision_v4.paired_runs import load, read_visibility, recall_by_bin, summary, unpaired_by_bin
from revision_v4.stats import wilson


OUTPUT_DIR = RESULTS_ROOT / "e2"
MANIFEST = Path("dataset_real_e2/metadata/image_manifest.jsonl")
REAL_RUNS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
REAL_VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
FACTORS = ("width_ms", "band", "scattering")
MIN_BURSTS_PER_BIN = 10


def output(model: str) -> Path:
    return OUTPUT_DIR / f"{model}_binary_full" / "predictions.jsonl"


def fit_logistic(x: np.ndarray, y: np.ndarray, *, max_iter: int = 100) -> tuple[float, float] | None:
    """MLE de P(y) = 1 / (1 + exp(-(a + b x))) por Newton; None sem convergencia ou com b <= 0."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        p = expit(design @ beta)
        hessian = design.T @ (design * (p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(hessian, design.T @ (y - p))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)) or np.abs(beta).max() > 1e6:
            return None
        if np.abs(step).max() < 1e-10:
            break
    else:
        return None
    a, b = float(beta[0]), float(beta[1])
    return (a, b) if b > 0 else None


def sigma50(sigma: np.ndarray, detected: np.ndarray) -> float | None:
    fit = fit_logistic(np.log10(sigma), detected.astype(float))
    return None if fit is None else float(10 ** (-fit[0] / fit[1]))


def sigma50_with_bootstrap(
    sigma: np.ndarray, detected: np.ndarray, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        index = rng.integers(0, sigma.size, sigma.size)
        value = sigma50(sigma[index], detected[index])
        if value is not None:
            estimates.append(value)
    point = sigma50(sigma, detected)
    fit = fit_logistic(np.log10(sigma), detected.astype(float))
    return {
        "n": int(sigma.size),
        "sigma50": point,
        "logistic_a_b": fit,
        "low": float(np.percentile(estimates, 2.5)) if estimates else None,
        "high": float(np.percentile(estimates, 97.5)) if estimates else None,
        "failed_resamples": iterations - len(estimates),
        "iterations": iterations,
    }


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    manifest = {r["sample_id"]: r for r in read_jsonl(MANIFEST)}
    sigma_of = {sid: float(r["measured_sigma"]) for sid, r in manifest.items() if r["role"] == "injected"}
    real_sigma = {key: float(row["peak_sigma"]) for key, row in read_visibility(REAL_VISIBILITY).items()}
    results: dict[str, Any] = {}
    inputs: dict[str, Any] = {
        "manifest": {"path": MANIFEST.as_posix(), "sha256": sha256_file(MANIFEST)},
        "real_visibility": {"path": REAL_VISIBILITY.as_posix(), "sha256": sha256_file(REAL_VISIBILITY)},
    }
    for model in args.models:
        run = load(output(model))
        injected = run.y_true
        detected = run.predict(0.5)
        factors = {factor: np.array([manifest[s]["injection"][factor] for s in run.sample_ids]) for factor in FACTORS}
        target = np.array([manifest[s]["injection"]["target_sigma"] for s in run.sample_ids])
        measured = np.array([sigma_of.get(s, np.nan) for s in run.sample_ids])
        controls = ~injected
        flagged_control = {
            manifest[s]["injection_id"] for s, hit in zip(run.sample_ids, detected) if manifest[s]["role"] == "control" and hit
        }
        injection_ids = np.array([manifest[s]["injection_id"] for s in run.sample_ids])
        with_flagged_control = injected & np.isin(injection_ids, sorted(flagged_control))
        real_by_bin = recall_by_bin(load(REAL_RUNS[model]), real_sigma)
        e2_by_bin = recall_by_bin(run, sigma_of)
        results[model] = {
            **summary(run),
            "control_fpr": wilson(int(detected[controls].sum()), int(controls.sum())).as_dict(),
            "injections_whose_control_was_flagged": {
                "n": int(with_flagged_control.sum()),
                "detected": int(detected[with_flagged_control].sum()),
            },
            "recall_by_measured_sigma_bin": e2_by_bin,
            "recall_by_factor_and_bin": {
                factor: {
                    str(level): recall_by_bin(run, sigma_of, factors[factor] == level)
                    for level in sorted(set(factors[factor][injected]), key=str)
                }
                for factor in FACTORS
            },
            "recall_by_target_sigma_descriptive": {
                str(level): wilson(int(detected[injected & (target == level)].sum()), int((injected & (target == level)).sum())).as_dict()
                for level in sorted(set(target[injected]))
            },
            "sigma50": sigma50_with_bootstrap(measured[injected], detected[injected]),
            "sigma50_by_factor": {
                factor: {
                    str(level): sigma50_with_bootstrap(
                        measured[injected & (factors[factor] == level)], detected[injected & (factors[factor] == level)]
                    )
                    for level in sorted(set(factors[factor][injected]), key=str)
                }
                for factor in FACTORS
            },
            "real_recall_by_sigma_bin": real_by_bin,
            "e2_minus_real_by_bin": unpaired_by_bin(e2_by_bin, real_by_bin, MIN_BURSTS_PER_BIN),
        }
        inputs[model] = {
            name: {"path": path.as_posix(), "sha256": sha256_file(path)}
            for name, path in (("e2", output(model)), ("real", REAL_RUNS[model]))
        }
    write_json(
        OUTPUT_DIR / ("e2_results.json" if len(args.models) == len(REAL_RUNS) else f"e2_results_{'_'.join(args.models)}.json"),
        run_manifest(
            experiment="E2-analysis", project_root=project_root, inputs=inputs,
            bootstrap={"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED},
            min_bursts_per_bin=MIN_BURSTS_PER_BIN, results=results,
        ),
    )
    for model, entry in results.items():
        s50 = entry["sigma50"]
        print(f"{model}: sigma50 {s50['sigma50']} [{s50['low']}; {s50['high']}], FPR controles {entry['control_fpr']['estimate']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="Tabelas do E2.")
    analyze.add_argument("--models", nargs="+", choices=sorted(REAL_RUNS), default=sorted(REAL_RUNS))
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
