"""E5 (avaliadores) e auditoria do E4: analise das rotulacoes humanas (plano, secoes 10.6 e 10.7;
regras na 10.12, 20/10, fixadas antes de qualquer rotulacao existir).

  1. ``check --rater A|B --csv <arquivo>``: confere um CSV da ferramenta (``labeling_tool``):
     cada imagem de cada etapa uma vez e campos validos. Da chave, usa so os nomes anonimos e as
     partes, sem rotulo nem sigma, e nao calcula taxa nenhuma.
  2. ``analyze``: com os CSVs dos dois avaliadores em results_v4/e5/ratings/, grava
     results_v4/e5/human_results.json (parte A) e results_v4/e4/audit_results.json (parte B).

Parte A (E5). A decisao e o "sim", com qualquer confianca. Por avaliador, pela uniao (sim de um
dos dois) e pelo consenso (sim dos dois): recall por faixa de sigma da selecao e FPR nos 100
negativos (Wilson), com o E2B e o E4B (P0 do artigo a 0,5) e a CNN (0,5) nas mesmas imagens e o
McNemar exato contra cada um. Regra do C4: nos positivos de 5-15 sigma, o recall do avaliador
entra descontado da FPR dele, (r - f) / (1 - f), com f = limite superior do Wilson da FPR (o
desconto do controle fora do burst da CNN). O avaliador passa contra um VLM se o Delta
descontado for >= 10 pp, com McNemar p < 0,05 nas decisoes brutas; os humanos passam se os dois
avaliadores passarem. A CNN passa se a regra e o desconto fora do burst valerem nos dois
limiares. Desfecho por VLM: tese forte com as duas referencias; as duas relatadas, sem a tese
forte, com uma; a versao moderada com nenhuma. Descritivos: o ponto de operacao conservador (sim
com confianca 3), a AUC da escala de 6 niveis, o kappa de Cohen entre os avaliadores, o treino e
o tempo de resposta.

Parte B (auditoria do E4). Negativos com RFI visivel por avaliador, uniao e consenso (Wilson),
por tipo, e o kappa. O numero que a C6 cita e o do consenso. Descritivo: FP do E2B e do E4B (P0,
P1 e P2) nos negativos com RFI visivel no consenso e nos demais.

CSVs (10.12, 01/11). A versao 1 da ferramenta grava uma linha por resposta, sem volta. A versao 2
deixa o avaliador voltar dentro das partes A e B e grava uma linha por imagem, com a resposta
final, a primeira e o resumo das mudancas; as respostas guardadas pela versao 1 entram nela como
primeiras respostas da versao 1. Os dois formatos sao aceitos. Vale a resposta final. Descritivos:
quantas respostas mudaram; o desfecho do C4, o kappa e o numero da C6 com as primeiras respostas;
e as respostas da parte B confirmadas antes do esclarecimento da parte B (01/11), com a contagem
de consenso e o kappa so nas imagens que os dois confirmaram depois dele. O esclarecimento conta,
para cada avaliador, do horario anotado (``--clarified``; o do avaliador A esta em
``CLARIFIED_UTC``) ou, se for anterior, da primeira resposta na versao 2, que traz o guia.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from revision_v4.common import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, RESULTS_ROOT, read_jsonl, run_manifest, sha256_file, write_json
from revision_v4.e4_preprocessing import variant_output
from revision_v4.e5_cnn import C4_BINS, C4_MIN_DELTA, VLM_PREDICTIONS, c4_rule, cnn_scores, vlm_decisions
from revision_v4.labeling_tool import KEY_PATH, OUTPUT_DIR, RATERS, RFI_TYPES
from revision_v4.selections import STRATA, stratum_of
from revision_v4.stats import PairedCounts, mcnemar_exact, wilson


RATINGS_DIR = OUTPUT_DIR / "ratings"
LABELING_MANIFEST = OUTPUT_DIR / "labeling_manifest.json"
CNN_RESULTS = OUTPUT_DIR / "cnn_results.json"
CNN_OFFBURST = OUTPUT_DIR / "cnn_offburst_results.json"
AUDIT_DIR = RESULTS_ROOT / "e4"
COLUMNS = ("rater", "part", "image", "answer", "confidence", "rfi_type", "response_ms", "time_utc")
# versao 2 da ferramenta (01/11): uma linha por imagem, com a primeira resposta e as mudancas
COLUMNS_V2 = COLUMNS + (
    "first_answer", "first_confidence", "first_rfi_type", "first_time_utc", "changes", "confirmations", "total_ms", "first_tool",
)
# horario em que cada avaliador recebeu o esclarecimento da parte B (10.12, 01/11)
CLARIFIED_UTC = {"A": "2026-11-01T15:43:00Z"}
VLMS = tuple(VLM_PREDICTIONS)
CNN_THRESHOLDS = ("0.5", "vlm_fpr")
ALPHA = 0.05
# escala ordinal da parte A: nao com confianca 3, 2, 1 -> 1, 2, 3; sim com confianca 1, 2, 3 -> 4, 5, 6
LEVELS = 6


def ratings_path(rater: str, directory: Path = RATINGS_DIR) -> Path:
    return directory / f"rotulacao_avaliador_{rater}.csv"


def expected_images(key: list[dict[str, Any]], training_count: int) -> dict[str, set[str]]:
    """Imagens de cada etapa da ferramenta: treino, parte A e parte B."""
    images: dict[str, set[str]] = {"treino": {f"treino_{i:02d}" for i in range(training_count)}, "A": set(), "B": set()}
    for entry in key:
        images[entry["part"]].add(entry["anon_id"])
    return images


# ---------------------------------------------------------------------------
# leitura dos CSVs


@dataclass(frozen=True)
class Rating:
    """Resposta final de uma imagem; ``first`` e a primeira (sim, confianca, tipo) quando o CSV
    e da versao 2, ou None quando a primeira e a propria resposta."""

    yes: bool
    confidence: int | None
    rfi_type: str | None
    response_ms: int
    time_utc: str = ""
    first: tuple[bool, int | None, str | None] | None = None
    first_time_utc: str = ""
    changes: int = 0
    confirmations: int = 1
    total_ms: int | None = None
    first_tool: int = 1

    @property
    def level(self) -> int:
        """Posicao na escala de 6 niveis (so nas etapas com confianca)."""
        assert self.confidence is not None
        return 3 + self.confidence if self.yes else 4 - self.confidence

    def first_rating(self) -> Rating:
        """A primeira resposta, como se fosse a final (para os descritivos)."""
        if self.first is None:
            return self
        yes, confidence, rfi_type = self.first
        return replace(self, yes=yes, confidence=confidence, rfi_type=rfi_type, first=None, changes=0)

    @property
    def decision_changed(self) -> bool:
        return self.first is not None and self.first[0] != self.yes


def parse_time(value: str) -> datetime | None:
    """Horario ISO 8601 do CSV (o ``toISOString`` do navegador), ou None."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _answer_errors(part: str, answer: str, confidence: str, rfi_type: str, prefix: str = "") -> list[str]:
    errors = []
    if answer not in ("yes", "no"):
        errors.append(f"{prefix}resposta {answer!r}")
    if part == "B":
        types = {value for value, _ in RFI_TYPES}
        valid_type = rfi_type in types if answer == "yes" else rfi_type == ""
        if confidence != "" or not valid_type:
            errors.append(f"{prefix}confianca {confidence!r} ou tipo {rfi_type!r}")
    elif confidence not in ("1", "2", "3") or rfi_type != "":
        errors.append(f"{prefix}confianca {confidence!r} ou tipo {rfi_type!r}")
    return errors


def _row_errors(fields: dict[str, str], rater: str, expected: dict[str, set[str]]) -> list[str]:
    part = fields["part"]
    errors = []
    if fields["rater"] != rater:
        errors.append(f"avaliador {fields['rater']!r}")
    if part not in expected or fields["image"] not in expected[part]:
        return errors + [f"imagem {fields['image']!r} fora da etapa {part!r}"]
    errors += _answer_errors(part, fields["answer"], fields["confidence"], fields["rfi_type"])
    if not fields["response_ms"].isdigit():
        errors.append(f"tempo {fields['response_ms']!r}")
    if "first_answer" not in fields:  # versao 1
        return errors
    errors += _answer_errors(part, fields["first_answer"], fields["first_confidence"], fields["first_rfi_type"], "primeira ")
    numbers = {c: fields[c] for c in ("changes", "confirmations", "total_ms")}
    if not all(v.isdigit() for v in numbers.values()) or fields["first_tool"] not in ("1", "2"):
        return errors + [f"contagens {numbers} ou versao {fields['first_tool']!r}"]
    changes, confirmations, total_ms = (int(v) for v in numbers.values())
    if confirmations < 1 or changes > confirmations - 1:
        errors.append(f"{changes} mudancas em {confirmations} confirmacoes")
    if fields["response_ms"].isdigit() and total_ms < int(fields["response_ms"]):
        errors.append(f"tempo total {total_ms} menor que o da primeira resposta")
    first_time, time = parse_time(fields["first_time_utc"]), parse_time(fields["time_utc"])
    if first_time is None or time is None or first_time > time:
        errors.append(f"horarios {fields['first_time_utc']!r} e {fields['time_utc']!r}")
    same = all(fields[c] == fields[f"first_{c}"] for c in ("answer", "confidence", "rfi_type"))
    if changes == 0 and not same:
        errors.append("resposta diferente da primeira sem mudanca registrada")
    return errors


def _rating(fields: dict[str, str]) -> Rating:
    part = fields["part"]
    rating = Rating(
        yes=fields["answer"] == "yes",
        confidence=int(fields["confidence"]) if part != "B" else None,
        rfi_type=fields["rfi_type"] or None,
        response_ms=int(fields["response_ms"]),
        time_utc=fields["time_utc"],
        first_time_utc=fields["time_utc"],
    )
    if "first_answer" not in fields:
        return rating
    return replace(
        rating,
        first=(fields["first_answer"] == "yes", int(fields["first_confidence"]) if part != "B" else None,
               fields["first_rfi_type"] or None),
        first_time_utc=fields["first_time_utc"],
        changes=int(fields["changes"]),
        confirmations=int(fields["confirmations"]),
        total_ms=int(fields["total_ms"]),
        first_tool=int(fields["first_tool"]),
    )


def read_ratings(
    path: Path, rater: str, expected: dict[str, set[str]]
) -> tuple[dict[str, dict[str, Rating]], dict[str, Any]]:
    """Respostas de um avaliador por etapa e imagem, e o relatorio da conferencia.

    Cada imagem de cada etapa precisa aparecer. Se aparecer mais de uma vez, vale a primeira
    linha e a repeticao e contada (a versao 2 grava uma linha por imagem, com a resposta final).
    Qualquer outro defeito interrompe a analise. Aceita os dois formatos da ferramenta.
    """
    ratings: dict[str, dict[str, Rating]] = {part: {} for part in expected}
    repeated = {part: 0 for part in expected}
    problems: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, None) or ())
        if header not in (COLUMNS, COLUMNS_V2):
            raise ValueError(f"{path}: cabecalho {list(header)} diferente de {list(COLUMNS)} e de {list(COLUMNS_V2)}")
        for number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                problems.append(f"linha {number}: {len(row)} campos")
                continue
            fields = dict(zip(header, row))
            errors = _row_errors(fields, rater, expected)
            if errors:
                problems += [f"linha {number}: {error}" for error in errors]
                continue
            part, image = fields["part"], fields["image"]
            if image in ratings[part]:
                repeated[part] += 1
                continue
            ratings[part][image] = _rating(fields)
    for part, images in expected.items():
        missing = len(images - ratings[part].keys())
        if missing:
            problems.append(f"etapa {part}: faltam {missing} de {len(images)} imagens")
    if problems:
        shown = "; ".join(problems[:20]) + (f" (e mais {len(problems) - 20})" if len(problems) > 20 else "")
        raise ValueError(f"{path}: {shown}")
    report = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "format": 2 if header == COLUMNS_V2 else 1,
        "answers": {part: len(answers) for part, answers in ratings.items()},
        "repeated_ignored": repeated,
        "changed": {part: sum(r.changes > 0 for r in answers.values()) for part, answers in ratings.items()},
    }
    return ratings, report


# ---------------------------------------------------------------------------
# estatisticas


def cohen_kappa(first: np.ndarray, second: np.ndarray) -> float | None:
    """Kappa de Cohen de duas decisoes binarias; None se a concordancia esperada for 1."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    observed = float(np.mean(first == second))
    p, q = float(np.mean(first)), float(np.mean(second))
    expected = p * q + (1.0 - p) * (1.0 - q)
    return None if expected >= 1.0 - 1e-12 else (observed - expected) / (1.0 - expected)


def kappa_with_bootstrap(
    first: np.ndarray, second: np.ndarray, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Kappa com IC 95% por bootstrap percentil das imagens; reamostras sem kappa sao contadas."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        index = rng.integers(0, first.size, first.size)
        value = cohen_kappa(first[index], second[index])
        if value is not None:
            draws.append(value)
    return {
        "n": int(first.size),
        "agreement": float(np.mean(first == second)),
        "estimate": cohen_kappa(first, second),
        "low": float(np.percentile(draws, 2.5)) if draws else None,
        "high": float(np.percentile(draws, 97.5)) if draws else None,
        "failed_resamples": iterations - len(draws),
    }


def auc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> float:
    """P(score de um positivo > score de um negativo), com empates valendo 1/2."""
    x = np.asarray(positive_scores, dtype=float)[:, None]
    y = np.asarray(negative_scores, dtype=float)[None, :]
    return float(np.mean((x > y) + 0.5 * (x == y)))


def auc_with_bootstrap(
    scores: np.ndarray, positive: np.ndarray, *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED
) -> dict[str, float]:
    """AUC com IC 95% por bootstrap percentil, reamostrando positivos e negativos em separado."""
    positives, negatives = np.asarray(scores)[positive], np.asarray(scores)[~positive]
    rng = np.random.default_rng(seed)
    draws = np.array([
        auc(positives[rng.integers(0, positives.size, positives.size)], negatives[rng.integers(0, negatives.size, negatives.size)])
        for _ in range(iterations)
    ])
    return {"estimate": auc(positives, negatives), "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5))}


def discounted_rule(reference_hits: np.ndarray, vlm_hits: np.ndarray, false_positives: int, negatives: int) -> dict[str, Any]:
    """Regra do C4 para um leitor humano, com o recall descontado da FPR dele.

    f = limite superior do Wilson da FPR do leitor; recall descontado = (r - f) / (1 - f). Passa
    se o recall descontado menos o recall do VLM for >= 10 pp e o McNemar exato nas decisoes
    brutas der p < 0,05.
    """
    raw = c4_rule(list(reference_hits), list(vlm_hits))
    fpr = wilson(false_positives, negatives)
    recall, vlm_recall = float(np.mean(reference_hits)), float(np.mean(vlm_hits))
    discounted = (recall - fpr.high) / (1.0 - fpr.high) if fpr.high < 1.0 else None
    delta = None if discounted is None else discounted - vlm_recall
    return {
        "n": raw["n"],
        "recall": recall,
        "vlm_recall": vlm_recall,
        "delta_recall": raw["delta_recall"],
        "mcnemar": raw["mcnemar"],
        "passes_undiscounted": raw["passes"],
        "fpr": fpr.as_dict(),
        "fpr_used": fpr.high,
        "recall_discounted": discounted,
        "delta_discounted": delta,
        "passes": delta is not None and delta >= C4_MIN_DELTA and raw["mcnemar"]["p_value_exact"] < ALPHA,
    }


# ---------------------------------------------------------------------------
# parte A (E5)


def reader_decisions(ratings: dict[str, dict[str, Rating]], images: list[str]) -> dict[str, np.ndarray]:
    """Decisoes de cada leitor: os avaliadores, a uniao, o consenso e o ponto conservador de cada
    avaliador (sim com confianca 3)."""
    yes = {rater: np.array([ratings[rater][i].yes for i in images]) for rater in RATERS}
    readers = {**yes, "union": yes["A"] | yes["B"], "consensus": yes["A"] & yes["B"]}
    for rater in RATERS:
        readers[f"{rater}_confidence3"] = np.array([ratings[rater][i].yes and ratings[rater][i].confidence == 3 for i in images])
    return readers


def reader_summary(hits: np.ndarray, positive: np.ndarray, strata: np.ndarray, machines: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recall por faixa (com as referencias automaticas e o McNemar contra cada uma) e FPR."""
    by_stratum: dict[str, Any] = {}
    for low, _ in STRATA:
        label = stratum_of(low)
        mask = strata == label
        if not mask.any():
            continue
        n = int(mask.sum())
        by_stratum[label] = {
            "n": n,
            "recall": wilson(int(hits[mask].sum()), n).as_dict(),
            **{
                name: {
                    "recall": wilson(int(decisions[mask].sum()), n).as_dict(),
                    "mcnemar": mcnemar_exact(PairedCounts.from_outcomes(hits[mask], decisions[mask])),
                }
                for name, decisions in machines.items()
            },
        }
    negatives = ~positive
    false_positives = int(hits[negatives].sum())
    return {
        "recall_by_stratum": by_stratum,
        "false_positives": false_positives,
        "negatives": int(negatives.sum()),
        "fpr": wilson(false_positives, int(negatives.sum())).as_dict(),
        "accuracy": wilson(int(np.sum(hits == positive)), int(positive.size)).as_dict(),
    }


def analyze_part_a(
    key: list[dict[str, Any]],
    ratings: dict[str, dict[str, Rating]],
    machines: dict[str, dict[str, bool]],
    machine_scores: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """``ratings`` leva cada avaliador as respostas da parte A; ``machines``, cada referencia
    automatica (e2b, e4b, cnn) a decisao por sample_id; ``machine_scores``, ao score por
    sample_id (so para a AUC)."""
    entries = sorted((e for e in key if e["part"] == "A"), key=lambda e: e["anon_id"])
    images = [e["anon_id"] for e in entries]
    samples = [e["sample_id"] for e in entries]
    positive = np.array([e["true_label"] == "FRB" for e in entries])
    strata = np.array([stratum_of(float(e["peak_sigma"])) if p else "" for e, p in zip(entries, positive)])
    faint = np.isin(strata, C4_BINS)
    decisions = {name: np.array([by_sample[s] for s in samples]) for name, by_sample in machines.items()}
    readers = reader_decisions(ratings, images)

    result: dict[str, Any] = {
        "n_images": len(images),
        "n_positives": int(positive.sum()),
        "n_positives_5_to_15_sigma": int(faint.sum()),
        "readers": {},
    }
    for name, hits in readers.items():
        # so os avaliadores entram na regra; uniao, consenso e o ponto conservador sao descritivos
        entry = {"role": "rule" if name in RATERS else "descriptive", **reader_summary(hits, positive, strata, decisions)}
        # descritivo de 02/11 (secao 10.12, depois do resultado): o recall dos 5-15 sigma juntos
        entry["recall_5_to_15_sigma"] = wilson(int(hits[faint].sum()), int(faint.sum())).as_dict()
        entry["c4_5_to_15_sigma"] = {
            vlm: discounted_rule(hits[faint], decisions[vlm][faint], entry["false_positives"], entry["negatives"])
            for vlm in VLMS
        }
        versus_cnn = c4_rule(list(hits[faint]), list(decisions["cnn"][faint]))
        entry["versus_cnn_5_to_15_sigma"] = {"delta_recall": versus_cnn["delta_recall"], "mcnemar": versus_cnn["mcnemar"]}
        result["readers"][name] = entry
    result["humans_pass"] = {
        vlm: all(result["readers"][rater]["c4_5_to_15_sigma"][vlm]["passes"] for rater in RATERS) for vlm in VLMS
    }
    # descritivo de 02/11 (secao 10.12, depois do resultado): as referencias automaticas nas mesmas
    # imagens, para a tabela do artigo (recall dos 5-15 sigma juntos e FP nos mesmos negativos)
    result["machines_descriptive"] = {
        name: {
            "recall_5_to_15_sigma": wilson(int(hits[faint].sum()), int(faint.sum())).as_dict(),
            "false_positives": int(hits[~positive].sum()),
            "negatives": int((~positive).sum()),
            "fpr": wilson(int(hits[~positive].sum()), int((~positive).sum())).as_dict(),
        }
        for name, hits in decisions.items()
    }

    result["kappa"] = {
        "all": kappa_with_bootstrap(readers["A"], readers["B"]),
        "positives": kappa_with_bootstrap(readers["A"][positive], readers["B"][positive]),
    }
    levels = {rater: np.array([ratings[rater][i].level for i in images]) for rater in RATERS}
    scores = {name: np.array([by_sample[s] for s in samples]) for name, by_sample in machine_scores.items()}
    result["auc_descriptive"] = {}
    for subset, members in (("all_positives", positive), ("positives_5_to_15_sigma", faint)):
        keep = members | ~positive
        result["auc_descriptive"][subset] = {
            **{rater: auc_with_bootstrap(levels[rater][keep], positive[keep]) for rater in RATERS},
            **{name: auc(values[members], values[~positive]) for name, values in scores.items()},
        }
    return result


def cnn_reference(cnn_results: dict[str, Any], offburst_results: dict[str, Any]) -> dict[str, Any]:
    """A CNN passa contra um VLM se a regra do C4 e o desconto fora do burst valerem nos dois
    limiares (0,5 e o casado com a FPR do VLM)."""
    reference = {}
    for vlm in VLMS:
        rule = {t: cnn_results[vlm][t]["c4_rule_5_to_15_sigma"]["passes"] for t in CNN_THRESHOLDS}
        holds = {t: offburst_results[vlm][t]["c4_discounted"]["holds"] for t in CNN_THRESHOLDS}
        reference[vlm] = {"rule": rule, "offburst_discount": holds, "passes": all(rule.values()) and all(holds.values())}
    return reference


def c4_outcome(cnn_passes: bool, humans_pass: bool) -> str:
    """Secao 10.7, com a leitura de 20/10 (secao 10.12)."""
    if cnn_passes and humans_pass:
        return "strong_thesis"
    if cnn_passes or humans_pass:
        return "both_reported_without_strong_thesis"
    return "moderate_thesis"


def c4_decision(cnn: dict[str, bool], humans: dict[str, bool]) -> dict[str, Any]:
    """Desfecho por VLM; se os dois VLMs diferirem, o texto fala de cada um, e o abstract so
    leva a tese forte se ela valer nos dois."""
    by_vlm = {vlm: c4_outcome(cnn[vlm], humans[vlm]) for vlm in VLMS}
    outcomes = set(by_vlm.values())
    return {"by_vlm": by_vlm, "text": outcomes.pop() if len(outcomes) == 1 else "per_model",
            "abstract_strong_thesis": all(o == "strong_thesis" for o in by_vlm.values())}


# ---------------------------------------------------------------------------
# parte B (auditoria do E4)


def analyze_part_b(
    key: list[dict[str, Any]], ratings: dict[str, dict[str, Rating]], vlm_runs: dict[str, dict[str, dict[str, bool]]]
) -> dict[str, Any]:
    """``ratings`` leva cada avaliador as respostas da parte B; ``vlm_runs``, cada modelo e
    variante (P0, P1, P2) a decisao FRB por sample_id (nos negativos, um FP)."""
    entries = sorted((e for e in key if e["part"] == "B"), key=lambda e: e["anon_id"])
    images = [e["anon_id"] for e in entries]
    samples = [e["sample_id"] for e in entries]
    yes = {rater: np.array([ratings[rater][i].yes for i in images]) for rater in RATERS}
    readers = {**yes, "union": yes["A"] | yes["B"], "consensus": yes["A"] & yes["B"]}
    n = len(images)
    visible = {name: {"count": int(hits.sum()), **wilson(int(hits.sum()), n).as_dict()} for name, hits in readers.items()}
    type_names = [value for value, _ in RFI_TYPES]
    consensus_types = dict.fromkeys(type_names + ["discordant"], 0)
    for image, both in zip(images, readers["consensus"]):
        if both:
            first, second = ratings["A"][image].rfi_type, ratings["B"][image].rfi_type
            consensus_types[first if first == second else "discordant"] += 1
    false_positives = {}
    for model, variants in vlm_runs.items():
        false_positives[model] = {}
        for variant, flagged in variants.items():
            hits = np.array([flagged[s] for s in samples])
            false_positives[model][variant] = {
                "all": int(hits.sum()),
                "consensus_visible_rfi": int(hits[readers["consensus"]].sum()),
                "others": int(hits[~readers["consensus"]].sum()),
            }
    return {
        "n": n,
        "visible_rfi": visible,
        # o numero que a C6 cita (secao 10.12, 20/10)
        "c6_number": {"reader": "consensus", **visible["consensus"]},
        "types_by_rater": {
            rater: {value: sum(ratings[rater][i].rfi_type == value for i in images) for value in type_names} for rater in RATERS
        },
        "types_consensus": consensus_types,
        "kappa": kappa_with_bootstrap(readers["A"], readers["B"]),
        "vlm_false_positives_descriptive": false_positives,
    }


# ---------------------------------------------------------------------------
# treino e tempo de resposta (descritivos)


def training_summary(ratings: dict[str, Rating], answers: list[str]) -> dict[str, int]:
    correct = sum(ratings[f"treino_{i:02d}"].yes == (answer == "FRB") for i, answer in enumerate(answers))
    return {"correct": correct, "n": len(answers)}


def response_times(ratings: dict[str, dict[str, Rating]]) -> dict[str, dict[str, float]]:
    """Mediana e quartis do tempo da primeira resposta por etapa, em segundos, e o tempo total
    com as revisitas (versao 2)."""
    result = {}
    for part, answers in ratings.items():
        seconds = np.array([r.response_ms for r in answers.values()]) / 1000.0
        total = np.array([r.response_ms if r.total_ms is None else r.total_ms for r in answers.values()]) / 1000.0
        q1, median, q3 = np.percentile(seconds, [25, 50, 75])
        result[part] = {"median_s": float(median), "q1_s": float(q1), "q3_s": float(q3), "total_min": float(seconds.sum() / 60.0),
                        "total_with_revisits_min": float(total.sum() / 60.0)}
    return result


# ---------------------------------------------------------------------------
# versao 2 da ferramenta: mudancas, primeiras respostas e o esclarecimento da parte B (descritivos)


def revision_summary(ratings: dict[str, dict[str, Rating]]) -> dict[str, dict[str, Any]]:
    """Por etapa: respostas que mudaram depois da primeira, decisoes (sim/nao) que mudaram,
    confirmacoes e a versao da ferramenta de cada primeira resposta."""
    summary = {}
    for part, answers in ratings.items():
        values = list(answers.values())
        summary[part] = {
            "n": len(values),
            "changed": sum(r.changes > 0 for r in values),
            "decision_changed": sum(r.decision_changed for r in values),
            "confirmations": sum(r.confirmations for r in values),
            "first_tool": {str(v): sum(r.first_tool == v for r in values) for v in (1, 2)},
        }
    return summary


def first_answers(ratings: dict[str, dict[str, Rating]]) -> dict[str, dict[str, Rating]]:
    return {part: {image: r.first_rating() for image, r in answers.items()} for part, answers in ratings.items()}


def clarified_at(ratings: dict[str, dict[str, Rating]], noted: str | None) -> datetime | None:
    """Quando o avaliador recebeu o esclarecimento da parte B: o horario anotado ou, se for
    anterior, a primeira resposta dada na versao 2 da ferramenta, que traz o guia. None se nao
    houver nenhum dos dois."""
    candidates = [parse_time(r.first_time_utc) for answers in ratings.values() for r in answers.values() if r.first_tool == 2]
    candidates = [c for c in candidates if c is not None]
    if noted is not None:
        when = parse_time(noted)
        if when is None:
            raise ValueError(f"horario do esclarecimento {noted!r}")
        candidates.append(when)
    return min(candidates) if candidates else None


def clarification_summary(
    key: list[dict[str, Any]], ratings: dict[str, dict[str, Rating]], times: dict[str, datetime | None]
) -> dict[str, Any]:
    """Respostas da parte B cuja ultima confirmacao veio antes do esclarecimento (sem horario
    legivel contam como antes) e, nas imagens que os dois avaliadores confirmaram depois dele, a
    contagem de consenso e o kappa."""
    images = sorted(e["anon_id"] for e in key if e["part"] == "B")

    def after(rater: str, image: str) -> bool:
        when, time = times[rater], parse_time(ratings[rater][image].time_utc)
        return when is not None and time is not None and time >= when

    later = {rater: np.array([after(rater, i) for i in images], dtype=bool) for rater in RATERS}
    both = later["A"] & later["B"]
    yes = {rater: np.array([ratings[rater][i].yes for i in images], dtype=bool) for rater in RATERS}
    consensus = int((yes["A"] & yes["B"])[both].sum())
    n = int(both.sum())
    return {
        "clarified_utc": {rater: times[rater].isoformat() if times[rater] else None for rater in RATERS},
        "answers_before": {rater: int((~later[rater]).sum()) for rater in RATERS},
        "images_both_after": n,
        "consensus_after": {"count": consensus, **wilson(consensus, n).as_dict()} if n else None,
        "kappa_after": kappa_with_bootstrap(yes["A"][both], yes["B"][both]) if n else None,
    }


def parse_clarified(values: list[str]) -> dict[str, str]:
    """``--clarified B=2026-11-02T10:00:00Z`` -> {"B": "2026-11-02T10:00:00Z"}."""
    noted = {}
    for value in values:
        rater, _, when = value.partition("=")
        if rater not in RATERS or parse_time(when) is None:
            raise SystemExit(f"--clarified {value!r}: use AVALIADOR=horario ISO 8601 com fuso, como B=2026-11-02T10:00:00Z")
        noted[rater] = when
    return noted


# ---------------------------------------------------------------------------
# comandos


def _labeling_manifest() -> dict[str, Any]:
    return json.loads(LABELING_MANIFEST.read_text(encoding="utf-8"))


def _check(args: argparse.Namespace) -> None:
    key = [{"anon_id": e["anon_id"], "part": e["part"]} for e in read_jsonl(KEY_PATH)]
    expected = expected_images(key, len(_labeling_manifest()["training"]))
    try:
        _, report = read_ratings(args.csv, args.rater, expected)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(f"{args.csv}: ok; formato v{report['format']}; respostas {report['answers']}; "
          f"repeticoes ignoradas {report['repeated_ignored']}; respostas mudadas {report['changed']}")


def _analyze(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    manifest = _labeling_manifest()
    key_sha = sha256_file(KEY_PATH)
    if key_sha != manifest["key"]["sha256"]:
        raise SystemExit("A chave mudou depois da montagem da ferramenta (labeling_manifest.json).")
    key = read_jsonl(KEY_PATH)
    expected = expected_images(key, len(manifest["training"]))
    missing = [p.as_posix() for p in (ratings_path(r, args.ratings_dir) for r in RATERS) if not p.exists()]
    if missing:
        raise SystemExit(f"A analise precisa dos CSVs dos dois avaliadores; faltam {missing}.")
    ratings, reports = {}, {}
    for rater in RATERS:
        try:
            ratings[rater], reports[rater] = read_ratings(ratings_path(rater, args.ratings_dir), rater, expected)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    answers = [t["answer"] for t in manifest["training"]]
    descriptive = {
        rater: {"training": training_summary(ratings[rater]["treino"], answers), "response_times": response_times(ratings[rater]),
                "revisions": revision_summary(ratings[rater])}
        for rater in RATERS
    }
    noted = {**CLARIFIED_UTC, **parse_clarified(args.clarified)}
    try:
        clarified = {rater: clarified_at(ratings[rater], noted.get(rater)) for rater in RATERS}
    except ValueError as error:
        raise SystemExit(str(error)) from None

    test, _ = cnn_scores()
    machines: dict[str, dict[str, bool]] = {vlm: vlm_decisions(path)[0] for vlm, path in VLM_PREDICTIONS.items()}
    machines["cnn"] = {i: r["cnn_probability"] >= 0.5 for i, r in test.items()}
    machine_scores = {
        vlm: {r["sample_id"]: float(r["parsed_response"]["frb_probability"]) for r in read_jsonl(path)}
        for vlm, path in VLM_PREDICTIONS.items()
    }
    machine_scores["cnn"] = {i: float(r["cnn_probability"]) for i, r in test.items()}
    part_a = analyze_part_a(key, {rater: ratings[rater]["A"] for rater in RATERS}, machines, machine_scores)
    cnn = cnn_reference(
        json.loads(CNN_RESULTS.read_text(encoding="utf-8"))["results"],
        json.loads(CNN_OFFBURST.read_text(encoding="utf-8"))["results"],
    )
    decision = c4_decision({vlm: cnn[vlm]["passes"] for vlm in VLMS}, part_a["humans_pass"])

    # o P0 do E4 e a rodada do artigo
    vlm_runs = {
        model: {"P0": machines[model], **{v: vlm_decisions(variant_output(model, v))[0] for v in ("P1", "P2")}}
        for model in VLMS
    }
    part_b = analyze_part_b(key, {rater: ratings[rater]["B"] for rater in RATERS}, vlm_runs)

    # descritivos da versao 2: as primeiras respostas no lugar das finais, e o esclarecimento
    first = {rater: first_answers(ratings[rater]) for rater in RATERS}
    part_a_first = analyze_part_a(key, {rater: first[rater]["A"] for rater in RATERS}, machines, machine_scores)
    part_b_first = analyze_part_b(key, {rater: first[rater]["B"] for rater in RATERS}, vlm_runs)
    sensitivity_a = {
        "humans_pass": part_a_first["humans_pass"],
        "c4_rule": {rater: part_a_first["readers"][rater]["c4_5_to_15_sigma"] for rater in RATERS},
        "kappa": part_a_first["kappa"],
        "c4_decision": c4_decision({vlm: cnn[vlm]["passes"] for vlm in VLMS}, part_a_first["humans_pass"]),
    }
    sensitivity_b = {key_: part_b_first[key_] for key_ in ("visible_rfi", "c6_number", "kappa")}
    clarification = clarification_summary(key, {rater: ratings[rater]["B"] for rater in RATERS}, clarified)

    inputs = {
        "key": {"path": KEY_PATH.as_posix(), "sha256": key_sha},
        "ratings": reports,
        "cnn_results": {"path": CNN_RESULTS.as_posix(), "sha256": sha256_file(CNN_RESULTS)},
        "cnn_offburst": {"path": CNN_OFFBURST.as_posix(), "sha256": sha256_file(CNN_OFFBURST)},
        "vlm_predictions": {vlm: {"path": p.as_posix(), "sha256": sha256_file(p)} for vlm, p in VLM_PREDICTIONS.items()},
    }
    bootstrap = {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED}
    e5_dir = args.output_dir or OUTPUT_DIR
    e4_dir = args.output_dir or AUDIT_DIR
    write_json(
        e5_dir / "human_results.json",
        run_manifest(experiment="E5-humans-analysis", project_root=project_root, inputs=inputs, bootstrap=bootstrap,
                     descriptive=descriptive, part_a=part_a, cnn_reference=cnn, c4_decision=decision,
                     first_answers_descriptive=sensitivity_a),
    )
    write_json(
        e4_dir / "audit_results.json",
        run_manifest(experiment="E4-audit-analysis", project_root=project_root, inputs=inputs, bootstrap=bootstrap,
                     part_b=part_b, first_answers_descriptive=sensitivity_b, clarification_descriptive=clarification),
    )
    for rater in RATERS:
        for vlm in VLMS:
            rule = part_a["readers"][rater]["c4_5_to_15_sigma"][vlm]
            print(f"avaliador {rater} x {vlm}: delta descontado {rule['delta_discounted']}, "
                  f"p = {rule['mcnemar']['p_value_exact']:.3g}, passa = {rule['passes']}")
    print(f"C4: {decision}")
    c6 = part_b["c6_number"]
    print(f"C6: {c6['count']} de {part_b['n']} negativos com RFI visivel no consenso")
    print(f"primeiras respostas (descritivo): C4 {sensitivity_a['c4_decision']['by_vlm']}; "
          f"C6 {sensitivity_b['c6_number']['count']} de {part_b['n']}")
    print(f"esclarecimento (descritivo): respostas da parte B antes dele {clarification['answers_before']}; "
          f"imagens confirmadas depois pelos dois {clarification['images_both_after']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Conferir um CSV de um avaliador.")
    check.add_argument("--rater", choices=RATERS, required=True)
    check.add_argument("--csv", type=Path, required=True)
    check.set_defaults(func=_check)
    analyze = subparsers.add_parser("analyze", help="Partes A e B, com os CSVs dos dois avaliadores.")
    analyze.add_argument("--ratings-dir", type=Path, default=RATINGS_DIR)
    analyze.add_argument("--output-dir", type=Path, default=None,
                         help="Grava os dois JSONs aqui (o padrao e results_v4/e5 e results_v4/e4).")
    analyze.add_argument("--clarified", action="append", default=[], metavar="AVALIADOR=HORARIO",
                         help="Horario em que o avaliador recebeu o esclarecimento da parte B (o do A ja esta no codigo).")
    analyze.set_defaults(func=_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
