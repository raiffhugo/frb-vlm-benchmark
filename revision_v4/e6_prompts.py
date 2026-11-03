"""Prompts do E6 (plano, secao 10.8), derivados do prompt binario do artigo (P0).

P-b, P-d, P-e e R saem do P0 por recortes e trocas exatas, para que a unica diferenca seja a
descrita; P-a e P-c sao reescritas. O esquema JSON e o mesmo do P0 em todas, exceto no P-c
(compactado numa linha, com as mesmas chaves e os mesmos valores) e no P-d (sem as quatro
features que nomeiam familias de RFI). O P-c mantem todas as regras de decisao e todas as
faixas de calibracao, e tira as definicoes das features auxiliares (os nomes ficam no
esquema), que a decisao nao usa.

  - P-a: mesmas regras, redacao nova.
  - P-b: as mesmas regras, com a lista de NON_FRB primeiro.
  - P-c: ~50% dos tokens do P0, mesmas regras.
  - P-d: NON_FRB generico, sem a lista de familias de RFI (nem nas faixas de calibracao e nas
    features).
  - P-e: sem a clausula Low-S/N.
  - R: morfologia relaxada; aceita emissao de banda limitada e varios componentes com a mesma
    dispersao, e continua rejeitando impulso de DM zero e estrutura persistente.

``write`` grava os arquivos em prompts_v4/ e registra os hashes; os testes conferem que os
arquivos sao a saida de ``build``. Depois do congelamento nada aqui muda (prompts_v4/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROMPTS_DIR = Path("prompts_v4")
P0_FILE = PROMPTS_DIR / "binary_full.txt"
FROZEN = "2026-10-19"

_RULE_HEADS = (
    "1. FRB candidates\n",
    "2. Low-S/N sensitivity\n",
    "3. NON_FRB cases\n",
    "4. Probability calibration\n",
)

_SCHEMA_START = "{\n  \"label\""


def _split_rules(p0: str) -> tuple[str, list[str]]:
    """Cabecalho e corpo das quatro regras; o corpo da 4a vai ate o fim do prompt."""
    positions = [p0.index("\n\n" + head) + 2 for head in _RULE_HEADS]
    header = p0[: positions[0]]
    bodies = []
    for index, head in enumerate(_RULE_HEADS):
        start = positions[index] + len(head)
        stop = positions[index + 1] - 2 if index + 1 < len(positions) else len(p0)
        bodies.append(p0[start:stop])
    return header, bodies


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Trecho deveria aparecer uma vez: {old[:60]!r}")
    return text.replace(old, new)


def _join(header: str, rules: list[tuple[str, str]]) -> str:
    parts = [f"{number}. {title}\n{body}" for number, (title, body) in enumerate(rules, start=1)]
    return header + "\n\n".join(parts)


def build_pb(p0: str) -> str:
    header, (frb, low_snr, non_frb, calibration) = _split_rules(p0)
    return _join(
        header,
        [
            ("NON_FRB cases", non_frb),
            ("FRB candidates", frb),
            ("Low-S/N sensitivity", low_snr),
            ("Probability calibration", calibration),
        ],
    )


def build_pe(p0: str) -> str:
    header, (frb, _low_snr, non_frb, calibration) = _split_rules(p0)
    return _join(
        header,
        [
            ("FRB candidates", frb),
            ("NON_FRB cases", non_frb),
            ("Probability calibration", calibration),
        ],
    )


_P0_NON_FRB_LIST = """Label as NON_FRB when the image is dominated by:
- pure random background with no coherent transient,
- persistent horizontal frequency bands,
- persistent vertical time bands,
- periodic stripes or grid-like structure,
- zero-DM broadband vertical impulse with no dispersion delay,
- saturation, clipping, blank regions, uniform panels, or normalization artifacts,
- two or more temporally separated broadband pulses or repeating patterns."""

_FAMILY_FEATURES = (
    "persistent_bands",
    "repeating_pattern",
    "uniform_or_constant_background",
    "saturated_or_clipped",
)


def build_pd(p0: str) -> str:
    text = _replace_once(
        p0, _P0_NON_FRB_LIST, "Label as NON_FRB when the image does not contain such a transient."
    )
    text = _replace_once(
        text,
        "- 0.05-0.17: clear RFI, persistent bands, artifacts, or strong NON_FRB evidence.",
        "- 0.05-0.17: strong NON_FRB evidence.",
    )
    text = _replace_once(
        text,
        "- 0.01-0.04: blank, saturated, clipped, or unambiguous artifact.",
        "- 0.01-0.04: unambiguous NON_FRB.",
    )
    lines = text.split("\n")
    kept = [
        line
        for line in lines
        if not any(line.lstrip().startswith((f'"{name}"', f"- {name}:")) for name in _FAMILY_FEATURES)
    ]
    if len(lines) - len(kept) != 2 * len(_FAMILY_FEATURES):
        raise ValueError("Esperava tirar cada feature de familia do esquema e das definicoes.")
    return "\n".join(kept)


def build_r(p0: str) -> str:
    text = _replace_once(
        p0,
        "Label as FRB only when the dominant signal is a SINGLE localized transient that is "
        "plausibly broadband and shows a frequency-dependent arrival-time trend, usually later "
        "arrival at lower frequency.",
        "Label as FRB only when the dominant signal is a localized transient that shows a "
        "frequency-dependent arrival-time trend, usually later arrival at lower frequency. It may "
        "span the full band or only part of it, and it may consist of two or more components "
        "close in time that follow the same dispersion sweep.",
    )
    text = _replace_once(
        text,
        "- zero-DM broadband vertical impulse with no dispersion delay,\n"
        "- saturation, clipping, blank regions, uniform panels, or normalization artifacts,\n"
        "- two or more temporally separated broadband pulses or repeating patterns.",
        "- one or more zero-DM broadband vertical impulses with no dispersion delay,\n"
        "- saturation, clipping, blank regions, uniform panels, or normalization artifacts.",
    )
    text = _replace_once(
        text,
        "- 0.95-0.99: textbook FRB, clear broadband dispersed sweep.",
        "- 0.95-0.99: textbook FRB, clear dispersed sweep.",
    )
    # a definicao do P0 ("multiple pulses") contradiria os varios componentes aceitos acima
    return _replace_once(
        text,
        "- repeating_pattern: multiple pulses, stripes, grids, or periodic structures.",
        "- repeating_pattern: periodic stripes, grids, or regularly repeating structures.",
    )


def _schema(p0: str) -> str:
    start = p0.index(_SCHEMA_START)
    return p0[start : p0.index("\n}\n", start) + 2]


PA_TEMPLATE = """You are a radio astronomer inspecting dynamic-spectrum plots produced by a search for radio transients.

How to read the plot:
- horizontal axis: time
- vertical axis: frequency
- color: received power

Assign the plot to exactly one class:
- FRB
- NON_FRB

Your answer must be valid JSON only: no markdown, no comments and no keys beyond the schema below.

Rules:

1. When to call it an FRB
Choose FRB only if the main feature is ONE short, localized burst that plausibly covers a wide range of frequencies and whose arrival time depends on frequency, typically arriving later at lower frequencies. The track can be curved or almost straight, and it can be weak or partly hidden by masking.

2. Faint signals
Do not dismiss weak but coherent structure as noise. If you see any plausible weak diagonal or curved track, a localized brightening across many frequencies, a fragment of a dispersed track, or extra variance along a slanted band, give an intermediate probability rather than a very low one.

3. When to call it NON_FRB
Choose NON_FRB when the plot is dominated by any of the following:
- random background only, with no coherent burst;
- horizontal bands at fixed frequencies that persist in time;
- vertical bands at fixed times that persist across the panel;
- periodic stripes or a grid-like pattern;
- a vertical impulse covering many frequencies at once, with no dispersion delay (zero DM);
- saturation, clipping, empty regions, flat panels, or normalization artifacts;
- two or more broadband pulses separated in time, or any repeating pattern.

4. Calibrating the probability
frb_probability is a number between 0 and 1.
Spread your values over the whole range instead of repeating the same few numbers.

Reference scale:
- 0.95-0.99: unmistakable FRB, a clear dispersed sweep across the band.
- 0.80-0.94: likely FRB, with small doubts.
- 0.60-0.79: possible FRB that is weak, incomplete, or noisy.
- 0.45-0.59: truly uncertain.
- 0.30-0.44: possible weak structure, but NON_FRB is more likely.
- 0.18-0.29: probably noise or mild contamination.
- 0.05-0.17: obvious RFI, persistent bands, artifacts, or strong NON_FRB evidence.
- 0.01-0.04: empty, saturated, clipped, or an obvious artifact.

Keep label and probability consistent:
- label = "FRB" exactly when frb_probability >= 0.50
- label = "NON_FRB" exactly when frb_probability < 0.50

confidence expresses how sure you are of the label, not the probability of an FRB.

Answer with exactly this JSON schema:

{schema}
What each feature means:
- visible_structure: some coherent structure can be seen.
- broadband: the signal covers a large part of the frequency range.
- localized_transient: the signal is limited to a short stretch of time.
- frequency_dependent_delay: the arrival time shifts steadily with frequency.
- persistent_bands: horizontal or vertical bands extend over most of the panel.
- repeating_pattern: several pulses, stripes, grids, or periodic features.
- uniform_or_constant_background: the panel is almost empty or flat.
- saturated_or_clipped: the plot shows saturation, clipping, or rectangular masked areas.
- random_background_texture: random noise dominates and no coherent candidate is present.

Write the reason as a single short sentence.
"""

PC_TEMPLATE = """You are an expert radio astronomer classifying dynamic-spectrum images (x: time, y: frequency, intensity: power) from a transient search as FRB or NON_FRB. Return ONLY valid JSON, no markdown, comments or extra keys.

FRB: the dominant signal is a SINGLE localized, plausibly broadband transient whose arrival time depends on frequency, usually later at lower frequency; the sweep may be curved, nearly linear, faint or partially masked.
Low S/N: faint coherent structure is not pure noise; a plausible faint diagonal/curved sweep, localized broadband brightening, partial dispersed track or excess variance along a tilted band gets an intermediate probability, not a very low one.
NON_FRB: dominated by random background, persistent horizontal or vertical bands, periodic stripes or grids, a zero-DM broadband vertical impulse, saturation, clipping, blank regions, uniform panels, normalization artifacts, or two or more separated broadband pulses or repeating patterns.

frb_probability in [0,1], full range: 0.95-0.99 textbook dispersed sweep; 0.80-0.94 probable; 0.60-0.79 plausible but faint, partial or noisy; 0.45-0.59 ambiguous; 0.30-0.44 weak structure, likely NON_FRB; 0.18-0.29 likely noise; 0.05-0.17 clear RFI, bands or artifacts; 0.01-0.04 blank, saturated, clipped or unambiguous artifact. label = "FRB" if frb_probability >= 0.50, else "NON_FRB"; confidence = certainty about the label, not the FRB probability.

Schema (reason: one short sentence; features: true or false for what is visible):
{schema}
"""


def build_pa(p0: str) -> str:
    return PA_TEMPLATE.replace("{schema}", _schema(p0))


def build_pc(p0: str) -> str:
    compact = json.dumps(json.loads(_schema(p0)), separators=(",", ":"))
    return PC_TEMPLATE.replace("{schema}", compact)


BUILDERS = {
    "e6_pa_paraphrase": (build_pa, "P-a: mesmas regras, redacao nova"),
    "e6_pb_non_frb_first": (build_pb, "P-b: mesmas regras, lista de NON_FRB primeiro"),
    "e6_pc_short": (build_pc, "P-c: ~50% dos tokens, mesmas regras"),
    "e6_pd_generic_non_frb": (build_pd, "P-d: NON_FRB generico, sem as familias de RFI"),
    "e6_pe_no_low_snr": (build_pe, "P-e: sem a clausula Low-S/N"),
    "e6_r_relaxed_morphology": (build_r, "R: morfologia relaxada"),
}


def build(p0: str) -> dict[str, str]:
    return {name: builder(p0) for name, (builder, _) in BUILDERS.items()}


def _write(_args: argparse.Namespace) -> None:
    p0 = P0_FILE.read_text(encoding="utf-8")
    registry_path = PROMPTS_DIR / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for name, text in build(p0).items():
        path = PROMPTS_DIR / f"{name}.txt"
        if name in registry or path.exists():
            raise SystemExit(f"{name} ja congelado; nada foi gravado.")
        path.write_bytes(text.encode("utf-8"))
        registry[name] = {
            "file": path.as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "origin": f"revision_v4/e6_prompts.py ({BUILDERS[name][1]}), a partir de binary_full",
            "frozen": FROZEN,
        }
        print(f"{path}: {registry[name]['sha256']}")
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("write", help="Gravar e registrar os prompts.").set_defaults(func=_write)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
