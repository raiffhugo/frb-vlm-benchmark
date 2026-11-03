"""Prompts do E7 (plano, secoes 10.9 e 10.12), derivados do lean+reason da Tabela 10 (V2).

O lean (V1) e o lean+reason (V2) da Tabela 10 diferem em duas coisas: a justificativa (a frase
"Describe the visual evidence first, then decide." e o campo ``reason``) e a ordem dos campos
de decisao (``label`` antes de ``frb_probability`` no V1, depois no V2). Por isso o E7 ganha o
V1o, o V2 sem a justificativa e com a ordem do V2, e as outras variantes saem do V2 trocando so
as duas ultimas linhas (a frase de instrucao e o esquema JSON):

  - V1o: sem justificativa, campos de decisao na ordem do V2;
  - V3: justificativa depois dos campos de decisao;
  - V4: filler neutro antes dos campos, um por modelo, com tantos tokens quanto a media da
    justificativa do proprio modelo nas 2000 respostas lean+reason do E1 (pontos separados por
    espaco, um token cada no tokenizador do Gemma 4);
  - V5: a instrucao de justificativa reescrita, antes dos campos.

V1 e V2 sao os arquivos ``binary_lean.txt`` e ``binary_lean_reason.txt``, sem copia. ``write``
grava os arquivos em prompts_v4/ e registra os hashes; os testes conferem que os arquivos sao
a saida de ``build``. Depois do congelamento nada aqui muda (prompts_v4/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROMPTS_DIR = Path("prompts_v4")
V2_FILE = PROMPTS_DIR / "binary_lean_reason.txt"
FROZEN = "2026-10-19"

# Media de tokens do campo reason nas 2000 respostas lean+reason do E1 de cada modelo
# (tokenizador do Gemma 4 na revisao fixada): 21,41 no E2B e 24,18 no E4B.
FILLER_TOKENS = {"e2b": 21, "e4b": 24}

_V2_INSTRUCTION = "Describe the visual evidence first, then decide. "
_RETURN = "Return ONLY valid JSON, no markdown, no comments, no extra keys, exactly this schema:\n"
_V2_SCHEMA = '{"reason": "one short sentence citing the visual evidence", "frb_probability": 0.0, "label": "FRB|NON_FRB"}'


def filler(tokens: int) -> str:
    return " ".join(["."] * tokens)


def _tail(v2: str) -> tuple[str, str]:
    """O texto comum a todas as variantes e as duas ultimas linhas do V2."""
    tail = _V2_INSTRUCTION + _RETURN + _V2_SCHEMA
    if not v2.endswith(tail) or v2.count(_V2_INSTRUCTION) != 1:
        raise ValueError("O V2 nao termina na instrucao e no esquema esperados.")
    return v2[: -len(tail)], tail


def _variant(v2: str, instruction: str, schema: str) -> str:
    body, _ = _tail(v2)
    return body + instruction + _RETURN + schema


def build(v2: str) -> dict[str, str]:
    prompts = {
        "e7_v1o_no_reason": _variant(v2, "", '{"frb_probability": 0.0, "label": "FRB|NON_FRB"}'),
        "e7_v3_reason_after": _variant(
            v2,
            "Decide first, then describe the visual evidence. ",
            '{"frb_probability": 0.0, "label": "FRB|NON_FRB", "reason": "one short sentence citing the visual evidence"}',
        ),
        "e7_v5_reason_reworded": _variant(
            v2,
            "Before giving the probability and the label, summarize in one brief sentence what you"
            " observe in the image that bears on the decision. ",
            '{"reason": "one brief sentence summarizing the observed evidence", "frb_probability": 0.0, "label": "FRB|NON_FRB"}',
        ),
    }
    for model, tokens in FILLER_TOKENS.items():
        prompts[f"e7_v4_filler_{model}"] = _variant(
            v2,
            "First copy the filler string exactly as shown; it carries no information. Then decide. ",
            '{"filler": "' + filler(tokens) + '", "frb_probability": 0.0, "label": "FRB|NON_FRB"}',
        )
    return prompts


DESCRIPTIONS = {
    "e7_v1o_no_reason": "V1o: V2 sem a justificativa, mesma ordem dos campos de decisao",
    "e7_v3_reason_after": "V3: justificativa depois dos campos de decisao",
    "e7_v4_filler_e2b": f"V4 do E2B: filler de {FILLER_TOKENS['e2b']} tokens antes dos campos",
    "e7_v4_filler_e4b": f"V4 do E4B: filler de {FILLER_TOKENS['e4b']} tokens antes dos campos",
    "e7_v5_reason_reworded": "V5: instrucao de justificativa reescrita",
}


def _write(_args: argparse.Namespace) -> None:
    v2 = V2_FILE.read_text(encoding="utf-8")
    registry_path = PROMPTS_DIR / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for name, text in build(v2).items():
        path = PROMPTS_DIR / f"{name}.txt"
        if name in registry or path.exists():
            raise SystemExit(f"{name} ja congelado; nada foi gravado.")
        path.write_bytes(text.encode("utf-8"))
        registry[name] = {
            "file": path.as_posix(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "origin": f"revision_v4/e7_prompts.py ({DESCRIPTIONS[name]}), a partir de binary_lean_reason",
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
