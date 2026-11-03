"""Ferramenta de rotulacao humana do E5 e da auditoria do E4 (plano, secoes 10.6 e 10.7).

``build`` monta, para cada avaliador, uma pasta com ``index.html`` (funciona offline, sem
bibliotecas externas), ``LEIA-ME.txt`` e as imagens com nomes anonimos:
  1. treino: 10 imagens simuladas, com a resposta certa mostrada depois de cada uma;
  2. parte A (E5): os 300 PNGs do artigo, "a imagem contem um FRB (varredura dispersa)?",
     sim/nao e confianca de 1 a 3;
  3. parte B (auditoria do E4): os 200 negativos, "ha RFI visivel?", sim/nao e o tipo.
A ordem de cada parte e sorteada por avaliador. A pagina nao contem rotulo, sigma nem resposta
de modelo: a chave (nome anonimo -> amostra) fica em results_v4/e5/labeling_key.jsonl, que
nao vai para os avaliadores. As respostas ficam no navegador a cada imagem e saem num CSV.

Versao 2 (01/11/2026, plano 10.12): dentro das partes A e B o avaliador pode voltar as imagens
anteriores e mudar a resposta, que so muda quando ele avanca; o treino nao volta, e a parte A
fecha quando a parte B comeca. O guia da parte B (o esclarecimento de 01/11) esta na pagina e
no LEIA-ME. O CSV tem uma linha por imagem, com a resposta final, a primeira, o numero de
mudancas e de confirmacoes e a versao da ferramenta da primeira resposta. As respostas que a
versao 1 guardou no navegador continuam valendo, como primeiras respostas da versao 1.

``pages`` regrava o ``index.html`` e o ``LEIA-ME.txt`` com a chave ja montada, sem mexer nela
nem nas imagens, e para se a sequencia de imagens de algum avaliador mudaria. ``pack`` faz o zip
da pasta de um avaliador, com nomes em ordem e datas fixas. O teste de ponta a ponta da pagina,
no Chrome headless, e ``tests/e2e_labeling_tool.mjs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from revision_v4.common import REAL_SEED, RESULTS_ROOT, SIM_SEED, read_jsonl, run_manifest, sha256_file, write_json, write_jsonl


OUTPUT_DIR = RESULTS_ROOT / "e5"
KEY_PATH = OUTPUT_DIR / "labeling_key.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "labeling_manifest.json"
PART_A = RESULTS_ROOT / "selections" / "e5_humans.jsonl"
PART_B = RESULTS_ROOT / "selections" / "e4_audit.jsonl"
SIM_MANIFEST = Path("dataset/metadata/image_manifest.jsonl")
SIM_VISIBILITY = Path("dataset/metadata/burst_visibility_sim.csv")
RATERS = ("A", "B")

# Treino: 5 FRB com o sweep inteiro no arquivo, 3 RFI de familias diferentes e 2 NOISE,
# sorteados do benchmark com a seed do simulado. No benchmark quase nao ha FRB contido abaixo
# de 30 sigma (1 entre 8 e 15, nenhum entre 15 e 30), entao as faixas comecam em 30.
TRAINING_FRB_STRATA = (((30, 100), 2), ((100, 300), 2), ((300, float("inf")), 1))
TRAINING_RFI = ("persistent_narrowband", "impulsive_broadband", "wifi_multiband")
RFI_TYPES = (
    ("horizontal", "Faixa horizontal persistente (banda estreita)"),
    ("vertical", "Impulso vertical (banda larga, sem atraso)"),
    ("periodic", "Padrão periódico ou em grade"),
    ("other", "Outro"),
)
CONFIG_OPEN = '<script id="config" type="application/json">'
# Campos do config que definem o que o avaliador ve e em que ordem.
SEQUENCE_FIELDS = ("rater", "training", "partA", "partB", "rfiTypes")
PACK_DATE = (2026, 11, 1, 0, 0, 0)


def training_items() -> list[dict[str, Any]]:
    rng = np.random.default_rng([SIM_SEED, 5])
    records = {r["sample_id"]: r for r in read_jsonl(SIM_MANIFEST)}
    with SIM_VISIBILITY.open(encoding="utf-8") as handle:
        visibility = {row["sample_id"]: row for row in csv.DictReader(handle)}

    def draw(pool: list[str], k: int) -> list[str]:
        return [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]

    chosen: list[tuple[str, str]] = []
    for (low, high), k in TRAINING_FRB_STRATA:
        pool = sorted(
            sample_id
            for sample_id, r in records.items()
            if r["true_label"] == "FRB"
            and visibility[sample_id]["sweep_contained"] == "1"
            and low <= float(visibility[sample_id]["peak_sigma"]) < high
        )
        chosen += [(sample_id, "FRB") for sample_id in draw(pool, k)]
    for family in TRAINING_RFI:
        pool = sorted(
            sample_id
            for sample_id, r in records.items()
            if r["true_label"] == "RFI" and family in r["simulation_parameters"].get("rfi_types", [])
        )
        chosen += [(sample_id, "NON_FRB") for sample_id in draw(pool, 1)]
    noise = sorted(sample_id for sample_id, r in records.items() if r["true_label"] == "NOISE")
    chosen += [(sample_id, "NON_FRB") for sample_id in draw(noise, 2)]
    return [
        {"sample_id": chosen[i][0], "image_path": records[chosen[i][0]]["image_path"], "answer": chosen[i][1]}
        for i in rng.permutation(len(chosen))
    ]


def anonymous_key() -> list[dict[str, Any]]:
    """Nome anonimo de cada imagem das partes A e B (o mesmo para os dois avaliadores)."""
    items = [("A", r) for r in read_jsonl(PART_A)] + [("B", r) for r in read_jsonl(PART_B)]
    rng = np.random.default_rng([REAL_SEED, 5])
    tokens = rng.choice(16**8, size=len(items), replace=False)
    return [
        {
            "anon_id": f"img_{int(token):08x}",
            "part": part,
            "sample_id": record["sample_id"],
            "image_path": record["image_path"],
            "true_label": record["true_label"],
            "peak_sigma": record.get("selection", {}).get("peak_sigma"),
        }
        for (part, record), token in zip(items, tokens)
    ]


def embed_json(payload: Any) -> str:
    """JSON seguro dentro de <script>: o texto e lido cru, entao so "</" precisa de escape."""
    return json.dumps(payload).replace("</", "<\\/")


def rater_order(key: list[dict[str, Any]], rater: str, part: str) -> list[str]:
    ids = sorted(entry["anon_id"] for entry in key if entry["part"] == part)
    rng = np.random.default_rng([REAL_SEED, 6, RATERS.index(rater), ord(part)])
    return [ids[i] for i in rng.permutation(len(ids))]


def _place(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def rater_config(key: list[dict[str, Any]], training: list[dict[str, str]], rater: str) -> dict[str, Any]:
    return {
        "rater": rater,
        "training": training,
        "partA": rater_order(key, rater, "A"),
        "partB": rater_order(key, rater, "B"),
        "rfiTypes": [{"value": v, "label": label} for v, label in RFI_TYPES],
    }


def embedded_config(page: str) -> dict[str, Any]:
    """O config de um index.html (o inverso do embed_json)."""
    start = page.index(CONFIG_OPEN) + len(CONFIG_OPEN)
    return json.loads(page[start : page.index("</script>", start)])


def write_pages(folder: Path, config: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(HTML_TEMPLATE.replace("__CONFIG__", embed_json(config)), encoding="utf-8")
    counts = {
        "training": len(config["training"]),
        "part_a": len(config["partA"]),
        "part_b": len(config["partB"]),
    }
    (folder / "LEIA-ME.txt").write_text(
        README_TEMPLATE.format(rater=config["rater"], total=sum(counts.values()), **counts), encoding="utf-8"
    )


def regenerate_pages(out: Path, key_path: Path = KEY_PATH, manifest_path: Path = MANIFEST_PATH) -> list[Path]:
    """Regrava as paginas com a chave ja montada. Para se a chave mudou depois da montagem ou se a
    sequencia de imagens de algum avaliador mudaria (as respostas guardadas no navegador seguem
    a posicao na sequencia)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(key_path) != manifest["key"]["sha256"]:
        raise SystemExit(f"{key_path} mudou depois da montagem da ferramenta ({manifest_path}).")
    key = read_jsonl(key_path)
    training = [{"id": f"treino_{i:02d}", "answer": t["answer"]} for i, t in enumerate(manifest["training"])]
    configs = {}
    for rater in manifest["raters"]:
        config = rater_config(key, training, rater)
        recorded = manifest["orders"][rater]
        if config["partA"][:5] != recorded["A"] or config["partB"][:5] != recorded["B"]:
            raise SystemExit(f"avaliador {rater}: a ordem sorteada difere da gravada em {manifest_path}.")
        page = out / f"avaliador_{rater}" / "index.html"
        if page.exists():
            old = embedded_config(page.read_text(encoding="utf-8"))
            changed = [field for field in SEQUENCE_FIELDS if old.get(field) != config[field]]
            if changed:
                raise SystemExit(f"{page}: a sequencia mudaria ({', '.join(changed)}).")
        configs[rater] = config
    for rater, config in configs.items():
        write_pages(out / f"avaliador_{rater}", config)
    return [out / f"avaliador_{rater}" for rater in configs]


def pack(folder: Path, target: Path) -> str:
    """Zip da pasta de um avaliador, com nomes em ordem e datas fixas; devolve o SHA-256. Para se
    a pasta tiver qualquer arquivo alem da pagina, do LEIA-ME e das imagens que a pagina usa."""
    config = embedded_config((folder / "index.html").read_text(encoding="utf-8"))
    ids = [t["id"] for t in config["training"]] + config["partA"] + config["partB"]
    expected = {"index.html", "LEIA-ME.txt"} | {f"images/{i}.png" for i in ids}
    files = {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}
    if files.keys() != expected:
        extra, missing = sorted(files.keys() - expected), sorted(expected - files.keys())
        raise SystemExit(f"{folder}: arquivos a mais {extra[:5]}, faltando {missing[:5]}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(f"{folder.name}/{name}", date_time=PACK_DATE)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_STORED if name.endswith(".png") else zipfile.ZIP_DEFLATED
            archive.writestr(info, files[name].read_bytes())
    return sha256_file(target)


def _build(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    key = anonymous_key()
    write_jsonl(KEY_PATH, key)
    training = training_items()
    training_config = [{"id": f"treino_{i:02d}", "answer": item["answer"]} for i, item in enumerate(training)]
    by_anon = {entry["anon_id"]: entry for entry in key}
    for rater in RATERS:
        folder = args.out / f"avaliador_{rater}"
        for entry in key:
            _place(Path(entry["image_path"]), folder / "images" / f"{entry['anon_id']}.png")
        for index, item in enumerate(training):
            _place(Path(item["image_path"]), folder / "images" / f"treino_{index:02d}.png")
        config = rater_config(key, training_config, rater)
        assert all(by_anon[i]["part"] == "A" for i in config["partA"])
        write_pages(folder, config)
        print(f"{folder}: {len(config['partA'])} + {len(config['partB'])} imagens, 10 de treino")
    write_json(
        MANIFEST_PATH,
        run_manifest(
            experiment="E5-E4-labeling-tool",
            project_root=project_root,
            key={"path": KEY_PATH.as_posix(), "sha256": sha256_file(KEY_PATH)},
            part_a={"path": PART_A.as_posix(), "sha256": sha256_file(PART_A)},
            part_b={"path": PART_B.as_posix(), "sha256": sha256_file(PART_B)},
            training=[{"sample_id": t["sample_id"], "answer": t["answer"]} for t in training],
            raters=RATERS,
            orders={r: {"A": rater_order(key, r, "A")[:5], "B": rater_order(key, r, "B")[:5]} for r in RATERS},
        ),
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rotulação FRB</title>
<style>
  :root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d8d8d2; --accent: #1f5fa8; --ok: #2e7d32; --bad: #b3261e; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ececea; --muted: #a2a29c; --line: #34342f; --accent: #7fb0ea; --ok: #81c784; --bad: #ef8a80; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; }
  main { max-width: 1360px; margin: 0 auto; padding: 12px 16px; }
  .text { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 4px 0 12px; }
  #task { display: grid; grid-template-columns: auto 290px; grid-template-areas: "bar bar" "img side"; gap: 8px 16px; justify-content: center; align-items: start; }
  .bar { grid-area: bar; display: flex; gap: 10px; align-items: center; color: var(--muted); }
  .progress { flex: 1; min-width: 120px; height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--accent); width: 0; }
  /* A imagem inteira cabe na tela, sem rolagem, e nunca passa do tamanho natural (1024 x 768). */
  #img { grid-area: img; display: block; box-sizing: content-box; width: min(1024px, (100vh - 70px) * 4 / 3, 100vw - 362px); height: auto; border: 1px solid var(--line); background: #000; }
  .side { grid-area: side; }
  .q { font-weight: 600; margin: 0 0 6px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
  .col { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
  .col button { text-align: left; padding: 5px 10px; font-size: 14px; line-height: 1.3; }
  button { font: inherit; padding: 8px 14px; border: 1px solid var(--line); border-radius: 6px; background: transparent; color: var(--fg); cursor: pointer; }
  button.sel { border-color: var(--accent); outline: 2px solid var(--accent); }
  button.primary { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  button.small { padding: 3px 10px; font-size: 13px; }
  button:disabled { opacity: .45; cursor: default; }
  .feedback { min-height: 1.4em; font-weight: 600; }
  .ok { color: var(--ok); } .bad { color: var(--bad); }
  .hint { color: var(--muted); font-size: 13px; }
  .warn { color: var(--bad); font-weight: 600; font-size: 13px; }
  #note { min-height: 1.3em; margin: 0 0 6px; }
  .guide { font-size: 12px; line-height: 1.3; color: var(--muted); border-top: 1px solid var(--line); padding-top: 6px; }
  .guide p { margin: 0 0 4px; }
  .guide ul { margin: 0 0 4px; padding-left: 16px; }
  .guide-full ul { padding-left: 20px; }
  .hidden { display: none !important; }
  @media (max-width: 799px) {
    #task { grid-template-columns: auto; grid-template-areas: "bar" "img" "side"; }
    #img { width: min(1024px, 100vw - 36px, (100vh - 300px) * 4 / 3); }
    .col { flex-direction: row; flex-wrap: wrap; }
  }
</style>
</head>
<body>
<main>
  <section id="intro" class="text">
    <h1>Rotulação de espectros dinâmicos</h1>
    <p>Cada imagem é um espectro dinâmico de 2 s do radiotelescópio FAST: tempo no eixo horizontal, frequência no vertical, intensidade na cor.</p>
    <p><b>Treino (<span data-count="training"></span> imagens simuladas):</b> responda e veja a resposta certa.</p>
    <p><b>Parte A (<span data-count="partA"></span> imagens):</b> a imagem contém um FRB, isto é, uma <b>varredura dispersa</b> (o sinal chega mais tarde nas frequências mais baixas)? Sim ou não, e a confiança (1 = pouca, 3 = muita).</p>
    <p><b>Parte B (<span data-count="partB"></span> imagens):</b> há <b>RFI visível</b>? RFI é interferência: qualquer estrutura que não seja o ruído de fundo nem uma varredura dispersa. Se houver, escolha o tipo; se houver mais de um, o mais evidente. O guia da parte B aparece antes dela e ao lado de cada imagem.</p>
    <p class="hint">Responda sozinho, sem consultar ninguém. Veja a imagem inteira, como a página a mostra, sem ampliar partes dela. Dentro das partes A e B, você pode voltar às imagens anteriores (botão Anterior ou tecla ←) e mudar a resposta; a mudança só vale quando você avança com Enter. O treino não volta, e a parte A fica fechada quando a parte B começa. As respostas ficam guardadas neste navegador: para pausar, feche a página e, ao reabrir, clique em Continuar. Use sempre o mesmo navegador, numa janela normal (não anônima). No fim, baixe o CSV e envie o arquivo. Atalhos: S/N para sim/não, 1–3 para a confiança, Enter para avançar, ← para voltar.</p>
    <p class="warn hidden" id="mismatch">As respostas guardadas neste navegador não batem com esta página. Não continue: feche a página e avise quem lhe enviou a ferramenta.</p>
    <button class="primary" id="start">Começar</button>
    <p class="hint">Versão 2 (01/11/2026).</p>
  </section>
  <section id="between" class="text hidden">
    <p id="betweenText"></p>
    <div id="guideFull" class="guide-full hidden">
      <p><b>Ruído de fundo</b> é a textura granulada e aleatória que cobre a imagem inteira: pontos mais claros e mais escuros espalhados ao acaso, sem forma, sem direção e sem repetição. Se a imagem só tem essa textura, a resposta é não.</p>
      <p><b>RFI</b> é uma estrutura: algo com forma, que se destaca dessa textura. Os tipos:</p>
      <ul>
        <li><b>Faixa horizontal persistente (banda estreita):</b> uma linha ou faixa horizontal, numa frequência fixa, que dura boa parte do tempo da imagem.</li>
        <li><b>Impulso vertical (banda larga, sem atraso):</b> uma linha vertical, num só instante, que ocupa muitas frequências ao mesmo tempo, sem a inclinação de uma varredura dispersa.</li>
        <li><b>Padrão periódico ou em grade:</b> uma estrutura que se repete em intervalos regulares, no tempo, na frequência ou nos dois.</li>
        <li><b>Outro:</b> qualquer outra estrutura que não seja a textura do ruído nem uma varredura dispersa.</li>
      </ul>
      <p>A parte B não tem confiança. Na dúvida, responda o que achar mais provável, olhando a imagem inteira como a página a mostra, sem ampliar.</p>
    </div>
    <div class="row"><button class="primary" id="go"></button><button id="backA" class="hidden">Voltar à parte A</button></div>
  </section>
  <section id="task" class="hidden">
    <div class="bar"><span id="stage"></span><div class="progress"><div id="prog"></div></div><span id="count"></span><button class="small" id="first">Início da parte</button><button class="small" id="resume">Ir para onde parei</button></div>
    <img id="img" alt="espectro dinâmico">
    <div class="side">
      <div class="q" id="question"></div>
      <div class="row" id="yesno"><button data-v="yes">Sim (S)</button><button data-v="no">Não (N)</button></div>
      <div id="confBlock"><div class="q">Confiança</div><div class="row" id="conf"><button data-v="1">1</button><button data-v="2">2</button><button data-v="3">3</button></div></div>
      <div id="typeBlock" class="hidden"><div class="q">Tipo de RFI</div><div class="col" id="types"></div></div>
      <div class="row"><button id="prev">← Anterior</button><button class="primary" id="next" disabled>Avançar (Enter)</button></div>
      <p class="hint" id="note"></p>
      <div class="feedback" id="feedback"></div>
      <p class="warn hidden" id="warn">O navegador não está guardando as respostas. Não feche esta página e avise quem lhe enviou a ferramenta.</p>
      <div class="guide hidden" id="guide">
        <p><b>Não:</b> só o ruído de fundo, a textura granulada e aleatória, sem forma, direção ou repetição.</p>
        <p><b>Sim:</b> uma estrutura com forma que se destaca dessa textura e não é uma varredura dispersa.</p>
        <ul>
          <li><b>Horizontal:</b> frequência fixa, boa parte do tempo.</li>
          <li><b>Vertical:</b> um só instante, muitas frequências, sem inclinação.</li>
          <li><b>Periódico:</b> repete-se em intervalos regulares.</li>
          <li><b>Outro:</b> qualquer outra estrutura.</li>
        </ul>
        <p>Mais de um tipo: o mais evidente. Na dúvida: o mais provável.</p>
      </div>
    </div>
  </section>
  <section id="done" class="text hidden">
    <p><b>Fim.</b> Obrigado! Clique em Baixar CSV e envie o arquivo. Se voltar e mudar alguma resposta, baixe o CSV de novo. Não apague os dados do navegador até receber a confirmação de que o arquivo chegou certo: a página baixa o CSV de novo se for preciso.</p>
    <div class="row"><button class="primary" id="download">Baixar CSV</button><button id="backB">Voltar à parte B</button></div>
  </section>
</main>
<script id="config" type="application/json">__CONFIG__</script>
<script>
const CONFIG = JSON.parse(document.getElementById("config").textContent);
const VERSION = 2;
const STORE = "frb_rotulacao_" + CONFIG.rater;
const NL = String.fromCharCode(10);
const steps = [
  ...CONFIG.training.map(t => ({ part: "treino", id: t.id, answer: t.answer })),
  ...CONFIG.partA.map(id => ({ part: "A", id })),
  ...CONFIG.partB.map(id => ({ part: "B", id })),
];
const partStart = part => steps.findIndex(s => s.part === part);
const partSize = part => steps.filter(s => s.part === part).length;
const FRB_QUESTION = "A imagem contém um FRB (varredura dispersa)?";
const QUESTIONS = { treino: FRB_QUESTION, A: FRB_QUESTION, B: "Há RFI visível nesta imagem?" };
const BETWEEN = {
  A: "<b>Fim do treino.</b> Agora começa a parte A: " + CONFIG.partA.length + " imagens, sem a resposta certa. Para cada uma: a imagem contém um FRB (varredura dispersa)? Sim ou não, e a confiança (1 = pouca, 3 = muita). Dentro da parte, você pode voltar às imagens anteriores (Anterior ou ←) e mudar a resposta; a mudança só vale quando você avança com Enter.",
  B: "<b>Fim da parte A.</b> Se quiser, faça uma pausa agora: pode fechar a página e voltar depois. Quando a parte B começar, a parte A fica fechada: se quiser rever alguma resposta dela, volte agora. Na parte B (" + CONFIG.partB.length + " imagens), <b>a pergunta muda</b>: há RFI visível na imagem? Se houver, escolha o tipo com o mouse; se houver mais de um, o mais evidente.",
};
// Uma linha por imagem: a resposta final, a primeira e o histórico resumido.
const COLUMNS = ["rater", "part", "image", "answer", "confidence", "rfi_type", "response_ms", "time_utc",
  "first_answer", "first_confidence", "first_rfi_type", "first_time_utc", "changes", "confirmations", "total_ms", "first_tool"];

let state = { version: VERSION, index: 0, answers: [], started: {} };
try { const saved = JSON.parse(localStorage.getItem(STORE)); if (saved && Array.isArray(saved.answers)) state = saved; } catch (e) {}
state.started = state.started || {};
for (const a of state.answers) {  // resposta gravada pela versão 1, sem volta: a primeira é a própria resposta
  if (a.first_answer === undefined) {
    Object.assign(a, { first_answer: a.answer, first_confidence: a.confidence, first_rfi_type: a.rfi_type,
      first_time_utc: a.time_utc, changes: 0, confirmations: 1, total_ms: a.response_ms, first_tool: 1 });
  }
}
const consistent = state.answers.length <= steps.length && state.answers.every(
  (a, i) => a.rater === CONFIG.rater && a.part === steps[i].part && a.image === steps[i].id);
let pos = state.answers.length;  // a primeira imagem sem resposta
let current = {}, shownAt = 0;
const $ = id => document.getElementById(id);

function save() {
  state.version = VERSION; state.index = state.answers.length;
  let saved = true;
  try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { saved = false; }
  $("warn").classList.toggle("hidden", saved);
}

function view(name) {
  for (const id of ["intro", "between", "task", "done"]) $(id).classList.toggle("hidden", id !== name);
}

function select(group, value) {
  for (const b of $(group).querySelectorAll("button")) b.classList.toggle("sel", b.dataset.v === value);
}

const stored = () => pos < state.answers.length ? state.answers[pos] : null;

function ready() {
  const step = steps[pos];
  if (!current.answer) return false;
  if (step.part === "B") return current.answer === "no" || !!current.type;
  return !!current.confidence;
}

function dirty() {  // a resposta na tela difere da guardada
  const s = stored();
  return !!s && ((current.answer || "") !== s.answer || (current.confidence || "") !== (s.confidence || "")
    || (current.type || "") !== (s.rfi_type || ""));
}

function canGoBack() {
  const step = steps[pos];
  return step.part !== "treino" && pos > partStart(step.part);
}

function refresh() {
  const step = steps[pos], s = stored();
  $("next").disabled = !ready();
  $("prev").disabled = !canGoBack();
  $("prev").classList.toggle("hidden", step.part === "treino");
  $("first").classList.toggle("hidden", !canGoBack());
  $("resume").classList.toggle("hidden", !s);
  const note = $("note");
  if (s && dirty()) { note.textContent = "Mudança não gravada: avance com Enter."; note.className = "warn"; }
  else if (s) { note.textContent = "Já respondida: para mudar, responda e avance."; note.className = "hint"; }
  else { note.textContent = ""; note.className = "hint"; }
}

function show() {
  if (!consistent) { view("intro"); return; }
  if (pos >= steps.length) { view("done"); return; }
  const step = steps[pos];
  if (step.part !== "treino" && !state.started[step.part]) {  // tela entre as etapas
    $("betweenText").innerHTML = BETWEEN[step.part];
    $("go").textContent = "Começar a parte " + step.part + " (Enter)";
    $("guideFull").classList.toggle("hidden", step.part !== "B");
    $("backA").classList.toggle("hidden", step.part !== "B");
    view("between");
    return;
  }
  view("task");
  const s = stored();
  current = s ? { answer: s.answer, confidence: s.confidence || undefined, type: s.rfi_type || undefined } : {};
  select("yesno", current.answer || null); select("conf", current.confidence || null); select("types", current.type || null);
  $("feedback").textContent = ""; $("feedback").className = "feedback";
  $("img").src = "images/" + step.id + ".png";
  const partB = step.part === "B";
  $("question").textContent = QUESTIONS[step.part];
  $("confBlock").classList.toggle("hidden", partB);
  $("typeBlock").classList.toggle("hidden", !(partB && current.answer === "yes"));
  $("guide").classList.toggle("hidden", !partB);
  const labels = { treino: "Treino", A: "Parte A", B: "Parte B" };
  $("stage").textContent = labels[step.part];
  $("count").textContent = (pos - partStart(step.part) + 1) + " / " + partSize(step.part);
  $("prog").style.width = (100 * state.answers.length / steps.length) + "%";
  shownAt = performance.now();
  refresh();
}

function begin() {
  state.started[steps[pos].part] = true;
  save(); show();
}

function answer(field, value) {
  const step = steps[pos];
  if (step.part === "treino" && $("feedback").textContent) return;  // no treino, a resposta trava com o feedback
  current[field] = value;
  if (field === "answer") {
    select("yesno", value);
    if (step.part === "B") {
      $("typeBlock").classList.toggle("hidden", value !== "yes");
      if (value !== "yes") { delete current.type; select("types", null); }  // o "não" apaga o tipo escolhido antes
    }
  }
  if (field === "confidence") select("conf", value);
  if (field === "type") select("types", value);
  refresh();
}

function next() {
  if (!ready()) return;
  const step = steps[pos];
  if (step.part === "treino" && !$("feedback").textContent) {
    const right = (current.answer === "yes") === (step.answer === "FRB");
    $("feedback").textContent = (right ? "Certo: " : "Errado: ") + (step.answer === "FRB" ? "esta imagem contém um FRB." : "esta imagem não contém FRB.");
    $("feedback").classList.add(right ? "ok" : "bad");
    return;
  }
  const now = new Date().toISOString(), elapsed = Math.round(performance.now() - shownAt), s = stored();
  if (s) {  // imagem revisitada: a resposta muda só se for diferente; a confirmação conta sempre
    if (dirty()) { s.answer = current.answer; s.confidence = current.confidence || ""; s.rfi_type = current.type || ""; s.changes += 1; }
    s.confirmations += 1; s.time_utc = now; s.total_ms += elapsed;
  } else {
    const a = { rater: CONFIG.rater, part: step.part, image: step.id, answer: current.answer,
      confidence: current.confidence || "", rfi_type: current.type || "", response_ms: elapsed, time_utc: now };
    Object.assign(a, { first_answer: a.answer, first_confidence: a.confidence, first_rfi_type: a.rfi_type,
      first_time_utc: now, changes: 0, confirmations: 1, total_ms: elapsed, first_tool: VERSION });
    state.answers.push(a);
  }
  pos += 1; save(); show();
}

function onTask() { return !$("task").classList.contains("hidden"); }
function prev() { if (onTask() && canGoBack()) { pos -= 1; show(); } }
function toFirst() { if (onTask() && canGoBack()) { pos = partStart(steps[pos].part); show(); } }
function resume() { pos = state.answers.length; show(); }
function backToA() { if (!state.started.B) { pos = partStart("B") - 1; show(); } }
function backToB() { pos = steps.length - 1; show(); }

function download() {
  const lines = [COLUMNS.join(",")].concat(state.answers.map(a => COLUMNS.map(c => String(a[c])).join(",")));
  const blob = new Blob([lines.join(NL) + NL], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "rotulacao_avaliador_" + CONFIG.rater + ".csv";
  document.body.appendChild(link); link.click(); link.remove();
}

for (const el of document.querySelectorAll("[data-count]")) el.textContent = CONFIG[el.dataset.count].length;
for (const t of CONFIG.rfiTypes) {
  const b = document.createElement("button"); b.dataset.v = t.value; b.textContent = t.label;
  b.onclick = () => answer("type", t.value); $("types").appendChild(b);
}
for (const b of $("yesno").querySelectorAll("button")) b.onclick = () => answer("answer", b.dataset.v);
for (const b of $("conf").querySelectorAll("button")) b.onclick = () => answer("confidence", b.dataset.v);
$("next").onclick = next;
$("prev").onclick = prev;
$("first").onclick = toFirst;
$("resume").onclick = resume;
$("backA").onclick = backToA;
$("backB").onclick = backToB;
$("download").onclick = download;
$("go").onclick = begin;
$("start").onclick = show;
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (!$("between").classList.contains("hidden")) {
    if (k === "enter") { e.preventDefault(); begin(); }
    return;
  }
  if (!onTask()) return;
  if (k === "s") answer("answer", "yes");
  else if (k === "n") answer("answer", "no");
  else if (["1", "2", "3"].includes(k) && steps[pos].part !== "B") answer("confidence", k);
  else if (k === "enter") { e.preventDefault(); next(); }  // sem o clique duplo do botao em foco
  else if (k === "arrowleft") { e.preventDefault(); prev(); }
});
if (!consistent) { $("mismatch").classList.remove("hidden"); $("start").disabled = true; }
else if (state.answers.length > 0) $("start").textContent = "Continuar (" + state.answers.length + " de " + steps.length + ")";
if (consistent && state.answers.length >= steps.length) view("done");
</script>
</body>
</html>
"""

README_TEMPLATE = """ROTULAÇÃO DE ESPECTROS DINÂMICOS (avaliador {rater}), VERSÃO 2 (01/11/2026)

Você vai classificar, sozinho, {total} imagens de espectros dinâmicos de 2 s do radiotelescópio
FAST: tempo no eixo horizontal, frequência no vertical e intensidade na cor. Leva cerca de uma
hora. A página guia as três etapas, nesta ordem:

  1. Treino: {training} imagens simuladas, com a resposta certa depois de cada uma.
  2. Parte A: {part_a} imagens. A imagem contém um FRB, isto é, uma varredura dispersa (o sinal
     chega mais tarde nas frequências mais baixas)? Sim ou não, e a confiança: 1 = pouca,
     2 = média, 3 = muita.
  3. Parte B: {part_b} imagens. A pergunta muda: há RFI visível na imagem? RFI é interferência:
     qualquer estrutura que não seja o ruído de fundo nem uma varredura dispersa. Se houver,
     escolha o tipo; se houver mais de um, o mais evidente. O guia abaixo também aparece na
     página, antes da parte B e ao lado de cada imagem.

GUIA DA PARTE B

  - Ruído de fundo é a textura granulada e aleatória que cobre a imagem inteira: pontos mais
    claros e mais escuros espalhados ao acaso, sem forma, sem direção e sem repetição. Se a
    imagem só tem essa textura, a resposta é não.
  - RFI é uma estrutura: algo com forma, que se destaca dessa textura. Os tipos da página:
      Faixa horizontal persistente (banda estreita): uma linha ou faixa horizontal, numa
        frequência fixa, que dura boa parte do tempo da imagem.
      Impulso vertical (banda larga, sem atraso): uma linha vertical, num só instante, que
        ocupa muitas frequências ao mesmo tempo, sem a inclinação de uma varredura dispersa.
      Padrão periódico ou em grade: uma estrutura que se repete em intervalos regulares, no
        tempo, na frequência ou nos dois.
      Outro: qualquer outra estrutura que não seja a textura do ruído nem uma varredura
        dispersa.
  - A parte B não tem confiança. Na dúvida, responda o que achar mais provável, olhando a
    imagem inteira como a página a mostra, sem ampliar.

ANTES DE COMEÇAR

  1. Se recebeu a ferramenta num zip, extraia o zip inteiro numa pasta (no Windows: botão
     direito > Extrair tudo). Não abra o index.html de dentro do zip: as imagens não carregam.
  2. Abra o index.html no Chrome, no Edge ou no Firefox (no Mac, não use o Safari), numa janela
     normal, não anônima nem privada. Use sempre o mesmo navegador: as respostas ficam nele.
  3. Use a maior tela que tiver, com o navegador em tela cheia (F11; no Mac, Ctrl+Cmd+F) e o
     zoom em 100% (Ctrl+0; no Mac, Cmd+0). A página ajusta a imagem para caber inteira.
  4. Depois do treino, feche o navegador inteiro e abra o index.html de novo. O botão deve dizer
     "Continuar ({training} de {total})". Se disser "Começar", o navegador não está guardando as
     respostas: pare e avise quem lhe enviou a ferramenta.

SE VOCÊ JÁ COMEÇOU NA VERSÃO 1

  As respostas ficam no navegador, não na pasta, e a versão 2 continua de onde você parou.
  1. Não apague os dados do navegador. Apague a pasta avaliador_{rater} antiga e extraia este
     zip no mesmo lugar, de modo que o index.html fique no mesmo caminho de antes.
  2. Abra o index.html no mesmo navegador. O botão deve dizer "Continuar (N de {total})", com o
     número de imagens que você já respondeu. Se disser "Começar", não clique: feche a página e
     avise quem lhe enviou a ferramenta.
  3. Se você já respondeu imagens da parte B, volte ao começo dela (botão "Início da parte"),
     leia o guia e confirme cada resposta com Enter, ou corrija-a antes. Depois, o botão "Ir
     para onde parei" leva à primeira imagem sem resposta.

DURANTE

  - Responda sozinho. Não converse sobre as imagens com ninguém, nem com o outro avaliador, até
    os dois terminarem.
  - Veja cada imagem inteira, como a página a mostra. Não amplie partes dela e não abra a pasta
    images nem as imagens em outro programa.
  - Não há limite de tempo: olhe cada imagem o quanto precisar.
  - Dentro das partes A e B, você pode voltar às imagens anteriores (botão "Anterior" ou tecla
    ←) e mudar a resposta. A mudança só vale quando você avança com Enter. O treino não volta,
    e a parte A fica fechada quando a parte B começa: a tela antes da parte B deixa voltar à
    parte A.
  - Para pausar, feche a página; ao reabrir, clique em "Continuar". A pausa natural é entre as
    partes A e B.
  - Atalhos: S = sim, N = não, 1, 2 e 3 = confiança, Enter = avançar, ← = voltar. Na parte B, o
    tipo de RFI se escolhe com o mouse.

NO FIM

  Clique em "Baixar CSV". O arquivo rotulacao_avaliador_{rater}.csv vai para a pasta de
  downloads. Se depois voltar e mudar alguma resposta, baixe o CSV de novo. Envie esse arquivo a
  quem lhe enviou a ferramenta. Não apague os dados do navegador nem esta pasta até receber a
  confirmação de que o arquivo chegou certo; se for preciso, a página baixa o CSV de novo.
"""


def _pages(args: argparse.Namespace) -> None:
    for folder in regenerate_pages(args.out):
        print(f"{folder}: index.html e LEIA-ME.txt regravados, com a mesma sequencia de imagens")


def _pack(args: argparse.Namespace) -> None:
    target = args.out / f"avaliador_{args.rater}.zip"
    print(f"{target}: sha256 {pack(args.out / f'avaliador_{args.rater}', target)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Montar as pastas dos avaliadores.")
    build.add_argument("--out", type=Path, default=Path("labeling_v4"))
    build.set_defaults(func=_build)
    pages = subparsers.add_parser("pages", help="Regravar index.html e LEIA-ME.txt com a chave ja montada.")
    pages.add_argument("--out", type=Path, default=Path("labeling_v4"))
    pages.set_defaults(func=_pages)
    packing = subparsers.add_parser("pack", help="Zip da pasta de um avaliador, para enviar.")
    packing.add_argument("--rater", choices=RATERS, required=True)
    packing.add_argument("--out", type=Path, default=Path("labeling_v4"))
    packing.set_defaults(func=_pack)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
