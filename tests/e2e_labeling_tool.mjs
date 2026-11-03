// Teste de ponta a ponta da ferramenta de rotulacao (revision_v4/labeling_tool.py) no Chrome
// headless, pelo Chrome DevTools Protocol, sem bibliotecas (Node >= 22).
//
//   node tests/e2e_labeling_tool.mjs run <pasta do avaliador> <pasta de trabalho>
//   node tests/e2e_labeling_tool.mjs migrate <pasta com a pagina da versao 1> <index.html da versao 2> <pasta de trabalho>
//   node tests/e2e_labeling_tool.mjs shots <pasta do avaliador> <pasta de trabalho>
//
// "run" responde todas as imagens pelo teclado, com respostas sorteadas (seed fixa), fecha e
// reabre o navegador depois do treino e no meio da parte A, testa as travas (Enter sem resposta,
// resposta do treino travada depois do feedback, tipo de RFI apagado por um "nao" e confianca
// ignorada na parte B) e a navegacao da versao 2 (voltar, mudar e gravar com Enter, mudanca
// descartada ao voltar sem gravar, "Inicio da parte", "Ir para onde parei", treino sem volta,
// parte A fechada quando a parte B comeca, volta a parte A antes disso e a parte B depois do
// fim). Baixa o CSV, compara cada linha com as respostas dadas (final, primeira, mudancas e
// confirmacoes) e roda o `check` do e5_humans.
// "migrate" responde o treino, a parte A e 15 imagens da parte B na pagina da versao 1, troca o
// index.html pela versao 2 no mesmo caminho, confere que ela continua de onde parou, muda uma
// resposta antiga, termina, confere o CSV e o `check`, e confere que um estado que nao bate com a
// pagina e recusado.
// "shots" grava capturas de cada tela em varios tamanhos e confere se a pagina cabe sem rolagem.
// O perfil do navegador fica na pasta de trabalho; a chave nao e lida.
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const CHROME = process.env.CHROME || "google-chrome";
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function mulberry32(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Browser {
  static async launch(profile) {
    fs.mkdirSync(profile, { recursive: true });
    const portFile = path.join(profile, "DevToolsActivePort");
    fs.rmSync(portFile, { force: true });
    const proc = spawn(CHROME, [
      "--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
      "--no-first-run", "--no-default-browser-check", "--disable-gpu", "about:blank",
    ], { stdio: "ignore" });
    for (let i = 0; i < 300 && !fs.existsSync(portFile); i++) await sleep(50);
    const [port, wsPath] = fs.readFileSync(portFile, "utf8").trim().split("\n");
    const browser = new Browser(proc, `ws://127.0.0.1:${port}${wsPath}`);
    await browser.connect();
    return browser;
  }

  constructor(proc, url) {
    Object.assign(this, { proc, url, seq: 0, pending: new Map(), handlers: new Set() });
  }

  connect() {
    this.ws = new WebSocket(this.url);
    this.ws.onmessage = event => {
      const msg = JSON.parse(event.data);
      if (msg.id === undefined) { for (const handler of this.handlers) handler(msg); return; }
      const waiting = this.pending.get(msg.id);
      if (!waiting) return;
      this.pending.delete(msg.id);
      if (msg.error) waiting.reject(new Error(msg.error.message)); else waiting.resolve(msg.result);
    };
    return new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; });
  }

  send(method, params = {}, sessionId) {
    const id = ++this.seq;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }

  once(method, predicate = () => true, timeout = 20000) {
    return new Promise((resolve, reject) => {
      const handler = msg => {
        if (msg.method !== method || !predicate(msg.params)) return;
        clearTimeout(timer); this.handlers.delete(handler); resolve(msg.params);
      };
      const timer = setTimeout(() => { this.handlers.delete(handler); reject(new Error("tempo esgotado: " + method)); }, timeout);
      this.handlers.add(handler);
    });
  }

  async page(width, height) {
    const { targetId } = await this.send("Target.createTarget", { url: "about:blank" });
    const { sessionId } = await this.send("Target.attachToTarget", { targetId, flatten: true });
    const page = new Page(this, sessionId, targetId);
    await page.send("Page.enable");
    await page.send("Runtime.enable");
    await page.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
    return page;
  }

  async close() {
    const exited = new Promise(resolve => this.proc.once("exit", resolve));
    this.send("Browser.close").catch(() => {});
    if (await Promise.race([exited.then(() => true), sleep(10000).then(() => false)])) return;
    this.proc.kill("SIGKILL");
    await exited;
  }
}

const KEYS = {
  Enter: ["Enter", "Enter", 13, "\r"], s: ["s", "KeyS", 83, "s"], n: ["n", "KeyN", 78, "n"],
  1: ["1", "Digit1", 49, "1"], 2: ["2", "Digit2", 50, "2"], 3: ["3", "Digit3", 51, "3"],
  ArrowLeft: ["ArrowLeft", "ArrowLeft", 37, ""],
};

class Page {
  constructor(browser, sessionId, targetId) { Object.assign(this, { browser, sessionId, targetId }); }
  send(method, params) { return this.browser.send(method, params, this.sessionId); }

  async goto(url) {
    const loaded = this.browser.once("Page.loadEventFired");
    await this.send("Page.navigate", { url });
    await loaded;
  }

  async eval(expression) {
    const result = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }

  async key(name) {
    const [key, code, vk, text] = KEYS[name];
    const down = text ? { type: "keyDown", text } : { type: "rawKeyDown" };
    await this.send("Input.dispatchKeyEvent", { ...down, key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", key, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk });
  }

  click(selector) { return this.eval(`document.querySelector(${JSON.stringify(selector)}).click()`); }
}

// Estado visivel da pagina. Na versao 1 nao ha navegacao: os campos dela saem nulos ou vazios.
const PROBE = `(() => {
  const $ = id => document.getElementById(id);
  const visible = id => !!$(id) && !$(id).classList.contains("hidden") && $(id).getClientRects().length > 0;
  const sel = group => { const b = document.querySelector("#" + group + " button.sel"); return b ? b.dataset.v : ""; };
  const img = $("img"), box = img.getBoundingClientRect(), next = $("next").getBoundingClientRect();
  return {
    view: ["intro", "between", "task", "done"].find(visible) || null,
    stage: $("stage").textContent, count: $("count").textContent,
    src: img.getAttribute("src"), loaded: img.complete ? img.naturalWidth : 0,
    feedback: $("feedback").textContent, nextDisabled: $("next").disabled,
    typeVisible: visible("typeBlock"), selectedTypes: [...document.querySelectorAll("#types button.sel")].length,
    selection: { answer: sel("yesno"), confidence: sel("conf"), type: sel("types") },
    start: $("start").textContent, startDisabled: $("start").disabled, warn: visible("warn"),
    prev: $("prev") ? { visible: visible("prev"), disabled: $("prev").disabled } : null,
    first: visible("first"), resume: visible("resume"), backA: visible("backA"),
    note: $("note") ? $("note").textContent : "", guide: visible("guide"), guideFull: visible("guideFull"),
    mismatch: visible("mismatch"),
    page: [document.documentElement.scrollWidth, document.documentElement.scrollHeight, innerWidth, innerHeight],
    img: [Math.round(box.left), Math.round(box.top), Math.round(box.width), Math.round(box.height)],
    nextBottom: Math.round(next.bottom),
  };
})()`;

const PARTS = { "Treino": "treino", "Parte A": "A", "Parte B": "B" };
const COLUMNS = ["rater", "part", "image", "answer", "confidence", "rfi_type", "response_ms", "time_utc",
  "first_answer", "first_confidence", "first_rfi_type", "first_time_utc", "changes", "confirmations", "total_ms", "first_tool"];

async function waitFor(page, condition, what) {
  for (let i = 0; i < 500; i++) {
    const state = await page.eval(PROBE);
    if (condition(state)) return state;
    await sleep(10);
  }
  throw new Error("esperando " + what);
}

async function open(profile, url) {
  const browser = await Browser.launch(profile);
  const page = await browser.page(1920, 1080);
  await page.goto(url);
  return { browser, page };
}

function readConfig(folder) {
  return JSON.parse(
    fs.readFileSync(path.join(folder, "index.html"), "utf8")
      .match(/<script id="config" type="application\/json">([\s\S]*?)<\/script>/)[1],
  );
}

function parseCsv(text) {
  const lines = text.split("\n").filter(line => line.length);
  return { header: lines[0].split(","), rows: lines.slice(1).map(line => line.split(",")) };
}

const imageOf = src => src.replace(/^images\//, "").replace(/\.png$/, "");
const sameSelection = (selection, row) =>
  selection.answer === row.answer && selection.confidence === row.confidence && selection.type === row.rfi_type;

function newRow(rater, part, image, answer, confidence, rfiType, tool) {
  const row = { rater, part, image, answer, confidence, rfi_type: rfiType, changes: 0, confirmations: 1, first_tool: tool };
  row.first = { answer, confidence, rfi_type: rfiType };
  return row;
}

// Responde a imagem da vez com respostas sorteadas; devolve a linha esperada.
async function answerRandom(page, part, image, rater, types, pick, tool) {
  const answer = pick(["yes", "no"]);
  let confidence = "", rfiType = "";
  await page.key(answer === "yes" ? "s" : "n");
  if (part === "B") {
    if (answer === "yes") { rfiType = pick(types); await page.click(`#types button[data-v="${rfiType}"]`); }
  } else {
    confidence = pick(["1", "2", "3"]);
    await page.key(confidence);
  }
  return newRow(rater, part, image, answer, confidence, rfiType, tool);
}

// Na primeira imagem da parte B: muda a resposta guardada (sim vira nao; nao vira sim com tipo) e grava.
async function flipPartB(page, row, types, pick) {
  if (row.answer === "yes") {
    await page.key("n");
    Object.assign(row, { answer: "no", rfi_type: "" });
  } else {
    const type = pick(types);
    await page.key("s");
    await page.click(`#types button[data-v="${type}"]`);
    Object.assign(row, { answer: "yes", rfi_type: type });
  }
  await page.key("Enter");
  row.changes += 1; row.confirmations += 1;
}

async function download(browser, page, downloads, rater) {
  fs.rmSync(downloads, { recursive: true, force: true });
  fs.mkdirSync(downloads, { recursive: true });
  await browser.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: path.resolve(downloads), eventsEnabled: true });
  const finished = browser.once("Browser.downloadProgress", p => p.state === "completed" || p.state === "canceled");
  await page.click("#download");
  const progress = await finished;
  return { state: progress.state, csvPath: path.join(downloads, `rotulacao_avaliador_${rater}.csv`) };
}

// Compara o CSV com as linhas esperadas e roda o check; devolve o numero de linhas diferentes.
function compareCsv(csvPath, expected, rater, check, findings) {
  const { header, rows } = parseCsv(fs.readFileSync(csvPath, "utf8"));
  check(header.join(",") === COLUMNS.join(","), "cabecalho " + header.join(","));
  check(rows.length === expected.length, `linhas ${rows.length}, esperadas ${expected.length}`);
  let mismatches = 0;
  rows.forEach((row, i) => {
    const f = Object.fromEntries(COLUMNS.map((c, j) => [c, row[j]]));
    const want = expected[i] || { first: {} };
    const wanted = {
      rater: want.rater, part: want.part, image: want.image, answer: want.answer, confidence: want.confidence,
      rfi_type: want.rfi_type, first_answer: want.first.answer, first_confidence: want.first.confidence,
      first_rfi_type: want.first.rfi_type, changes: String(want.changes), confirmations: String(want.confirmations),
      first_tool: want.first_tool,
    };
    const differs = Object.keys(wanted).filter(c => f[c] !== wanted[c]);
    if (differs.length && mismatches++ < 10) findings.push(`linha ${i + 2}: ${differs.map(c => `${c}=${f[c]} (esperado ${wanted[c]})`).join(", ")}`);
    const times = [Date.parse(f.first_time_utc), Date.parse(f.time_utc)];
    check(/^\d+$/.test(f.response_ms) && /^\d+$/.test(f.total_ms) && Number(f.total_ms) >= Number(f.response_ms)
      && !times.some(Number.isNaN) && times[0] <= times[1], `linha ${i + 2}: tempos`);
  });
  const checked = spawnSync("uv", ["run", "python", "-m", "revision_v4.e5_humans", "check", "--rater", rater, "--csv", csvPath], { encoding: "utf8" });
  check(checked.status === 0, "check: " + (checked.stderr || checked.stdout).trim());
  return { rows: rows.length, mismatches, check: (checked.stdout || checked.stderr).trim() };
}

async function run(folder, work) {
  const config = readConfig(folder);
  const rater = config.rater, total = config.training.length + config.partA.length + config.partB.length;
  const nA = config.partA.length, nB = config.partB.length;
  const types = config.rfiTypes.map(t => t.value);
  const profile = path.join(work, "profile_" + rater), downloads = path.join(work, "downloads_" + rater);
  fs.rmSync(profile, { recursive: true, force: true });
  const url = pathToFileURL(path.resolve(folder, "index.html")).href;
  const rng = mulberry32(20261020), pick = values => values[Math.floor(rng() * values.length)];
  const restartAt = new Set([config.training.length, config.training.length + 150]);
  const expected = [], findings = [], screens = [];
  const check = (ok, message) => { if (!ok) findings.push(message); };
  const rowsOf = part => expected.filter(r => r.part === part);

  let { browser, page } = await open(profile, url);
  let state = await page.eval(PROBE);
  check(state.view === "intro" && state.start === "Começar", "tela inicial: " + JSON.stringify([state.view, state.start]));
  await page.click("#start");
  let previous = null, backToA = false, backToB = false;
  const seen = { treino: 0, A: 0, B: 0 };
  while (true) {
    state = await page.eval(PROBE);
    if (state.view === "between") {
      if (state.backA && !backToA) {  // antes da parte B: volta a ultima da parte A e confirma sem mudar
        backToA = true;
        check(state.guideFull, "guia completo ausente antes da parte B");
        const last = rowsOf("A").at(-1);
        await page.click("#backA");
        const back = await waitFor(page, s => s.view === "task" && s.loaded > 0, "volta a parte A");
        check(back.count === `${nA} / ${nA}` && sameSelection(back.selection, last), "volta a parte A: " + JSON.stringify([back.count, back.selection]));
        await page.key("Enter");
        last.confirmations += 1;
        await waitFor(page, s => s.view === "between", "tela antes da parte B de novo");
        continue;
      }
      screens.push(expected.length);
      await page.key("Enter");
      continue;
    }
    if (state.view === "done") {
      if (!backToB) {  // depois do fim: volta a ultima da parte B e confirma sem mudar
        backToB = true;
        const last = rowsOf("B").at(-1);
        await page.click("#backB");
        const back = await waitFor(page, s => s.view === "task" && s.loaded > 0, "volta a parte B");
        check(back.count === `${nB} / ${nB}` && sameSelection(back.selection, last), "volta a parte B: " + JSON.stringify([back.count, back.selection]));
        await page.key("Enter");
        last.confirmations += 1;
        await waitFor(page, s => s.view === "done", "fim de novo");
      }
      break;
    }
    state = await waitFor(page, s => s.view === "task" && s.loaded > 0 && s.src !== previous, "imagem nova");
    const part = PARTS[state.stage], image = imageOf(state.src);
    seen[part] += 1;

    if (part === "treino" && seen.treino === 1) {  // o treino nao volta
      check(state.prev && !state.prev.visible && !state.first && !state.resume, "navegacao visivel no treino");
      await page.key("ArrowLeft");
      check((await page.eval(PROBE)).src === state.src, "← mudou de imagem no treino");
    }
    if (part === "A" && seen.A === 1) {  // sem volta na primeira; Enter sem resposta e sem confianca nao avanca
      check(state.prev.visible && state.prev.disabled && !state.first, "Anterior ativo na primeira imagem da parte A");
      await page.key("ArrowLeft");
      await page.key("Enter");
      await page.key("s");
      await page.key("Enter");
      const after = await page.eval(PROBE);
      check(after.src === state.src && after.count === state.count, "parte A avancou sem confianca ou voltou da primeira");
    }
    if (part === "A" && seen.A === 6) {  // volta tres, muda e grava; muda a seguinte sem gravar e volta; retoma
      const [r3, r4] = [rowsOf("A")[2], rowsOf("A")[3]];
      for (let i = 0; i < 3; i++) await page.key("ArrowLeft");
      let at = await waitFor(page, s => s.count === `3 / ${nA}` && s.loaded > 0, "volta tres imagens");
      check(sameSelection(at.selection, r3) && /^Já respondida/.test(at.note) && !at.prev.disabled && at.resume && at.first,
        "revisita na parte A: " + JSON.stringify([at.selection, at.note]));
      const answer = r3.answer === "yes" ? "no" : "yes", confidence = r3.confidence === "1" ? "3" : "1";
      await page.key(answer === "yes" ? "s" : "n");
      await page.key(confidence);
      check(/^Mudança não gravada/.test((await page.eval(PROBE)).note), "sem aviso de mudanca nao gravada");
      await page.key("Enter");
      Object.assign(r3, { answer, confidence });
      r3.changes += 1; r3.confirmations += 1;
      at = await waitFor(page, s => s.count === `4 / ${nA}` && s.loaded > 0, "quarta imagem");
      check(sameSelection(at.selection, r4), "quarta imagem sem a resposta guardada");
      await page.key(r4.answer === "yes" ? "n" : "s");
      await page.key("ArrowLeft");
      at = await waitFor(page, s => s.count === `3 / ${nA}` && s.loaded > 0, "volta para a terceira");
      check(sameSelection(at.selection, r3), "terceira sem a resposta mudada");
      await page.click("#resume");
      at = await waitFor(page, s => s.count === `6 / ${nA}` && s.loaded > 0, "volta para onde parou");
      check(at.selection.answer === "" && !at.resume && at.note === "", "imagem da vez nao esta vazia depois de Ir para onde parei");
    }
    if (part === "B" && seen.B === 1) {  // a parte A fechou; o guia aparece; "sim", tipo, "nao", "sim" apaga o tipo
      check(state.prev.visible && state.prev.disabled && state.guide, "parte A aberta ou guia ausente na parte B");
      await page.key("ArrowLeft");
      check((await page.eval(PROBE)).src === state.src, "← voltou para a parte A");
      await page.key("s");
      await page.click(`#types button[data-v="${types[0]}"]`);
      await page.key("n");
      const hidden = await page.eval(PROBE);
      await page.key("s");
      const again = await page.eval(PROBE);
      check(!hidden.typeVisible, "tipo de RFI visivel depois do nao");
      check(again.selectedTypes === 0 && again.nextDisabled, "o nao nao apagou o tipo de RFI escolhido antes");
    }
    if (part === "B" && seen.B === 8) {  // "Inicio da parte": muda a primeira da parte B; retoma
      const r1 = rowsOf("B")[0];
      await page.click("#first");
      const at = await waitFor(page, s => s.count === `1 / ${nB}` && s.loaded > 0, "inicio da parte B");
      check(sameSelection(at.selection, r1) && at.typeVisible === (r1.answer === "yes"), "primeira da parte B sem a resposta guardada");
      await flipPartB(page, r1, types, pick);
      await waitFor(page, s => s.count === `2 / ${nB}` && s.loaded > 0, "segunda da parte B");
      await page.click("#resume");
      await waitFor(page, s => s.count === `8 / ${nB}` && s.loaded > 0, "volta para onde parou na parte B");
    }

    let row;
    if (part === "B" && seen.B === 2) {  // "sim", tipo, "nao", Enter: a linha sai sem tipo
      await page.key("s");
      await page.click(`#types button[data-v="${pick(types)}"]`);
      await page.key("n");
      await page.key("1");  // a confianca nao existe na parte B
      row = newRow(rater, part, image, "no", "", "", "2");
    } else {
      row = await answerRandom(page, part, image, rater, types, pick, "2");
      if (part === "B") await page.key("2");
    }
    if (part === "treino") {
      await page.key("Enter");
      const feedback = await page.eval(PROBE);
      check(/^(Certo|Errado)/.test(feedback.feedback) && feedback.src === state.src, "treino sem feedback");
      if (seen.treino === 3) {  // depois do feedback, a resposta nao muda
        await page.key(row.answer === "yes" ? "n" : "s");
        await page.key(row.confidence === "1" ? "3" : "1");
      }
    }
    await page.key("Enter");
    expected.push(row);
    previous = state.src;

    if (restartAt.has(expected.length)) {  // fecha o navegador inteiro e reabre a pagina
      await browser.close();
      ({ browser, page } = await open(profile, url));
      const reopened = await page.eval(PROBE);
      check(reopened.view === "intro" && reopened.start === `Continuar (${expected.length} de ${total})`,
        `reabertura depois de ${expected.length}: ` + JSON.stringify([reopened.view, reopened.start]));
      await page.click("#start");
      previous = null;
    }
  }

  const got = await download(browser, page, downloads, rater);
  check(got.state === "completed", "download " + got.state);
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
  const order = part => rowsOf(part).map(r => r.image);
  check(JSON.stringify(order("A")) === JSON.stringify(config.partA), "ordem da parte A");
  check(JSON.stringify(order("B")) === JSON.stringify(config.partB), "ordem da parte B");
  check(expected.length === total, `respostas ${expected.length}, etapas ${total}`);
  const compared = compareCsv(got.csvPath, expected, rater, check, findings);
  console.log(JSON.stringify({
    mode: "run", rater, steps: expected.length, csv_rows: compared.rows, mismatched_rows: compared.mismatches,
    screens_between_parts: screens, changed: expected.filter(r => r.changes > 0).map(r => r.part + ":" + r.image),
    check: compared.check, findings,
  }, null, 2));
  if (findings.length) process.exitCode = 1;
}

async function migrate(folder, newPage, work) {
  const config = readConfig(folder);
  const rater = config.rater, total = config.training.length + config.partA.length + config.partB.length;
  const nB = config.partB.length, inB = 15, v1Steps = config.training.length + config.partA.length + inB;
  const types = config.rfiTypes.map(t => t.value);
  const profile = path.join(work, "profile_migrate_" + rater), downloads = path.join(work, "downloads_migrate_" + rater);
  fs.rmSync(profile, { recursive: true, force: true });
  const url = pathToFileURL(path.resolve(folder, "index.html")).href;
  const rng = mulberry32(20261101), pick = values => values[Math.floor(rng() * values.length)];
  const expected = [], findings = [];
  const check = (ok, message) => { if (!ok) findings.push(message); };

  // versao 1: treino, parte A e as primeiras imagens da parte B
  let { browser, page } = await open(profile, url);
  let state = await page.eval(PROBE);
  check(state.prev === null, "a pagina de partida nao e a versao 1");
  await page.click("#start");
  let previous = null;
  while (expected.length < v1Steps) {
    state = await page.eval(PROBE);
    if (state.view === "between") { await page.key("Enter"); continue; }
    state = await waitFor(page, s => s.view === "task" && s.loaded > 0 && s.src !== previous, "imagem nova na versao 1");
    const part = PARTS[state.stage];
    const row = await answerRandom(page, part, imageOf(state.src), rater, types, pick, "1");
    if (part === "treino") await page.key("Enter");
    await page.key("Enter");
    expected.push(row);
    previous = state.src;
  }
  await browser.close();

  // versao 2 no mesmo caminho: continua de onde parou
  fs.copyFileSync(newPage, path.join(folder, "index.html"));
  ({ browser, page } = await open(profile, url));
  state = await page.eval(PROBE);
  check(state.view === "intro" && state.start === `Continuar (${v1Steps} de ${total})`, "versao 2: " + JSON.stringify([state.view, state.start]));
  await page.click("#start");
  state = await waitFor(page, s => s.view === "task" && s.loaded > 0, "imagem na versao 2");
  check(state.count === `${inB + 1} / ${nB}` && state.prev.visible && !state.prev.disabled && state.guide, "versao 2 parou em " + state.count);
  const r1 = expected.find(r => r.part === "B");
  await page.click("#first");
  const at = await waitFor(page, s => s.count === `1 / ${nB}` && s.loaded > 0, "inicio da parte B na versao 2");
  check(sameSelection(at.selection, r1), "resposta da versao 1 nao aparece na versao 2: " + JSON.stringify(at.selection));
  await flipPartB(page, r1, types, pick);
  await page.click("#resume");
  previous = null;
  while (true) {
    state = await page.eval(PROBE);
    if (state.view === "done") break;
    state = await waitFor(page, s => s.view === "task" && s.loaded > 0 && s.src !== previous, "imagem nova na versao 2");
    expected.push(await answerRandom(page, "B", imageOf(state.src), rater, types, pick, "2"));
    await page.key("Enter");
    previous = state.src;
  }
  const got = await download(browser, page, downloads, rater);
  check(got.state === "completed", "download " + got.state);

  // um estado que nao bate com a pagina (duas imagens trocadas) e recusado
  await page.eval(`(() => {
    const key = "frb_rotulacao_${rater}", s = JSON.parse(localStorage.getItem(key));
    [s.answers[11].image, s.answers[12].image] = [s.answers[12].image, s.answers[11].image];
    localStorage.setItem(key, JSON.stringify(s));
  })()`);
  await page.goto(url);
  state = await page.eval(PROBE);
  check(state.mismatch && state.startDisabled && state.view === "intro", "estado trocado aceito: " + JSON.stringify([state.mismatch, state.startDisabled, state.view]));
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });

  const compared = compareCsv(got.csvPath, expected, rater, check, findings);
  console.log(JSON.stringify({
    mode: "migrate", rater, v1_steps: v1Steps, steps: expected.length, csv_rows: compared.rows,
    mismatched_rows: compared.mismatches, check: compared.check, findings,
  }, null, 2));
  if (findings.length) process.exitCode = 1;
}

// Telas para as capturas: respostas ja dadas, partes ja comecadas e o que fazer antes da captura.
const SCREENS = [
  { name: "treino_feedback", answered: 0, started: {}, keys: ["s", "3", "Enter"] },
  { name: "parteA", answered: 10, started: { A: true }, keys: ["s", "2"] },
  { name: "parteA_revisita", answered: 20, started: { A: true }, keys: ["ArrowLeft", "ArrowLeft", "s", "3"] },
  { name: "entre_A_B", answered: 310, started: { A: true }, keys: [] },
  { name: "parteB", answered: 320, started: { A: true, B: true }, keys: ["s"], type: true },
  { name: "parteB_revisita", answered: 330, started: { A: true, B: true }, keys: ["ArrowLeft", "s"], type: true },
  { name: "fim", answered: 510, started: { A: true, B: true }, keys: [] },
];
const SIZES = [[1920, 1080], [1920, 970], [1440, 790], [1366, 657], [1280, 720], [1024, 700]];

function savedState(config, answered, started) {
  const steps = [
    ...config.training.map(t => ({ part: "treino", id: t.id })),
    ...config.partA.map(id => ({ part: "A", id })),
    ...config.partB.map(id => ({ part: "B", id })),
  ];
  const answers = steps.slice(0, answered).map(s => ({
    rater: config.rater, part: s.part, image: s.id, answer: "no", confidence: s.part === "B" ? "" : "1", rfi_type: "",
    response_ms: 1000, time_utc: "2026-11-01T12:00:00.000Z",
  }));
  return { index: answered, answers, started };
}

async function shots(folder, work) {
  const config = readConfig(folder);
  const profile = path.join(work, "profile_shots"), out = path.join(work, "shots");
  fs.rmSync(profile, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  const url = pathToFileURL(path.resolve(folder, "index.html")).href;
  const browser = await Browser.launch(profile);
  const report = [];
  for (const dark of [false, true]) {
    for (const [width, height] of dark ? [[1440, 790]] : SIZES) {
      const page = await browser.page(width, height);
      if (dark) await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: "dark" }] });
      for (const screen of SCREENS) {
        await page.goto(url);
        const saved = savedState(config, screen.answered, screen.started);
        await page.eval(`localStorage.setItem("frb_rotulacao_${config.rater}", ${JSON.stringify(JSON.stringify(saved))})`);
        await page.goto(url);
        if (screen.answered < 510) {
          await page.click("#start");
          await waitFor(page, s => s.view !== "task" || s.loaded > 0, "imagem");
        }
        for (const key of screen.keys) await page.key(key);
        if (screen.type) await page.click(`#types button[data-v="${config.rfiTypes[0].value}"]`);
        await waitFor(page, s => s.view !== "task" || s.loaded > 0, "imagem");
        const state = await page.eval(PROBE);
        const [scrollW, scrollH, innerW, innerH] = state.page;
        const name = `${screen.name}_${width}x${height}${dark ? "_escuro" : ""}.png`;
        const { data } = await page.send("Page.captureScreenshot", { format: "png" });
        fs.writeFileSync(path.join(out, name), Buffer.from(data, "base64"));
        report.push({
          name, view: state.view, img: state.img, fits: scrollW <= innerW && scrollH <= innerH,
          nextVisible: state.view !== "task" || state.nextBottom <= innerH, page: state.page,
        });
      }
      await browser.send("Target.closeTarget", { targetId: page.targetId });
    }
  }
  await browser.close();
  fs.rmSync(profile, { recursive: true, force: true });
  console.log(JSON.stringify(report.filter(r => !r.fits || !r.nextVisible), null, 1));
  console.log(`${report.length} capturas; ${report.filter(r => !r.fits || !r.nextVisible).length} com rolagem ou sem o botao Avancar`);
  if (report.some(r => !r.fits || !r.nextVisible)) process.exitCode = 1;
}

const [mode, ...rest] = process.argv.slice(2);
const usage = "uso: node tests/e2e_labeling_tool.mjs run|shots <pasta do avaliador> <pasta de trabalho>\n"
  + "     node tests/e2e_labeling_tool.mjs migrate <pasta com a pagina da versao 1> <index.html da versao 2> <pasta de trabalho>";
if (!(["run", "shots"].includes(mode) && rest.length === 2) && !(mode === "migrate" && rest.length === 3)) {
  console.error(usage);
  process.exit(2);
}
await ({ run, shots, migrate })[mode](...rest);
