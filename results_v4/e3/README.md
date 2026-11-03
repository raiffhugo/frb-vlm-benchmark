# E3 — tiling com agregação por arquivo

Protocolo: `article/plano_revisao_referee_v4.md`, §10.5; definições na §10.12 (19/10).
Código: `prepare-real --sliding-windows` e `revision_v4/e3_tiling.py`.

## Verificação do P0 dos dados reais

Antes do E3, o P0 do artigo foi refeito nas 1600 janelas do artigo e comparado com
`dataset_real_full`: **1600/1600 PNGs idênticos byte a byte**, com todos os campos do manifesto
iguais (`p0_repro.json`). É o E0 dos dados reais.

## Janelas

9600 janelas de 2,013 s, 6 por arquivo, com inícios em 0, 1, 2, 3, 4 e 4,027 s
(`image_manifest.jsonl`, cópia versionada de `dataset_real_e3/metadata/`). Dois arquivos
negativos (`00324_neg` e `00853_neg`) têm sub-integrações com DAT_SCL corrompido, e algumas
janelas cegas caem nelas. Essas janelas ficam, porque uma busca cega real as veria, e a análise
também sai sem os dois arquivos.

## Comandos

```bash
uv run python run_pipeline.py prepare-real --input-manifest fast_frex/real_manifest_full.csv \
  --output-dir /var/tmp/frb_svlm_real_p0 --window-seconds 2.0 --time-bins 1024 --freq-bins 256 \
  --zap-sigma 3.0 --edge-channels 410 --toa-ref top --negative-window random --seed 42 \
  --reader astropy --cmap viridis --normalization percentile --percentile-low 1 \
  --percentile-high 99 --width 1024 --height 768 --workers 32
uv run python -m revision_v4.e3_tiling check-p0 --workdir /var/tmp/frb_svlm_real_p0
uv run python run_pipeline.py prepare-real --input-manifest fast_frex/real_manifest_full.csv \
  --output-dir dataset_real_e3 --sliding-windows --window-stride-seconds 1.0 <mesmos parâmetros> \
  --workers 48
uv run python -m revision_v4.gpu_runs e3 --model e4b --gpu 1   # e o mesmo com e2b
uv run python -m revision_v4.e3_tiling analyze   # e3_results.json; com --models e2b, só o E2B
```

## Resultado (20/10)

Rodadas `e2b_binary_full/` e `e4b_binary_full/`: 9600/9600 cada, 0 falhas, sem alterações locais.
O E2B rodou de uma vez no commit `899bec5`. O E4B foi interrompido no registro 8280, quando a
sessão que o lançou caiu (20/10, ~0h48). A fila da GPU 1 o retomou com `classify --resume`, que
recusa a retomada se o código de inferência, o prompt, o modelo ou o manifesto mudaram. Os 8280
primeiros registros têm o commit `c84ce2b`, e os 1320 seguintes, o `57820a5` com a marca da
retomada. Entre os dois commits, os arquivos de inferência (`models.py`, `parser.py`,
`prompt.py`, `pyproject.toml` e `uv.lock`) são idênticos, e as seções de manifesto, prompt, modelo
e classificador da proveniência são iguais nos registros 8280 e 8281. No classificador, só mudou
a orquestração (commit `5ac5634`: a própria retomada e a gravação linha a linha); a função que
classifica cada imagem não mudou. A resposta de cada imagem não depende das que vieram antes na
mesma rodada: o P0 do E6 (200 imagens) reproduz o do artigo (1600 ou 2000) em 200/200, e o V1 e o
V2 do E7 reproduzem o E1 em 400/400. Análise final em `e3_results.json`; o `e3_results_e2b.json` é a análise parcial de
19/10, com o mesmo resultado para o E2B.

O score do arquivo é o máximo entre as 6 janelas, e o arquivo é positivo se esse máximo for
≥ 0,5 (Wilson 95%):

| Métrica | E2B | E4B |
|---|---|---|
| Recall por arquivo, qualquer janela | 0,327 [0,290; 0,365] (196/600) | 0,297 [0,262; 0,334] (178/600) |
| Recall com casamento temporal (sweep da banda inteira) | 0,243 [0,211; 0,279] (146) | 0,222 [0,190; 0,257] (133) |
| … só nos 490 bursts que têm uma janela com o sweep inteiro | 0,298 [0,259; 0,340] | 0,271 [0,234; 0,312] |
| Recall com casamento temporal, sweep só na banda visível (sensibilidade) | 0,283 [0,249; 0,321] (170) | 0,258 [0,225; 0,295] (155) |
| Recall na janela centrada do artigo (mesmos 600 bursts) | 0,285 [0,250; 0,322] (171) | 0,270 [0,236; 0,307] (162) |
| FPR por arquivo | 0,042 [0,031; 0,056] (42/1000) | 0,021 [0,014; 0,032] (21/1000) |
| FPR por janela | 0,0080 [0,0060; 0,0106] (48/6000) | 0,0042 [0,0028; 0,0061] (25/6000) |

Os bursts com o sweep da banda inteira em alguma janela são 490 de 600 (0,817 [0,784; 0,846]).
Com o sweep da banda visível, são todos os 600.

Pareado com a janela centrada do artigo (Newcombe pareado, McNemar exato; discordantes como "só
no primeiro × só no segundo"):

| | E2B | E4B |
|---|---|---|
| Recall, qualquer janela − centrada | +4,2 pp [+2,0; +6,4], p = 3 × 10⁻⁴ (36 × 11) | +2,7 pp [+0,6; +4,7], p = 0,017 (28 × 12) |
| Recall, casamento temporal − centrada | −4,2 pp [−6,5; −1,9], p = 6 × 10⁻⁴ (13 × 38) | −4,8 pp [−7,1; −2,6], p = 4 × 10⁻⁵ (10 × 39) |
| FPR, por arquivo − janela do artigo | +4,0 pp [+2,9; +5,4], p = 2 × 10⁻¹¹ (41 × 1) | +1,6 pp [+0,7; +2,7], p = 9 × 10⁻⁴ (19 × 3) |

A busca cega por arquivo acha um pouco mais de bursts que a janela centrada (+4,2 e +2,7 pp),
mas a FPR por arquivo sobe de 0,002 para 0,042 no E2B e de 0,005 para 0,021 no E4B: são 6
chances por arquivo. No E2B, os FP se espalham pelas 6 posições, de 5 a 10 por 1000 janelas em
cada uma (descritivo), sem concentração nas bordas do arquivo; são 42 arquivos com FP, e 6 deles
têm mais de uma janela FP. Exigir que a janela contenha o sweep inteiro baixa o recall abaixo do
da janela centrada nos dois modelos. Parte das detecções vem de janelas com o sweep cortado, e
110 bursts não têm nenhuma janela com o sweep da banda inteira.

Por bin de σ, o ganho do tiling fica abaixo de 30σ (qualquer janela × centrada):

| Bin de σ | n | E2B | E4B |
|---|---|---|---|
| [0, 5) | 77 | 0,052 × 0,013 | 0,026 × 0,013 |
| [5, 8) | 163 | 0,068 × 0,037 | 0,055 × 0,031 |
| [8, 15) | 208 | 0,240 × 0,173 | 0,154 × 0,130 |
| [15, 30) | 77 | 0,727 × 0,688 | 0,779 × 0,701 |
| ≥ 30 | 75 | 1,000 × 1,000 | 1,000 × 1,000 |

Por fonte, o recall em qualquer janela é 0,221 (E2B) e 0,189 (E4B) no FRB 20121102A (n = 470);
0,704 e 0,680 no FRB 20201124A (n = 125); 0,800 nos dois no FRB 20180301A (n = 5).

**Sem os dois arquivos com DAT_SCL corrompido:** FPR por arquivo 0,040 (40/998) no E2B e 0,019
(19/998) no E4B; por janela, 0,0075 (45/5988) e 0,0038 (23/5988). Os dois arquivos davam 2 FP
por arquivo em cada modelo. O recall não muda, porque os dois são negativos. Nenhuma janela
ficou sem `frb_probability` válida.
