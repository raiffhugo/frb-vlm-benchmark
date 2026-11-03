# E2 — injeção-recuperação em fundos reais

Protocolo: `article/plano_revisao_referee_v4.md`, §10.4; definições na §10.12 (19/10).
Código: `revision_v4/injection.py` (pulso, injeção, calibração) e `revision_v4/e2_injection.py`.

## Desenho

- 1024 janelas de 2 s dos 1000 negativos do artigo: todos uma vez e mais 24 sorteados de novo,
  com início uniforme no arquivo (seed 20261013, fluxos independentes).
- Grade com 8 réplicas por célula: σ alvo ∈ {3, 5, 8, 12, 15, 20, 30, 50} × largura ∈ {1, 3, 10,
  30} ms × banda larga ou limitada × sem espalhamento ou τ = 5 ms em 1,25 GHz.
- O pulso entra no z-score por canal, entre `normalize_per_channel` e `block_average`. Daí em
  diante é o P0 do artigo. Cada janela também sai sem pulso, como controle pareado.
- O σ medido usa o mesmo diagnóstico dos bursts reais, no DM injetado.

## Verificações

| Verificação | Resultado |
|---|---|
| Caminho sem pulso × os 1000 PNGs negativos do artigo | **1000/1000 idênticos byte a byte** (`control_check.json`) |
| σ medido mediano por σ alvo (3, 5, 8, 12, 15, 20, 30, 50) | 3,68; 5,14; 7,79; 11,56; 14,30; 19,34; 28,64; 47,98 |
| Fundos com dado corrompido | 1 (`inj_0443`, DAT_SCL corrompido em `00324_neg`): sorteado de novo no mesmo arquivo (§10.12) |

No alvo 3, o piso de ruído do diagnóstico puxa o σ medido para cima. O maior σ de um controle
é 62,9, de um bloco de RFI de banda larga na borda da janela.

## Comandos

```bash
uv run python -m revision_v4.e2_injection plan --workers 32
uv run python -m revision_v4.e2_injection check-control --workers 40 --workdir /var/tmp/frb_svlm_e2_check
uv run python -m revision_v4.e2_injection render --workers 32
uv run python -m revision_v4.gpu_runs e2 --model e2b --gpu 2   # e o mesmo com e4b
uv run python -m revision_v4.e2_analysis analyze --models e4b   # sem --models, os dois
```

## Resultado (20/10)

Rodadas `e2b_binary_full/` e `e4b_binary_full/`: 2048/2048 cada, 0 falhas, commits `57820a5`
(E2B) e `340fcf0` (E4B) sem alterações locais, sem retomada. Análise final em
`e2_results.json`. O `e2_results_e4b.json` é a análise parcial de 19/10, com o mesmo resultado
para o E4B.

**Injeções e controles** (Wilson 95%):

| | E2B | E4B |
|---|---|---|
| Recall nas 1024 injeções | 0,417 [0,387; 0,447] | 0,401 [0,372; 0,432] |
| FPR nos 1024 controles | 0,0059 [0,0027; 0,0127] (6) | 0,0039 [0,0015; 0,0100] (4) |
| Injeções nos fundos dos controles sinalizados | 5 de 6 detectadas | 4 de 4 detectadas |

**Completeza por σ medido**, contra o P0 do artigo nos 600 bursts reais. A diferença E2 − real
usa Newcombe para amostras independentes, nos bins com ≥ 10 nos dois:

| Bin de σ | n E2 / real | E2B: E2 | E2B: real | E2B: E2 − real | E4B: E2 | E4B: real | E4B: E2 − real |
|---|---|---|---|---|---|---|---|
| [0, 5) | 176 / 77 | 0,023 | 0,013 | +1,0 pp [−4,9; +4,6] | 0,017 | 0,013 | +0,4 pp [−5,4; +3,8] |
| [5, 8) | 155 / 163 | 0,097 | 0,037 | **+6,0 pp [+0,4; +12,0]** | 0,077 | 0,031 | +4,7 pp [−0,4; +10,3] |
| [8, 15) | 270 / 208 | 0,252 | 0,173 | **+7,9 pp [+0,4; +15,0]** | 0,237 | 0,130 | **+10,7 pp [+3,7; +17,4]** |
| [15, 30) | 273 / 77 | 0,703 | 0,688 | +1,5 pp [−9,3; +13,7] | 0,678 | 0,701 | −2,4 pp [−13,1; +9,8] |
| [30, 60) | 150 / 25 | 0,987 | 1,000 | −1,3 pp [−4,7; +12,0] | 0,980 | 1,000 | −2,0 pp [−5,7; +11,4] |

Por σ alvo (3, 5, 8, 12, 15, 20, 30, 50; só descritivo): E2B 0,016; 0,055; 0,109; 0,250; 0,320;
0,633; 0,961; 0,992. E4B 0,008; 0,047; 0,078; 0,250; 0,305; 0,586; 0,938; 1,000.

**σ de 50% de completeza** (logístico em log₁₀ do σ medido; IC por bootstrap percentil das
injeções, 10 000 reamostras; nenhuma reamostra ficou sem ajuste):

| Injeções | E2B | E4B |
|---|---|---|
| Todas (1024) | 15,7 [14,9; 16,5] | 16,3 [15,5; 17,2] |
| Banda larga (512) | 19,1 [17,9; 20,4] | 19,0 [17,9; 20,3] |
| Banda limitada (512) | 12,7 [11,7; 13,8] | 13,8 [12,7; 15,0] |
| Sem espalhamento (512) | 16,7 [15,5; 17,9] | 17,5 [16,2; 18,8] |
| τ = 5 ms (512) | 14,8 [13,7; 15,9] | 15,3 [14,2; 16,4] |
| Largura 1 ms (256) | 17,6 [15,9; 19,5] | 18,8 [17,1; 20,9] |
| Largura 3 ms (256) | 16,2 [14,5; 17,8] | 16,7 [15,0; 18,5] |
| Largura 10 ms (256) | 14,5 [13,1; 15,9] | 15,2 [13,8; 16,6] |
| Largura 30 ms (256) | 14,8 [13,2; 16,7] | 14,9 [13,4; 16,6] |

Os dois modelos têm a mesma curva. No mesmo σ medido, detectam mais o pulso de banda limitada, o
espalhado e o largo. O protocolo não fixou teste entre níveis, então essas diferenças são
descritivas. Os ICs de banda larga e limitada não se sobrepõem em nenhum dos dois modelos.

**C3/C10.** A curva de injeção acompanha o recall real por bin nos dois modelos. As diferenças
significativas estão todas entre 5 e 15σ e vão sempre no mesmo sentido: as injeções são
detectadas de 6 a 11 pp mais vezes que os bursts reais (E2B em 5–8 e 8–15σ, E4B em 8–15σ). Essa
tabela entra no lugar do "agree to within a few points".
