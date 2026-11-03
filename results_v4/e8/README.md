# E8 — renderização equivalente do benchmark simulado

Protocolo: `article/plano_revisao_referee_v4.md`, §10.10; regra na §10.12 (19/10).
Código: `revision_v4/e8_decimated.py`.

As 2000 amostras do binário re-renderizadas a partir do array da grade comum de
`check_burst_visibility_sim.load_spectrum`, o mesmo em que o artigo mediu o σ dos bursts
simulados: z-score robusto por canal, fator 10 no tempo (1,966 ms, como no real), 256 canais e
`flatten_rows`. A renderização é a do artigo. Os FITS simulados têm 10173 amostras, então a
imagem fica em 1017 × 256, e não em 1024 × 256. Mesmos `sample_id` do binário.

```bash
uv run python -m revision_v4.e8_decimated render --workers 40
uv run python -m revision_v4.gpu_runs e8 --model e2b --gpu 2   # e o mesmo com e4b
uv run python -m revision_v4.e8_decimated analyze --models e2b   # sem --models, os dois
```

## Resultado (20/10)

Rodadas `e2b_binary_full/` e `e4b_binary_full/`: 2000/2000 cada, 0 falhas, commits `fa07bf0`
(E2B) e `57820a5` (E4B) sem alterações locais, sem retomada. Análise final em `e8_results.json`;
o `e8_results_e2b.json` é a análise parcial de 19/10, com o mesmo resultado para o E2B.

**Decimado × original** (as mesmas 2000 amostras; Newcombe pareado, McNemar exato):

| | E2B: original | E2B: decimado | E2B: decimado − original | E4B: original | E4B: decimado | E4B: decimado − original |
|---|---|---|---|---|---|---|
| Recall de FRB | 0,905 | 1,000 [0,996; 1,000] | +9,5 pp [+7,8; +11,5], p < 10⁻²⁸ (95 × 0) | 0,824 | 0,999 [0,994; 1,000] | +17,5 pp [+15,2; +20,0], p < 10⁻⁵² (175 × 0) |
| FPR | 0,024 | 0,027 [0,019; 0,039] | +0,3 pp [−0,7; +1,3], p = 0,66 | 0,023 | 0,011 [0,006; 0,020] | −1,2 pp [−2,1; −0,5], p = 0,002 |
| Acurácia | 0,9405 | 0,9865 | +4,6 pp [+3,6; +5,7], p < 10⁻¹⁸ | 0,9005 | 0,9940 | +9,4 pp [+8,1; +10,7], p < 10⁻⁵⁴ |

Por bin de σ, o decimado tem recall de 0,998 a 1,000 em todos os bins, nos dois modelos. As
diferenças significativas estão em [0, 5) (n = 434; original 0,878 no E2B e 0,758 no E4B), em
[60, 100) (n = 34; 0,50 e 0,38) e em [100, 300) (n = 261; 0,923 e 0,828). Os bins de 5 a 60σ têm
de 3 a 7 bursts.

Isso confirma a hipótese da §1d do plano. O caminho original pega uma amostra por pixel; o
decimado faz a média de 10, o que reduz o ruído por pixel, e o burst, mais largo que o bin de
1,966 ms, mantém a amplitude. Dos 434 bursts abaixo de 5σ, 433 têm o sweep fora da janela
(`sweep_contained` = 0). O σ medido na janela não diz o quanto o sweep aparece na imagem. Por
exemplo, `sample_000023` (σ = 2,42, DM 844) é um traço fraco no original e uma linha nítida no
decimado; o E2B o perde no original e o acha no decimado.

**Simulado × real (C3/C10).** Só os 549 bursts simulados de sweep contido, nos bins com ≥ 10
bursts nos dois conjuntos, contra o P0 do artigo nos 600 reais (Newcombe, amostras
independentes):

| Bin | n simulado / real | E2B: original − real | E2B: decimado − real | E4B: original − real | E4B: decimado − real |
|---|---|---|---|---|---|
| [60, 100) | 32 / 14 | **−50,0 pp [−66,4; −23,0]** | 0,0 pp [−10,7; +21,5] | **−62,5 pp [−77,1; −34,9]** | 0,0 pp [−10,7; +21,5] |
| [100, 300) | 256 / 25 | −7,8 pp [−11,8; +5,8] | 0,0 pp [−1,5; +13,3] | **−17,6 pp [−22,7; −3,6]** | 0,0 pp [−1,5; +13,3] |
| ≥ 300 | 253 / 11 | 0,0 pp [−1,5; +25,9] | 0,0 pp [−1,5; +25,9] | 0,0 pp [−1,5; +25,9] | 0,0 pp [−1,5; +25,9] |

Pela regra da §10.12, o E8 explica a diferença simulado × real nos dois modelos. No caminho
original, os simulados de sweep contido tinham recall menor que os reais no mesmo σ (um bin
significativo no E2B, dois no E4B). No decimado, nenhuma diferença é significativa. Abaixo de
60σ a regra não tem o que testar: os simulados de sweep contido são 1 abaixo de 5σ, 1 entre 8 e
15σ e 6 entre 30 e 60σ. O E2 cobre essa faixa.
