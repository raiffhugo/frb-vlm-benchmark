# R0 — reanálises sem GPU (itens 1–7)

Protocolo: `article/plano_revisao_referee_v4.md`, §10.2. Rodado em 13/10/2026 com
`uv run python -m revision_v4.r0_reanalysis`. Todos os números estão em `r0_results.json`;
`operating_points.csv` e `sigma_bins.csv` são as tabelas prontas. Intervalos são Wilson 95%;
diferenças pareadas usam Newcombe (método 10) e bootstrap de pares (10 000, seed 42); McNemar é
exato. O item 8 (largura e ocupação espectral dos bursts reais) fica em `r0_morphology/`.

**Limiares casados (itens 1–2):** até o E1 ficar pronto, o limiar do SwinYNet é escolhido em
um de 2 folds estratificados por classe de origem (seed 42) e aplicado no outro. As duas
metades avaliadas fora do fold somam as 2000 amostras. Com o E1, a escolha passa a ser feita
num conjunto e aplicada no outro, como manda o protocolo.

## C2 — diferença de acurácia a 0,5 (item 3)

| | Diferença | Newcombe 95% | Bootstrap 95% | McNemar exato |
|---|---:|---|---|---|
| E2B − SwinYNet | +1,20 pp | [−0,25; +2,66] | [−0,25; +2,65] | p = 0,119 (121 × 97) |
| E4B − SwinYNet | −2,80 pp | [−4,50; −1,11] | [−4,50; −1,10] | p = 0,0015 (122 × 178) |

O qui-quadrado com correção de continuidade da v3 dá os mesmos p (0,119 e 0,0015).

## C1 — pontos de operação casados (itens 1–2)

No ponto de 0,5 de cada VLM, com o limiar do SwinYNet escolhido fora do fold:

| | VLM: recall / FPR / acurácia | SwinYNet com FPR casada | SwinYNet com recall casado |
|---|---|---|---|
| E2B (nível 0,85) | 0,905 / 0,024 / 0,9405 | recall 1,000, FPR 0,025, acc 0,9875; VLM − Swin = −4,70 pp [−5,89; −3,58], p = 2·10⁻¹⁶ | recall 0,930, FPR 0,000, acc 0,9650; −2,45 pp [−3,81; −1,11], p = 4·10⁻⁴ |
| E4B (nível 0,75) | 0,824 / 0,023 / 0,9005 | recall 1,000, FPR 0,023, acc 0,9885; −8,80 pp [−10,23; −7,46] | recall 0,930, FPR 0,000, acc 0,9650; −6,45 pp [−8,04; −4,90] |

Os limiares escolhidos ficam entre 0,915 e 0,958 para a FPR e em 0,9999999 para o recall. É a
saturação do score do SwinYNet.

**Híbridos (VLM a 0,5 combinado com o SwinYNet no mesmo limiar):**
- Com o SwinYNet no limiar de **máxima acurácia** escolhido fora do fold, a acurácia sozinho é
  0,9975 e **nenhum híbrido o supera**: E2B OR −0,95 pp (p = 5·10⁻⁴), AND −4,75 pp; E4B OR
  −0,90 pp, AND −8,80 pp.
- O OR só supera o SwinYNet quando ele é mantido no recall do VLM (limiar 1 − 10⁻⁷, recall
  0,930): +2,30 pp (E2B) e +2,20 pp (E4B). Ali o VLM recupera FRBs que o limiar extremo descarta.
- A 0,5 fixo, o AND do E2B (veto do VLM) dá +1,30 pp (p = 0,089) sobre o SwinYNet. É a
  "complementaridade" da v3, e ela some quando o limiar do SwinYNet é ajustado.

## Item 4 — Wilson nas Tabelas 5 e 9–12

- Tabela 5 (0,5): acurácia E2B 0,9405 [0,9293; 0,9500], E4B 0,9005 [0,8866; 0,9129], SwinYNet
  0,9285 [0,9164; 0,9390].
- Tabela 9: FPR em RFI de 0,048 (E2B), 0,046 (E4B) e 0,246 (SwinYNet), de 500 cada; em NOISE de
  0, 0 e 0,040. Também por subtipo de RFI.
- Tabela 10, acurácia em 200:

  | | completo | lean | lean+reason | multiclasse |
  |---|---|---|---|---|
  | E2B | 0,930 [0,886; 0,958] | 0,805 [0,745; 0,854] | 0,930 | 0,930 |
  | E4B | 0,910 [0,862; 0,942] | 0,910 | 0,930 | 0,945 [0,904; 0,969] |

- Tabelas 11 e 12: em `r0_results.json` (`item4_wilson`). A 12 reproduz os intervalos da v3.

## Item 5 — recall por bin de σ (`sigma_bins.csv`)

Os simulados são os 549 com sweep contido e os reais, os 600 bursts. Entre 5σ e 30σ há um único
simulado, no bin de 8–15σ. Onde há dados dos dois lados (≥ 30σ), o recall real é maior. O E2
preenche a faixa que falta.

## Item 6 — McNemar entre variantes de prompt (Tabela 10)

- E2B: lean − completo = −12,5 pp (p < 10⁻⁴); lean+reason − completo = 0,0 pp (p = 1);
  lean+reason − lean = +12,5 pp (p < 10⁻⁴).
- E4B: lean+reason − completo = +2,0 pp (p = 0,22); lean+reason − lean = +2,0 pp (p = 0,125);
  multiclasse − completo = +3,5 pp (p = 0,016).

No E2B, pôr o campo `reason` no prompt lean recupera os 12,5 pp que o lean perde. Isso é
consistente com a frase causal da v3, mas não a prova, por dois motivos: a variante foi escolhida
nestas mesmas 200 imagens, e o ganho pode vir do comprimento ou do formato da saída, e não do
raciocínio. Quem decide é o held-out do E1 com o E7 (filler neutro do mesmo comprimento).

## Item 7 — multiclasse fora das 400 de desenvolvimento

O subconjunto de desenvolvimento tem 200 RFI e 200 NOISE (nenhuma FRB). Acurácia:

| | 3000 | 400 de desenvolvimento | 2600 restantes |
|---|---|---|---|
| E2B | 80,1% | 76,2% | 80,7% |
| E4B | 86,0% | 83,0% | 86,5% |

O recall de RFI é o mesmo dentro e fora (E2B 0,53; E4B 0,66). Isto é uma análise de
sensibilidade, não held-out: a revisão de 03/07 foi motivada pelo resultado nas 3000. O número
held-out vem do E1.
