# R0, item 8 — largura e ocupação espectral dos 600 bursts reais

Protocolo: `article/plano_revisao_referee_v4.md`, §10.2 (definição registrada na §10.12 antes da
rodada). Rodado em 13/10/2026 com `uv run python -m revision_v4.r0_morphology --workers 16`
(~6 min).

Para cada burst, o array decimado que virou PNG é reconstruído com a janela do manifesto do
artigo, e `check_burst_visibility.burst_morphology` mede, no DM do catálogo:
- o pico dedispersado, que **reproduz o `peak_sigma` da v3 nos 600 bursts**;
- a largura do boxcar vencedor (bins de 1,97 ms);
- a ocupação espectral: a fração dos canais vivos na faixa contígua de 16 blocos que maximiza o
  S/N on-pulse. Abaixo de ~10σ ela é dominada pelo ruído, por isso só entra acima de 10σ.

`burst_morphology_real.csv` tem uma linha por burst; `morphology_summary.json` tem o recall
estratificado, com intervalos de Wilson.

## Resultado (descritivo)

Recall do E2B (o do E4B segue o mesmo padrão), dentro do mesmo bin de σ:

| σ | ≤ 4 ms | 8–16 ms | banda limitada (< 0,5) | banda larga (≥ 0,5) |
|---|---|---|---|---|
| 8–15 | 0,27 [0,20; 0,36], n = 103 | 0,08 [0,04; 0,14], n = 104 | 0,26 [0,16; 0,40], n = 46* | 0,17 [0,10; 0,27], n = 77* |
| 15–30 | 0,74 [0,58; 0,86], n = 35 | 0,64 [0,49; 0,77], n = 42 | 0,86 [0,71; 0,94], n = 36 | 0,54 [0,39; 0,68], n = 41 |
| ≥ 30 | 1,00, n = 35 | 1,00, n = 40 | 1,00, n = 40 | 1,00, n = 35 |

\* só os bursts acima de 10σ.

Com o σ fixo, os bursts estreitos e os de banda limitada são mais detectados. O padrão se mantém
dentro de cada fonte. Em FRB 20121102, com 8–15σ, os estreitos dão 25/94 e os largos 6/92; com
15–30σ, banda limitada dá 14/17 e banda larga 19/33. Em FRB 20201124, com 15–30σ, dá 15/17 contra
3/8.

**Leitura:** o σ é um filtro casado, que integra o sinal ao longo da largura e da banda. A imagem
mostra o brilho por pixel. Com o mesmo σ, um burst estreito ou de banda limitada concentra o sinal
em menos pixels e aparece com mais contraste. Isso é um argumento para C3 e C4: o que governa o
recall é o contraste por pixel no PNG, e não o σ dedispersado. O E2 (largura e banda controladas
na injeção) e o E5 (CNN e humanos nas mesmas imagens) testam isso diretamente.

Ressalvas: a largura vem de uma grade de boxcars com fator 2, e o espalhamento alarga os bursts.
As classes de banda misturam fontes (a mediana da ocupação é 0,61 em FRB 20121102 e 0,39 em
FRB 20201124), o que motivou a checagem por fonte acima.
