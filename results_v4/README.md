# Resultados da revisão v4 (parecer ApJS AAS80842)

Protocolos congelados: `article/plano_revisao_referee_v4.md`, §10. Cada subpasta tem um README
com o resultado e os comandos, e um manifesto JSON com commit, versões e hashes das entradas.
Os resultados da v3 (`results_*` na raiz) não são sobrescritos.

| Pasta | Experimento | Estado |
|---|---|---|
| `e0/` | Reconstrução dos PNGs do benchmark e reprodução da rodada do artigo | Passou |
| `swinynet/` | Reinstalação do SwinYNet e reprodução de 100 scores | Passou |
| `r0/` | Reanálises sem GPU, itens 1–7 | Feito |
| `r0_morphology/` | R0, item 8: largura e ocupação espectral dos 600 bursts reais | Feito |
| `e1/` | Conjunto simulado held-out (seed 20260927) | Feito: SwinYNet e os dois VLMs com os 4 prompts, e a análise |
| `e2/` | Injeção-recuperação em fundos reais | Feito: σ₅₀ = 15,7 (E2B) e 16,3 (E4B); a curva acompanha o recall real, com +6 a +11 pp a favor das injeções em 5–15σ |
| `e3/` | Tiling com agregação por arquivo | Feito: recall por arquivo 0,327 e 0,297 (centrada 0,285 e 0,270); FPR por arquivo 0,042 e 0,021 |
| `e4/` | Ablação do pré-processamento (P1, P2) | Feito: o P2 não multiplica os FP em nenhum modelo (E2B 2 → 9, p = 0,065; E4B 5 → 1); o recall cai nas duas variantes. Auditoria humana (02/11): 62 de 200 negativos com RFI visível no consenso dos dois avaliadores, e nenhum VLM marca nenhum deles |
| `e5/` | CNN de referência e avaliadores humanos | Feito: a CNN (Δ de +78 a +85 pp em 5–15σ, robusto ao controle fora do burst) e os dois avaliadores (02/11: recall de 0,79 e 0,59 em 5–15σ, sem falso positivo) passam o critério do C4 contra os dois VLMs, o que dá a tese forte |
| `e6/` | Sensibilidade ao prompt | Feito: E2B estável nas paráfrases das simuladas; E4B sensível à versão curta; a morfologia relaxada não recupera o FRB 20121102A |
| `e7/` | Ablação do raciocínio (V1, V1o, V2–V5) | Feito: a frase causal não volta (o filler reproduz parte do ganho no E2B) |
| `e8/` | Renderização equivalente do simulado | Feito: recall 0,905 → 1,000 (E2B) e 0,824 → 0,999 (E4B); explica a diferença simulado × real nos dois |
| `selections/` | Amostras do E4, do E5, do E6 e do E7 | Sorteadas |

Código: `revision_v4/` (scripts por experimento e `stats.py` com os intervalos e testes do
protocolo), com testes em `tests/test_revision_*.py`.

## Se o servidor cair

As filas de GPU estão em `revision_v4/queues.py`. Depois de reiniciar a máquina, rode uma vez por
GPU, em segundo plano:

```bash
for g in 0 1 2 3; do setsid nohup uv run python -m revision_v4.queues run --gpu $g >> results_v4/queue_gpu$g.log 2>&1 < /dev/null & done
uv run python -m revision_v4.queues show   # o que já foi feito
```

Rodadas completas são puladas. As interrompidas continuam de onde pararam (`classify --resume`,
que recusa a retomada se o manifesto, o prompt, o modelo, a configuração ou o código de
inferência mudaram). Cada predição é gravada no disco na hora. O SwinYNet e o treino da CNN
recomeçam do zero. O `/var/tmp` desta máquina não é apagado no boot nem por idade.
