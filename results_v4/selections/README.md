# Amostras do E4, do E5, do E6 e do E7

Sorteadas por `revision_v4/selections.py` (seed 20261013, fluxos independentes; regras na
§10.12, 19/10), a partir de `dataset_real_full` e das faixas de σ do diagnóstico do artigo
(<5, 5–8, 8–15, 15–30, ≥30):

| Arquivo | Uso | Composição |
|---|---|---|
| `e6_r200.jsonl` | E6, dados reais | 20 positivos por faixa e 100 negativos |
| `e6_s200.jsonl` | E6, simulado | As 200 imagens da Tabela 10 (seed 42), conferidas contra as predições dela |
| `e5_humans.jsonl` | E5, avaliadores (parte A) | 25/45/60/45/25 positivos por faixa e 100 negativos |
| `e4_audit.jsonl` | Auditoria do E4 (parte B) | 200 negativos, disjuntos dos negativos do E5 |
| `e7_heldout400.jsonl` | E7, simulado (held-out do E1) | 200 FRB, 20 por subtipo de RFI e 100 NOISE do binário do E1, pelo sorteador da Tabela 10 com seed 20260927 (`revision_v4/e7_reasoning.py select`). 72 FRB ficam abaixo de 15σ (70 abaixo de 5σ) |

Cada arquivo é um subconjunto do manifesto de imagens, pronto para `classify --manifest`.
Avaliadores: não abram estes arquivos antes de rotular.
