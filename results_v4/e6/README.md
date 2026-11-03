# E6 — sensibilidade ao prompt

Protocolo: `article/plano_revisao_referee_v4.md`, §10.8; definições da análise na §10.12 (19/10).
Prompts: `prompts_v4/` (P-a a P-e e R, congelados com hash antes de qualquer rodada).
Código: `revision_v4/gpu_runs.py` (rodadas) e `revision_v4/e6_analysis.py` (análise).

| Braço | Imagens | Prompts |
|---|---|---|
| `s200_<prompt>` | As 200 simuladas da Tabela 10 (`selections/e6_s200.jsonl`) | P0, P-a, P-b, P-c, P-d, P-e e R |
| `r200_<prompt>` | 100 positivos reais estratificados por σ e 100 negativos (`selections/e6_r200.jsonl`) | Os mesmos 7 |
| `sim2000_r` | As 2000 simuladas do binário | R |
| `real1600_r` | As 1600 janelas reais | R |

Cada braço fica em `<modelo>_<braço>/predictions.jsonl`, com as flags do artigo. O P0 roda de
novo nas 200 imagens de cada subconjunto, na mesma sessão. É contra ele que entra o McNemar dos
outros prompts, e a concordância dele com o P0 do artigo é relatada. O R nos conjuntos inteiros é
comparado com as rodadas do artigo.

```bash
uv run python -m revision_v4.gpu_runs e6 --model e2b --gpu 2   # os 16 braços
uv run python -m revision_v4.queues run --gpu 1   # E4B: os 7 braços r200_* e o real1600_r, depois do E3
uv run python -m revision_v4.queues run --gpu 3   # E4B: os 7 braços s200_* e o sim2000_r, depois da CNN do E5
uv run python -m revision_v4.e6_analysis analyze                 # e6_results.json; com --models e2b, só o E2B
```

Nas filas de 19/10, o E2B roda na GPU 2, e o E4B se divide entre as GPUs 1 e 3
(`revision_v4/queues.py`).

## Resultado (20/10)

Rodadas: os 16 braços de cada modelo, 0 falhas, sem alterações locais e sem retomada. Análise em
`e6_results.json`. O P0 refeito na sessão reproduz o P0 do artigo nas 200 imagens de cada
subconjunto: 200/200 respostas brutas idênticas, nos dois modelos e nos dois subconjuntos.

Acurácia de cada prompt (recall entre parênteses). Em negrito, as diferenças contra o P0 da mesma
sessão com McNemar exato p < 0,05, em acurácia, recall ou especificidade:

| Prompt | E2B, S200 | E2B, R200 | E4B, S200 | E4B, R200 |
|---|---|---|---|---|
| P0 | 0,930 (0,87) | 0,680 (0,37) | 0,910 (0,84) | 0,685 (0,38) |
| P-a, paráfrase | 0,930 (0,88) | 0,675 (0,35) | **0,875 (0,77)** | 0,685 (0,37) |
| P-b, NON_FRB primeiro | 0,945 (0,89) | 0,685 (0,37) | 0,930 (0,87) | 0,695 (0,40) |
| P-c, curto | 0,925 (0,87) | **0,640 (0,28)** | **0,825 (0,78)** | **0,670 (0,45)** |
| P-d, NON_FRB genérico | 0,935 (0,87) | 0,665 (0,33) | 0,890 (0,79) | 0,670 (0,34) |
| P-e, sem Low-S/N | 0,940 (0,89) | 0,650 (0,31) | 0,910 (0,83) | **0,660 (0,32)** |
| R, morfologia relaxada | **0,900 (0,87)** | 0,680 (0,37) | 0,895 (0,82) | **0,640 (0,29)** |

O que está em negrito:
- E2B, P-c no R200: recall −9,0 pp [−15,3; −2,7], p = 0,012; acurácia −4,0 pp, p = 0,039.
- E2B, R no S200: FP 7 contra 1 (especificidade p = 0,031); recall igual.
- E4B, P-a no S200: recall −7,0 pp [−12,7; −1,9], p = 0,016.
- E4B, P-c no S200: acurácia −8,5 pp [−12,9; −4,7], p = 2 × 10⁻⁵; FP 13 contra 2 (p = 0,001);
  recall −6,0 pp (p = 0,031).
- E4B, P-c no R200: recall +7,0 pp (p = 0,016), mas FP 11 contra 1 (p = 0,002).
- E4B, P-e no R200: recall −6,0 pp, p = 0,031.
- E4B, R no R200: recall −9,0 pp [−15,3; −2,7], p = 0,012.

**Faixa entre as paráfrases** (P-a, P-b e P-c), mínimo–máximo:

| | E2B, S200 | E2B, R200 | E4B, S200 | E4B, R200 |
|---|---|---|---|---|
| Acurácia | 0,925–0,945 | 0,640–0,685 | 0,825–0,930 | 0,670–0,695 |
| Recall | 0,87–0,89 | 0,28–0,37 | 0,77–0,87 | 0,37–0,45 |
| FPR | 0,00–0,02 | 0,00–0,00 | 0,01–0,13 | 0,00–0,11 |

O E2B é estável a paráfrases nas simuladas, e só a versão curta (P-c) perde recall nas reais. O
E4B é mais sensível: a versão curta muda tanto o recall quanto a FPR nos dois subconjuntos, e a
paráfrase P-a perde 7 pp de recall nas simuladas.

**R nos conjuntos inteiros**, contra o P0 do artigo nas mesmas imagens:

| | E2B: R | E2B: P0 | E2B: R − P0 | E4B: R | E4B: P0 | E4B: R − P0 |
|---|---|---|---|---|---|---|
| 1600 reais, recall | 0,290 | 0,285 | +0,5 pp, p = 0,68 | 0,220 | 0,270 | **−5,0 pp [−6,9; −3,1], p = 7 × 10⁻⁸** |
| 1600 reais, FP | 11 | 2 | **especificidade p = 0,012** | 5 | 5 | p = 1 |
| 2000 simuladas, recall | 0,866 | 0,905 | **−3,9 pp [−5,3; −2,6], p = 9 × 10⁻¹⁰** | 0,830 | 0,824 | +0,6 pp, p = 0,53 |
| 2000 simuladas, FP | 35 | 24 | p = 0,099 | 38 | 23 | **especificidade p = 3 × 10⁻⁴** |
| FRB 20121102A (470), recall | 0,198 | 0,187 | +1,1 pp, p = 0,36 | 0,123 | 0,166 | **−4,3 pp [−6,4; −2,3], p = 4 × 10⁻⁵** |

A morfologia relaxada não recupera os bursts de banda estreita do FRB 20121102A: no E2B não muda
o recall, e no E4B o reduz. Nos dois modelos, ela aumenta os FP em pelo menos um dos conjuntos.
