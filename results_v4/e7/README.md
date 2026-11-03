# E7 — ablação do raciocínio

Protocolo: `article/plano_revisao_referee_v4.md`, §10.9, com a emenda de 19/10 na §10.12.

O lean da Tabela 10 (V1) difere do lean+reason (V2) em duas coisas: a justificativa e a ordem dos
campos de decisão (`label` antes de `frb_probability` no V1, depois no V2). O V1o é o V2 sem a
justificativa, com a ordem do V2, e é a linha de base da regra. As outras variantes mudam só a
frase de instrução e o esquema do V2 (`prompts_v4/README.md`).

| Braço | Prompt | O que testa |
|---|---|---|
| `v1_lean` | `binary_lean.txt` | O lean da Tabela 10 |
| `v1o_no_reason` | `e7_v1o_no_reason.txt` | Sem justificativa, na ordem do V2 |
| `v2_reason_before` | `binary_lean_reason.txt` | Justificativa antes da decisão (a do artigo) |
| `v3_reason_after` | `e7_v3_reason_after.txt` | Justificativa depois da decisão |
| `v4_filler` | `e7_v4_filler_<modelo>.txt` | Filler neutro do tamanho médio da justificativa do modelo (21 tokens no E2B, 24 no E4B) |
| `v5_reason_reworded` | `e7_v5_reason_reworded.txt` | A instrução de justificativa reescrita |

**Amostra:** `results_v4/selections/e7_heldout400.jsonl`, com 200 FRB, 20 por subtipo de RFI e
100 NOISE do binário do E1 (sorteador da Tabela 10, seed 20260927).

**Regra** (decide; avaliada no E2B, o modelo da afirmação). A frase causal volta se:
- V2 > V1o e V2 > V1 no recall de FRB, cada um com McNemar exato p < 0,05;
- o V4 não reproduz o ganho (não há V4 > V1o com p < 0,05) e V2 > V4 (p < 0,05);
- V5 > V1o (p < 0,05).

Se V3 > V1o (p < 0,05), o efeito é atribuído ao pedido de justificativa, e não à posição dela. A
regra original, com o V1, também é calculada.

## Comandos

```bash
uv run python -m revision_v4.e7_reasoning select     # a amostra (já sorteada)
uv run python -m revision_v4.gpu_runs e7 --model e2b --gpu 2
uv run python -m revision_v4.gpu_runs e7 --model e4b --gpu 2
uv run python -m revision_v4.e7_reasoning analyze    # e7_results.json; com --models e2b, só o E2B
```

## Resultado (20/10)

Rodadas `<modelo>_<braço>/`: 400/400 cada, 0 falhas, commit `57820a5` sem alterações locais.
Análise em `e7_results.json`. O V1 e o V2 reproduzem as rodadas do E1 nas mesmas 400 imagens:
400/400 respostas brutas idênticas, nos dois modelos.

| Braço | E2B: recall | E2B: FPR | E2B: acurácia | E4B: recall | E4B: FPR | E4B: acurácia |
|---|---|---|---|---|---|---|
| V1, lean | 0,595 [0,526; 0,661] | 0,000 | 0,797 | 0,835 [0,777; 0,880] | 0,015 | 0,910 |
| V1o, sem justificativa | 0,755 [0,691; 0,809] | 0,040 | 0,858 | 0,815 [0,755; 0,863] | 0,005 | 0,905 |
| V2, justificativa antes | 0,895 [0,845; 0,930] | 0,015 | 0,940 | 0,830 [0,772; 0,876] | 0,000 | 0,915 |
| V3, justificativa depois | 0,840 [0,783; 0,884] | 0,090 | 0,875 | 0,840 [0,783; 0,884] | 0,010 | 0,915 |
| V4, filler | 0,830 [0,772; 0,876] | 0,035 | 0,897 | 0,790 [0,728; 0,841] | 0,000 | 0,895 |
| V5, justificativa reescrita | 0,895 [0,845; 0,930] | 0,015 | 0,940 | 0,865 [0,811; 0,906] | 0,000 | 0,932 |

Recall nos 200 FRB e FPR nos 200 negativos, Wilson 95%.

Comparações no recall de FRB (Newcombe pareado, McNemar exato; discordantes como "só no
primeiro × só no segundo"):

| | E2B | E4B |
|---|---|---|
| V1o − V1 (ordem dos campos) | +16,0 pp [+10,9; +21,0], p = 5 × 10⁻¹⁰ (32 × 0) | −2,0 pp [−4,4; +0,2], p = 0,13 |
| V4 − V1o (tokens antes da decisão) | **+7,5 pp [+3,9; +11,4], p = 6 × 10⁻⁵ (15 × 0)** | −2,5 pp [−5,4; +0,2], p = 0,13 |
| V2 − V4 (conteúdo da justificativa) | +6,5 pp [+2,8; +10,6], p = 0,001 (14 × 1) | +4,0 pp [+0,9; +7,4], p = 0,022 |
| V2 − V1o | +14,0 pp [+9,3; +19,1], p = 7 × 10⁻⁹ (28 × 0) | +1,5 pp [−1,3; +4,4], p = 0,45 |
| V2 − V1 | +30,0 pp [+23,6; +36,3], p = 2 × 10⁻¹⁸ (60 × 0) | −0,5 pp [−3,4; +2,3], p = 1 |
| V5 − V1o | +14,0 pp [+9,3; +19,1], p = 7 × 10⁻⁹ | +5,0 pp [+2,0; +8,4], p = 0,002 |
| V3 − V1o | +8,5 pp [+4,7; +12,6], p = 2 × 10⁻⁵ (17 × 0) | +2,5 pp [+0,2; +5,1], p = 0,063 |
| V2 − V3 | +5,5 pp [+2,1; +9,4], p = 0,003 (12 × 1) | −1,0 pp [−3,4; +1,3], p = 0,63 |

**Regra (E2B): a frase causal não volta.** Das cinco condições da regra emendada, quatro passam:
V2 > V1o, V2 > V1, V2 > V4 e V5 > V1o. A que falha é a do filler: o V4 reproduz parte do ganho
sobre o V1o (+7,5 pp, p = 6 × 10⁻⁵). A regra original, com o V1 no lugar do V1o, também não passa:
V4 − V1 = +23,5 pp, p = 1 × 10⁻¹⁴.

No E2B, os +30 pp do lean+reason sobre o lean da Tabela 10 se decompõem assim. A ordem dos campos
de decisão responde por +16 pp. Gerar tokens sem conteúdo antes da decisão responde por mais
+7,5 pp. O conteúdo da justificativa acrescenta +6,5 pp. A redação da justificativa não importa
(V5 = V2). Pedir a justificativa depois da decisão (V3) também ajuda (+8,5 pp sobre o V1o, o que
pela regra atribui o efeito ao pedido, e não à posição), mas fica 5,5 pp abaixo do V2 e tem FPR de
0,090. No E4B, que é só descritivo, nada muda o recall de forma significativa em relação ao V1o,
exceto o V5 (+5,0 pp).

**Filler:** o E4B copiou os 24 pontos exatamente nas 400 respostas. O E2B escreveu 20 pontos em
vez de 21 nas 400: uma cópia inexata, um token mais curta. As justificativas têm em média 21,5
tokens (V2), 20,0 (V3) e 21,3 (V5) no E2B, e 23,9, 20,6 e 24,1 no E4B, o que bate com o tamanho do
filler.
