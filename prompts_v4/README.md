# Prompts congelados da revisão v4

Cada arquivo é usado byte a byte por `run_pipeline.py classify --prompt-file <arquivo>`. A
proveniência de cada predição grava o SHA-256 e a origem do texto, e `registry.json` guarda o
hash de referência, conferido por `tests/test_prompts_v4.py`.

- `binary_full.txt` e `multiclass.txt` são os prompts do artigo, idênticos aos de
  `vlm_classifier/prompt.py`.
- `binary_lean.txt` e `binary_lean_reason.txt` são os da Tabela 10, idênticos aos de
  `prompt_timing_experiment.py`.

Regra (plano, §10.8): os prompts do E6 (P-a a P-e, R) e do E7 (V1 a V5) entram aqui com o hash no
registro **antes** da primeira rodada, e nenhum é ajustado depois de ver resultados. Qualquer
exceção vai para o registro de emendas (§10.12).

## E6 (congelados em 19/10/2026, antes de qualquer rodada)

Gerados por `revision_v4/e6_prompts.py` a partir de `binary_full.txt` (P0). O teste
`tests/test_prompts_v4.py` confere que cada arquivo é a saída do gerador. Tokens contados com o
tokenizador do Gemma 4 E2B na revisão fixada (P0 = 893).

| Arquivo | Variante | O que muda em relação ao P0 | Tokens |
|---|---|---|---|
| `e6_pa_paraphrase.txt` | P-a | Todo o texto reescrito, com as mesmas regras, faixas de calibração e definições; o esquema JSON fica igual | 971 (1,09) |
| `e6_pb_non_frb_first.txt` | P-b | Só a ordem das regras: NON_FRB, FRB, Low-S/N e calibração, com as mesmas frases | 893 (1,00) |
| `e6_pc_short.txt` | P-c | Texto condensado, com todas as regras de decisão e todas as faixas de calibração; o esquema fica numa linha, com as mesmas chaves e valores; saem as definições das features auxiliares, que a decisão não usa | 495 (0,55) |
| `e6_pd_generic_non_frb.txt` | P-d | A lista de NON_FRB vira "does not contain such a transient"; as faixas 0,05–0,17 e 0,01–0,04 deixam de nomear famílias; saem as quatro features que nomeiam famílias de RFI (`persistent_bands`, `repeating_pattern`, `uniform_or_constant_background`, `saturated_or_clipped`) | 693 (0,78) |
| `e6_pe_no_low_snr.txt` | P-e | Sai a regra 2 (Low-S/N); as outras são renumeradas | 834 (0,93) |
| `e6_r_relaxed_morphology.txt` | R | FRB deixa de exigir pulso único e banda larga: aceita emissão em parte da banda e vários componentes próximos com o mesmo sweep. Continua rejeitando impulsos de DM zero (um ou mais) e estrutura persistente. Sai o item "two or more temporally separated broadband pulses"; "broadband" sai da faixa 0,95–0,99; a definição de `repeating_pattern` deixa de citar "multiple pulses" | 907 (1,02) |

## E7 (congelados em 19/10/2026, antes de qualquer rodada)

Gerados por `revision_v4/e7_prompts.py` a partir de `binary_lean_reason.txt` (V2). O V1 é
`binary_lean.txt` e o V2 é `binary_lean_reason.txt`, sem cópia. O lean da Tabela 10 difere do
lean+reason também na ordem dos campos de decisão, por isso entra o V1o (emenda na §10.12 do
plano). Cada variante muda só as duas últimas linhas do V2: a frase de instrução e o esquema
JSON. Tokens contados com o tokenizador do Gemma 4 E2B na revisão fixada (V1 = 373, V2 = 394).

| Arquivo | Variante | Frase de instrução e esquema | Tokens |
|---|---|---|---|
| `e7_v1o_no_reason.txt` | V1o | Sem frase; `{"frb_probability", "label"}`. É o V1 com a ordem dos campos do V2 | 373 |
| `e7_v3_reason_after.txt` | V3 | "Decide first, then describe the visual evidence."; `reason` depois dos campos de decisão | 395 |
| `e7_v4_filler_e2b.txt` | V4 do E2B | "First copy the filler string exactly as shown; it carries no information. Then decide."; `filler` com 21 pontos antes dos campos | 415 |
| `e7_v4_filler_e4b.txt` | V4 do E4B | A mesma frase; `filler` com 24 pontos | 418 |
| `e7_v5_reason_reworded.txt` | V5 | "Before giving the probability and the label, summarize in one brief sentence what you observe in the image that bears on the decision."; `reason` descrito como "one brief sentence summarizing the observed evidence", antes dos campos | 410 |

O filler tem tantos pontos (um token cada) quanto a média de tokens do `reason` do próprio modelo
nas 2000 respostas lean+reason do E1: 21,4 no E2B e 24,2 no E4B. Antes do congelamento, um
piloto de formato rodou as variantes novas em 2 imagens do benchmark, fora do E1, na CPU. Todas
as respostas foram JSON válido. O E2B copiou 20 dos 21 pontos e o E4B copiou os 24.
