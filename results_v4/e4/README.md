# E4 — ablação do pré-processamento

Protocolo: `article/plano_revisao_referee_v4.md`, §10.6; definições na §10.12 (19/10).
Código: `revision_v4/e4_preprocessing.py`.

As 1600 janelas do artigo, com o início e o fator de tempo gravados no manifesto do artigo (20
nos arquivos de 98,304 µs e 40 nos de 49,152 µs):

| Variante | O que faz | PNGs idênticos aos do artigo |
|---|---|---|
| P0 | O pré-processamento do artigo, refeito só para conferir | **1600/1600** |
| P1 | P0 sem `flatten_rows` | 0 (esperado) |
| P2 | Só decodificação, normalização por canal e `block_average` (sem zap, DAT_WTS, bordas nem `flatten_rows`; saem só os canais de variância zero) | 0 (esperado) |

O P1 e o P2 só foram gerados depois de o P0 reproduzir o artigo. As imagens ficam em
`dataset_real_e4_<variante>/`, com os mesmos `sample_id` do artigo, para o McNemar pareado. A
auditoria humana de 200 negativos está na ferramenta do E5 (`results_v4/e5/`).

```bash
uv run python -m revision_v4.e4_preprocessing render --workers 40
uv run python -m revision_v4.gpu_runs e4 --model e2b --gpu 2   # e o mesmo com e4b
uv run python -m revision_v4.e4_preprocessing analyze --models e2b   # sem --models, os dois
```

## Resultado (20/10)

Rodadas `<modelo>_p1_binary_full/` e `<modelo>_p2_binary_full/`: 1600/1600 cada, 0 falhas, sem
alterações locais, sem retomada. O P0 é a rodada do artigo (`results_real_binary_full_2b` e
`_4b`), feita com os mesmos PNGs. Análise final em `e4_results.json`; o `e4_results_e2b.json` é a
análise parcial de 19/10, com o mesmo resultado para o E2B.

| | E2B: P0 | E2B: P1 | E2B: P2 | E4B: P0 | E4B: P1 | E4B: P2 |
|---|---|---|---|---|---|---|
| FP nos 1000 negativos | 2 | 24 | 9 | 5 | 3 | 1 |
| FPR (Wilson 95%) | 0,002 [0,001; 0,007] | 0,024 [0,016; 0,035] | 0,009 [0,005; 0,017] | 0,005 [0,002; 0,012] | 0,003 [0,001; 0,009] | 0,001 [0,000; 0,006] |
| Recall nos 600 bursts | 0,285 | 0,248 | 0,230 | 0,270 | 0,213 | 0,228 |
| Acurácia | 0,731 | 0,703 | 0,706 | 0,723 | 0,703 | 0,710 |

O P1 é o P0 sem `flatten_rows`, e o P2 é o pré-processamento mínimo.

Pareado contra o P0 (Newcombe pareado, McNemar exato):

| | Especificidade | Recall | Acurácia |
|---|---|---|---|
| E2B, P1 − P0 | −2,20 pp [−3,36; −1,25], p = 1 × 10⁻⁵ | −3,67 pp [−6,16; −1,19], p = 0,005 | −2,75 pp [−3,87; −1,64], p = 2 × 10⁻⁶ |
| E2B, P2 − P0 | −0,70 pp [−1,52; −0,02], p = 0,065 | −5,50 pp [−7,95; −3,08], p = 1 × 10⁻⁵ | −2,50 pp [−3,51; −1,50], p = 1 × 10⁻⁶ |
| E4B, P1 − P0 | +0,2 pp [−0,3; +0,8], p = 0,63 | −5,7 pp [−7,7; −3,7], p = 2 × 10⁻⁸ | −2,0 pp [−2,8; −1,2], p = 9 × 10⁻⁷ |
| E4B, P2 − P0 | +0,4 pp [−0,1; +1,1], p = 0,22 | −4,2 pp [−6,3; −2,1], p = 2 × 10⁻⁴ | −1,3 pp [−2,2; −0,5], p = 0,004 |

**C6:** pela regra da §10.12, o P2 não "multiplica os FP" em nenhum dos dois modelos, e os dois
vão em sentidos opostos. No E2B, os FP vão de 2 para 9 (4,5 vezes), mas o McNemar nos negativos
dá p = 0,065. Os 11 discordantes são os 9 FP novos do P2 e os 2 FP do P0, que o P2 acerta. O IC
de Newcombe pareado da especificidade mal exclui o 0, e a regra fixada antes usa o McNemar. No
E4B, os FP caem de 5 para 1 (p = 0,22). Como os modelos discordam, o texto da C6 fala de cada um
e não pode dizer que o pré-processamento mínimo multiplica os FP. Tirar o `flatten_rows` (P1)
sobe os FP do E2B de 2 para 24 (p = 1 × 10⁻⁵) e não muda os do E4B (5 para 3). O que vale para os
dois modelos é o recall: cai de 3,7 a 5,7 pp nas duas variantes, sempre com p < 0,01. A auditoria
humana dos 200 negativos (parte B da ferramenta do E5) está abaixo. O número que a C6 cita é o
do consenso dos dois avaliadores (§10.12, 20/10).

## Auditoria humana (02/11)

`revision_v4/e5_humans.py analyze` grava `audit_results.json` nesta pasta, com os CSVs e a chave em
`results_v4/e5/`. Os 200 negativos foram sorteados depois do P0 (seed 20261013), disjuntos dos
negativos da parte A, e julgados às cegas pelos dois avaliadores: "há RFI visível?", sim ou não, e
o tipo. A definição escrita é a mesma para os dois: RFI é qualquer estrutura que não seja o ruído
de fundo nem uma varredura dispersa, e o esclarecimento de 01/11 (`results_v4/e5/`) só a desdobra.
Todas as respostas finais dos dois vieram depois do esclarecimento.

| Leitor | Negativos com RFI visível | Wilson 95% |
|---|---|---|
| Avaliador A | 74 de 200 (0,370) | [0,306; 0,439] |
| Avaliador B | 117 de 200 (0,585) | [0,516; 0,651] |
| **Consenso (o número da C6)** | **62 de 200 (0,310)** | **[0,250; 0,377]** |
| União | 129 de 200 (0,645) | [0,577; 0,708] |

- **Concordância:** 66,5% das imagens, κ = 0,36 [0,24; 0,47], só razoável. O tipo também diverge.
  O avaliador A marcou 30 faixas horizontais, 23 impulsos verticais, 5 padrões periódicos e 16
  "outro"; o B, 69 faixas, 16 impulsos, 32 periódicos e nenhum "outro". Nos 62 do consenso, os dois
  dão o mesmo tipo em 17 (10 impulsos, 6 faixas, 1 periódico), e 45 são discordantes.
- **Os VLMs nesses negativos (descritivo):** nenhum dos 200 é FP do E2B ou do E4B no P0, nem no P2.
  No P1, o E2B tem 3 FP, nenhum entre os 62 do consenso, e o E4B nenhum.
- **Primeiras respostas (descritivo):** os dois avaliadores mudaram respostas da parte B (o A, 21,
  15 delas entre sim e não; o B, 20 e 12). Com as primeiras respostas, o consenso cai para 52 de
  200 (0,26 [0,20; 0,32]) e o κ para 0,24.

**Leitura para a C6:** o pré-processamento completo não tira toda a RFI visível. Os dois avaliadores
ainda veem RFI em cerca de um terço dos negativos auditados, e os modelos não marcam nenhum deles
como FRB. Com a ablação acima, isso põe o mérito da rejeição nos modelos. O κ baixo mostra que
"RFI visível" é um julgamento subjetivo: o consenso é o limite inferior e a união, o superior.
