# E5 (avaliadores) e auditoria do E4 — ferramenta de rotulação

Protocolo: `article/plano_revisao_referee_v4.md`, §10.6 e §10.7; amostras em
`results_v4/selections/` (§10.12, 19/10). Montada em 19/10/2026 com
`uv run python -m revision_v4.labeling_tool build`.

## O que cada avaliador recebe

A pasta `labeling_v4/avaliador_<A|B>/` (fora do git), com `index.html`, `LEIA-ME.txt` e
`images/`; cada avaliador a recebe num zip, `labeling_v4/avaliador_<A|B>.zip`, e rotula na própria
máquina (o avaliador A acessa o servidor por ssh e copia o zip por `scp`). Funciona offline: basta
abrir o `index.html` num navegador. São três etapas, nesta ordem:

1. **Treino:** 10 imagens simuladas do benchmark (5 FRB com o sweep inteiro, 3 RFI de famílias
   diferentes e 2 NOISE), com a resposta certa mostrada depois de cada uma.
2. **Parte A (E5):** os 300 PNGs do artigo (200 positivos nas cotas do protocolo e 100
   negativos). Pergunta: "a imagem contém um FRB (varredura dispersa)?", sim/não e confiança
   de 1 a 3.
3. **Parte B (auditoria do E4):** 200 negativos depois do P0, disjuntos dos negativos da
   parte A. Pergunta: "há RFI visível?", sim/não e o tipo (faixa horizontal persistente,
   impulso vertical, padrão periódico ou outro).

A ordem de cada parte é sorteada por avaliador (seed 20261013). A página não contém rótulo, σ
nem resposta de modelo. As respostas ficam salvas no navegador a cada imagem (dá para fechar e
continuar depois) e saem no CSV `rotulacao_avaliador_<A|B>.csv`, que volta para
`results_v4/e5/ratings/`. Desde a versão 2 (01/11, abaixo), o avaliador pode voltar às imagens
anteriores dentro das partes A e B e mudar a resposta, e o CSV guarda a final e a primeira. Essa pasta fica fora do git até os dois CSVs chegarem, porque o
avaliador B pode ter acesso ao repositório.

## Instruções, teste e envio (20/10, antes de qualquer rotulação)

Cada pasta tem um `LEIA-ME.txt`, igual para os dois avaliadores, com o que a página já diz e o
que vem antes dela: extrair o zip, usar Chrome, Edge ou Firefox numa janela normal, tela cheia e
zoom em 100%, e fechar e reabrir o navegador depois do treino para conferir que as respostas
ficam guardadas. As instruções não dão a proporção de FRBs nem as faixas de σ. Uma tela entre as
etapas anuncia a parte A e a parte B, onde a pergunta muda, e a imagem sempre cabe inteira na tela
(no tamanho natural, 1024 × 768, quando a tela permite).

```bash
# regravar index.html e LEIA-ME.txt sem mexer na chave nem nas imagens; para se a sequência mudaria
uv run python -m revision_v4.labeling_tool pages
# teste de ponta a ponta no Chrome headless: 510 respostas pelo teclado, duas reaberturas, CSV e check
node tests/e2e_labeling_tool.mjs run labeling_v4/avaliador_B <pasta temporária>
# capturas em seis tamanhos de tela e no modo escuro, conferindo que nada precisa de rolagem
node tests/e2e_labeling_tool.mjs shots labeling_v4/avaliador_B <pasta temporária>
# zip de cada avaliador (datas fixas; só a página, o LEIA-ME e as 510 imagens)
uv run python -m revision_v4.labeling_tool pack --rater A
uv run python -m revision_v4.labeling_tool pack --rater B
```

Em 20/10, depois do `pages`, o teste passou nas duas pastas e na cópia extraída do zip do
avaliador B (271 MB, 512 arquivos, SHA-256 `0770f06a576c266b622af09dbebdeb9e4802555e915cd377d07df5ae2f587393`;
o `pack` refaz o mesmo zip). As capturas couberam sem rolagem em 1920 × 1080, 1920 × 970,
1440 × 790, 1366 × 657, 1280 × 720 e 1024 × 700, com a imagem no tamanho natural nas duas
primeiras.

Em 01/11, ainda sem nenhum CSV, o mesmo `pack` fez o zip do avaliador A, que rotula na sua
máquina: 271 MB, 512 arquivos, SHA-256 `ad053bcad9b082bdfe1f7aaf948a034999467b3abc4f1b30427579846333081c`.
A cópia extraída é idêntica à pasta, byte a byte, e o teste de ponta a ponta passou nela (510
respostas, 0 linhas diferentes, `check` ok). A pasta, a chave e a sequência não mudaram.

O teste achou dois defeitos na versão de 19/10, corrigidos antes de qualquer rotulação (§10.12):
na parte B, trocar um "sim" com tipo por "não" gravava uma linha que o `check` rejeita e que
pararia a análise; no treino, a resposta mudava se o avaliador apertasse outra tecla depois do
feedback.

## Esclarecimento da parte B (01/11, durante a rotulação)

O avaliador A perguntou como separar RFI de ruído na parte B. O treino só diz se há FRB, então
nada na ferramenta ancora essa fronteira além da definição escrita. A resposta é um texto único
para os dois avaliadores, `esclarecimento_parte_B.txt`, que só desdobra a definição da página e
os rótulos dos quatro tipos, sem citar imagem, proporção ou faixa de σ. Foi escrito sem abrir
nenhuma imagem nem resposta. O avaliador A o recebeu em 01/11 às 15:43 UTC; o B recebe o mesmo
texto, sem acréscimos, e o horário do envio fica anotado aqui. Na versão 2 (abaixo), o mesmo
texto entrou na página e no `LEIA-ME.txt`.

Descritivo, fixado antes de qualquer CSV (§10.12): este README relata quantas respostas da parte
B de cada avaliador vieram antes do esclarecimento, pelo `time_utc` do CSV, e, se houver alguma,
a contagem de consenso e o κ da parte B só nas imagens que os dois responderam depois dele. O
número que a C6 cita continua o das 200 imagens.

## Versão 2 da ferramenta (01/11, durante a rotulação, antes de qualquer CSV)

Pedida pelo primeiro autor, registrada na §10.12 e no commit `1821cdf` antes de gerar as páginas.

- **Navegação.** Dentro das partes A e B, o avaliador volta às imagens anteriores (botão Anterior,
  tecla ←, "Início da parte") e vê a resposta guardada. Ela só muda quando ele avança com Enter;
  uma mudança não gravada se perde ao sair da imagem, e a página avisa enquanto ela não foi
  gravada. "Ir para onde parei" leva à primeira imagem sem resposta. O treino continua sem volta.
- **A parte A fecha quando a parte B começa.** A tela antes da parte B deixa voltar à parte A;
  depois de começar a parte B, não. As 200 imagens da parte B não recalibram a referência do C4.
- **Guia da parte B.** O texto de `esclarecimento_parte_B.txt` está na tela antes da parte B e no
  `LEIA-ME.txt`; um resumo dele fica ao lado de cada imagem da parte B.
- **CSV v2.** Uma linha por imagem: a resposta final (`answer`, `confidence`, `rfi_type`), a primeira
  (`first_*`), o horário de cada uma (`time_utc` é o da última confirmação), o número de mudanças e
  de confirmações, o tempo da primeira resposta e o total, e a versão da ferramenta da primeira
  resposta (`first_tool`).
- **Continuidade.** As respostas ficam no navegador, com a mesma chave nas duas versões. A versão 2
  lê as que a versão 1 guardou (primeira = final, `first_tool` = 1) e recusa um estado que não bata
  com a sequência. O `LEIA-ME.txt` explica como trocar a pasta sem perder nada e pede a quem já
  respondeu imagens da parte B que as confirme com o guia, ou corrija.
- **Análise.** Vale a resposta final. O `check` e o `analyze` aceitam os dois formatos. O `analyze`
  acrescenta os descritivos: respostas mudadas, por avaliador e parte; o desfecho do C4, o κ e o
  número da C6 com as primeiras respostas (`first_answers_descriptive`); e o esclarecimento
  (`clarification_descriptive`), que conta para cada avaliador do horário anotado (o do A no código,
  o do B por `--clarified B=<horário>`) ou da primeira resposta na versão 2, o que vier antes.

A sequência de imagens, a chave e o `labeling_manifest.json` não mudaram (o `pages` recusaria).
Os zips da versão 2 substituem os da versão 1, com o mesmo nome e a mesma pasta interna, para que
o `index.html` fique no mesmo caminho:

| Zip | Tamanho | Arquivos | SHA-256 |
|---|---|---|---|
| `labeling_v4/avaliador_A.zip` | 271 MB | 512 | `7cdc9b5f391e12665de40c5f218b70eeea0975b49e6567b4bc41c222c9f6bac4` |
| `labeling_v4/avaliador_B.zip` | 271 MB | 512 | `1068accd615afab27c540b160aca28c9ac76f605593c8b188570a056ed3a5373` |

Testes em 01/11, nas cópias extraídas dos dois zips (idênticas às pastas, byte a byte):

```bash
# 510 respostas com navegação: voltar, mudar e gravar, mudança descartada, início da parte,
# ir para onde parei, treino sem volta, parte A fechada, volta à parte A antes da parte B e
# à parte B depois do fim; CSV linha a linha (final, primeira, mudanças, confirmações) e check
node tests/e2e_labeling_tool.mjs run <pasta extraída> <pasta temporária>
# a página da versão 1 (commit fdd01a2) responde 325 imagens; a versão 2, no mesmo caminho,
# continua de onde parou, muda uma resposta antiga e termina; um estado trocado é recusado
node tests/e2e_labeling_tool.mjs migrate <pasta com a versão 1> <index.html da versão 2> <pasta temporária>
# 7 telas em 6 tamanhos e no modo escuro, sem rolagem
node tests/e2e_labeling_tool.mjs shots <pasta extraída> <pasta temporária>
```

Os dois `run` e os dois `migrate` passaram (0 linhas diferentes, `check` ok), e as 49 capturas
couberam sem rolagem, de 1024 × 700 a 1920 × 1080. O `analyze`, ensaiado fora do repositório com
os CSVs sintéticos dos dois `run`, gera os descritivos novos. Os 115 testes unitários da revisão
passam.

## Cegamento

A chave (nome anônimo → amostra, rótulo e σ) é `labeling_key.jsonl`. Ela **ficou fora do git até
as duas rotulações chegarem** e entrou junto com os CSVs em 02/11 (abaixo). O `labeling_manifest.json`
guarda o SHA-256 dela desde 19/10, o que vale como pré-registro. O avaliador A é o primeiro autor e conhece o conjunto, uma limitação a declarar no
texto. Antes de rotular, ele não deve abrir a chave nem `results_v4/selections/`.

## CNN de referência: resultado (19/10)

Código: `revision_v4/e5_cnn.py`. Os dois folds treinaram com 9000 pares e validaram com 1000
(`cnn_train_fold{0,1}.json`). Os dois pararam na época 6, com a melhor na 3: perda de validação
0,215 e 0,211, acurácia de validação 0,921 e 0,922. Commits sem alterações locais. Pontuação em
`cnn_scores_fold{0,1}.jsonl`: 802 e 798 PNGs de teste, e 500 controles de validação por modelo.
Análise em `cnn_results.json`.

Recall nos 600 PNGs positivos do artigo (Wilson 95% no limiar 0,5) e FPR nos 1000 negativos:

| Bin de σ | n | CNN, 0,5 | CNN, limiar do E2B | CNN, limiar do E4B | E2B | E4B |
|---|---|---|---|---|---|---|
| < 5 | 77 | 0,610 [0,499; 0,712] | 0,429 | 0,429 | 0,013 | 0,013 |
| 5–8 | 163 | 0,883 [0,825; 0,924] | 0,791 | 0,804 | 0,037 | 0,031 |
| 8–15 | 208 | 0,976 [0,945; 0,990] | 0,966 | 0,966 | 0,173 | 0,130 |
| 15–30 | 77 | 1,000 [0,952; 1,000] | 1,000 | 1,000 | 0,688 | 0,701 |
| ≥ 30 | 75 | 1,000 [0,951; 1,000] | 1,000 | 1,000 | 1,000 | 1,000 |
| FPR | 1000 | 0,017 [0,011; 0,027] | 0,003 | 0,003 | 0,002 | 0,005 |

O limiar casado é o menor score cuja FPR nos controles da validação não passa da FPR do VLM nos
negativos reais: 0,852 e 0,696 nos modelos 0 e 1 para o E2B, e 0,851 e 0,646 para o E4B.

**C4** (Δ = recall da CNN − recall do VLM nos bins de 5–15σ, nas mesmas 371 imagens; Newcombe
pareado e McNemar exato):

| | Limiar 0,5 | Limiar casado com a FPR do VLM |
|---|---|---|
| E2B | +82,2 pp [+77,7; +85,6], p = 2 × 10⁻⁹⁰ | +77,6 pp [+72,7; +81,5], p = 1 × 10⁻⁸³ |
| E4B | +84,9 pp [+80,6; +88,0], p = 3 × 10⁻⁹⁵ | +80,9 pp [+76,2; +84,4], p = 7 × 10⁻⁸⁹ |

A CNN passa o critério do C4 nos dois VLMs e nos dois limiares. Abaixo de 5σ, a CNN acha de 43
a 61% dos bursts. Duas imagens com σ ≈ 2 que a CNN dá como FRB com probabilidade > 0,97 foram
inspecionadas. As duas mostram um sweep fraco de banda estreita, que o σ do diagnóstico, calculado
na banda inteira, subestima.

**Controle fora do burst** (§10.12, registrado depois do resultado acima e antes de rodar). A
CNN só viu fundos de arquivos negativos no treino. O controle mede se ela também dispara nas
janelas do E3 dos arquivos de burst que não tocam o burst (1259 janelas em 576 arquivos), em
comparação com as 6000 janelas dos arquivos negativos. Com f = limite superior do IC da taxa fora
do burst, a conclusão do C4 pela CNN só se mantém se o Δ descontado, (r − f)/(1 − f) − recall do
VLM, continuar ≥ 10 pp.

```bash
uv run python -m revision_v4.e5_cnn score-offburst --fold 0   # e --fold 1
uv run python -m revision_v4.e5_cnn analyze-offburst          # cnn_offburst_results.json
```

Resultado (`cnn_offburst_results.json`; IC 95% por bootstrap de arquivos):

| Limiar | Fora do burst | Negativas | Diferença | Δ do C4 descontado |
|---|---|---|---|---|
| 0,5 | 0,029 [0,019; 0,040] (36/1259) | 0,017 [0,014; 0,021] (104/6000) | +1,1 pp [+0,1; +2,3] | E2B +81,9 pp; E4B +84,6 pp |
| Casado com o E2B | 0,0040 [0,0008; 0,0079] (5/1259) | 0,0013 [0,0005; 0,0023] (8/6000) | +0,3 pp [−0,1; +0,7] | E2B +77,5 pp |
| Casado com o E4B | 0,0056 [0,0016; 0,0108] (7/1259) | 0,0028 [0,0017; 0,0042] (17/6000) | +0,3 pp [−0,2; +0,8] | E4B +80,7 pp |

Fora do burst, a CNN dispara quase tanto quanto nos arquivos negativos. No limiar 0,5 há um
excesso de 1,1 pp, cujo IC mal exclui o 0. Nos limiares casados, a diferença é compatível com 0.
Descontar até o limite superior da taxa fora do burst (no máximo 4,0%) quase não muda o recall
da CNN nos bins de 5–15σ (0,935 → 0,933 no limiar 0,5). O Δ continua entre +77 e +85 pp nos
quatro casos. A CNN responde ao burst, não ao arquivo, e continua passando o critério do C4. A
decisão do C4 espera a parte humana: pela leitura da regra registrada em 20/10 (§10.12), a tese
forte ("o sinal está no PNG, e o limite é do VLM") exige que a CNN e os humanos passem. Se só a CNN
passar, o texto relata as duas referências, sem a tese forte. As rotulações ainda não foram
entregues.

**Pesos (02/11).** Os dois checkpoints estão em `cnn_weights/resnet18_fold{0,1}.pt` (44,8 MB cada,
o `state_dict` do PyTorch da ResNet-18 de `e5_cnn.py`), copiados de `/var/tmp/frb_svlm_e5_cnn/`, o
`WORKDIR` do script, que é o caminho registrado nos manifestos. Os SHA-256 (`6156a040…` e
`0ae9e894…`) conferem com os de `cnn_train_fold{0,1}.json`, `cnn_score_fold{0,1}.json` e
`cnn_offburst_score_fold{0,1}.json`. Para pontuar de novo com estes pesos, sem treinar, copie-os para
o `WORKDIR` depois do `render` e rode `score` (ou `score-offburst`). O artigo promete "the reference
network" no repositório, e o `/var/tmp` será limpo depois do aceite.

## Análise das rotulações (código de 20/10, anterior a qualquer rotulação)

Código: `revision_v4/e5_humans.py`. As regras estão na §10.12 (20/10) e foram fixadas antes de
existir qualquer CSV. O código só foi ensaiado com respostas aleatórias, fora do repositório.

```bash
# ao receber cada CSV: confere o formato, sem ler rótulos nem calcular taxas
uv run python -m revision_v4.e5_humans check --rater A --csv results_v4/e5/ratings/rotulacao_avaliador_A.csv
# com os dois CSVs: human_results.json aqui e audit_results.json em results_v4/e4/
uv run python -m revision_v4.e5_humans analyze
```

- **Parte A.** A decisão é o "sim", com qualquer confiança. Saem o recall por faixa de σ e a FPR
  nos 100 negativos, por avaliador, pela união e pelo consenso, ao lado do E2B, do E4B e da CNN
  nas mesmas imagens, com McNemar.
- **Regra do C4.** Nos 105 positivos de 5–15σ, o recall de cada avaliador é descontado da FPR
  dele: (r − f)/(1 − f), com f = limite superior do Wilson da FPR, como no controle fora do burst
  da CNN. O avaliador passa contra um VLM com Δ descontado ≥ 10 pp e McNemar p < 0,05. Os humanos
  passam se os dois avaliadores passarem. Como a CNN passa, a tese forte depende dos humanos. Se
  eles não passarem, o texto relata as duas referências, sem a tese forte.
- **Descritivos:** κ de Cohen, o ponto conservador (sim com confiança 3), a AUC na escala de 6
  níveis, o treino e o tempo de resposta.
- **Parte B (auditoria do E4):** os negativos com RFI visível por avaliador, pela união e pelo
  consenso, por tipo, e o κ. A C6 cita o consenso.

Ordem depois das duas rotulações (feita em 02/11; abaixo):
1. conferir os dois CSVs com o `check`;
2. commitar os CSVs e a chave juntos, tirando do `.gitignore` as linhas de `ratings/` e da chave.
   O `analyze` confere se o SHA-256 da chave bate com o do `labeling_manifest.json`;
3. rodar o `analyze` e commitar os resultados.

## Rotulações recebidas (02/11)

Os dois CSVs chegaram em 02/11 e entraram no git com a chave, no mesmo commit que tirou as linhas
dela e de `ratings/` do `.gitignore`. O SHA-256 da chave (`b742b26d…6eee9b`) é o do
`labeling_manifest.json`, de 19/10.

| Avaliador | Quando rotulou (UTC) | Versão nas primeiras respostas | `check` | SHA-256 do CSV |
|---|---|---|---|---|
| A | 01/11: treino e parte A de 15:16 a 15:38; parte B de 15:38 a 17:52 | treino e parte A na v1; parte B: 13 na v1, 187 na v2 | ok; formato v2; 510 respostas, nenhuma repetida; mudadas: 21 na parte B | `1b37de6845b32b23e5b476612565bbd70d7d9f56ea18f02881fb1d3a2b410c06` |
| B | 02/11: de 17:04 a 17:56 | todas na v2 | ok; formato v2; 510 respostas, nenhuma repetida; mudadas: 7 na parte A, 20 na parte B | `21f73ffdd5bf316b365982983ee17b91122f2c879a8a8b5b5319211875bf1215` |

**Esclarecimento da parte B.** O avaliador A deu as primeiras respostas de 13 imagens da parte B na
versão 1, entre 15:38 e 15:40 UTC, antes do esclarecimento (15:43). Na versão 2, como o LEIA-ME
pede, voltou ao início da parte B: a última confirmação das 200 respostas finais dele (`time_utc`)
é posterior ao esclarecimento. O avaliador B rotulou só na versão 2, que traz o guia na página e
no LEIA-ME. Para ele, o esclarecimento conta da primeira resposta (02/11, 17:04 UTC), anterior a
todas, e um envio anterior do texto não mudaria nada. Por isso o `analyze` roda sem `--clarified`.

## Resultado (02/11)

`uv run python -m revision_v4.e5_humans analyze`, sem `--clarified`, grava `human_results.json`
aqui e `results_v4/e4/audit_results.json`. A primeira rodada foi no commit `843b624`. A segunda, no
`59818de`, só acrescentou o recall de 5–15σ agregado e as máquinas nos mesmos negativos
(`machines_descriptive`; §10.12, 02/11, depois do resultado), e a comparação dos JSONs confirma
que nenhum outro campo mudou.

**Parte A.** Recall por faixa de σ (Wilson 95% para os avaliadores) e FPR nos 100 negativos, com
a CNN (0,5) e os VLMs (P0 do artigo, 0,5) nas mesmas 300 imagens. A decisão é o "sim", com
qualquer confiança.

| Faixa de σ | n | Avaliador A | Avaliador B | União | Consenso | CNN, 0,5 | E2B | E4B |
|---|---|---|---|---|---|---|---|---|
| < 5 | 25 | 0,440 [0,267; 0,629] | 0,120 [0,042; 0,300] | 0,440 | 0,120 | 0,720 | 0,000 | 0,000 |
| 5–8 | 45 | 0,622 [0,476; 0,749] | 0,333 [0,214; 0,479] | 0,622 | 0,333 | 0,933 | 0,044 | 0,022 |
| 8–15 | 60 | 0,917 [0,819; 0,964] | 0,783 [0,664; 0,869] | 0,933 | 0,767 | 0,950 | 0,133 | 0,117 |
| 15–30 | 45 | 1,000 [0,921; 1,000] | 1,000 [0,921; 1,000] | 1,000 | 1,000 | 1,000 | 0,711 | 0,733 |
| ≥ 30 | 25 | 1,000 [0,867; 1,000] | 1,000 [0,867; 1,000] | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| **5–15** | **105** | **0,790 [0,703; 0,857]** | **0,590 [0,495; 0,680]** | **0,800** | **0,581** | **0,943** | **0,095** | **0,076** |
| FPR | 100 | 0,000 [0,000; 0,037] | 0,000 [0,000; 0,037] | 0,000 | 0,000 | 0,010 | 0,000 | 0,020 |

**Regra do C4** (105 positivos de 5–15σ). f = 0,037 para os dois avaliadores, o limite superior do
Wilson de 0/100. O recall descontado é (r − f)/(1 − f), e o McNemar exato é feito nas decisões
brutas:

| Avaliador | VLM | Recall | Descontado | Recall do VLM | Δ descontado | p | Passa |
|---|---|---|---|---|---|---|---|
| A | E2B | 0,790 | 0,782 | 0,095 | +68,7 pp | 2e-22 | sim |
| A | E4B | 0,790 | 0,782 | 0,076 | +70,6 pp | 5e-23 | sim |
| B | E2B | 0,590 | 0,575 | 0,095 | +48,0 pp | 6e-15 | sim |
| B | E4B | 0,590 | 0,575 | 0,076 | +49,9 pp | 2e-15 | sim |

**Desfecho do C4: tese forte nos dois VLMs.** A CNN passa nos dois limiares, com o desconto fora
do burst, e os dois avaliadores passam. Pela §10.7, com a leitura de 20/10, o sinal está no PNG e
o limite é do VLM. Isso vale nos dois modelos, então o abstract leva a tese forte.

Descritivos:
- **κ de Cohen:** 0,80 [0,72; 0,86] nas 300 imagens (concordância de 89,7%) e 0,60 [0,47; 0,72]
  nos 200 positivos.
- **AUC na escala de 6 níveis:** de todos os positivos contra os negativos, 0,943 [0,919; 0,964] (A)
  e 0,929 [0,902; 0,953] (B), contra 0,986 da CNN, 0,699 do E2B e 0,654 do E4B. Só com os positivos
  de 5–15σ: 0,941 e 0,917, contra 0,989, 0,567 e 0,528.
- **Ponto conservador** (sim com confiança 3): nos 5–15σ, 0,305 para o A e 0,010 para o B, sem FP.
  O B guardou a confiança 3 para os bursts mais fortes (0,156 em 15–30σ e 0,920 acima de 30σ), e
  nesse ponto ele não passaria a regra. A regra conta o "sim" com qualquer confiança, e nenhum dos
  dois disse "sim" a um negativo, com nenhuma confiança.
- **Contra a CNN** (5–15σ, mesmas imagens, CNN a 0,5 com 1 FP nos 100 negativos): A −15,2 pp
  [−23,3; −8,0], p = 1 × 10⁻⁴; B −35,2 pp [−44,9; −25,1], p = 8 × 10⁻¹⁰.
- **Treino:** 10 de 10 para os dois.
- **Tempo** (mediana da primeira resposta): parte A, 2,3 s (A) e 4,2 s (B), num total de 19 e 25
  min; parte B, 1,6 s e 3,3 s.
- **Revisões:** o A fez a parte A inteira na versão 1, sem volta, e não mudou nenhuma resposta. O B
  mudou 7 das 300 respostas da parte A, 3 delas entre sim e não. Com as primeiras respostas, o
  desfecho do C4 é o mesmo: o recall do B nos 5–15σ fica em 0,562 (Δ descontado +45,0 e +46,9 pp,
  p ≤ 3 × 10⁻¹³), e o κ nas 300 imagens fica em 0,78.
- **Esclarecimento da parte B:** nenhuma resposta final veio antes dele (o A reconfirmou na versão
  2 as 13 que tinha dado antes). O consenso e o κ nas 200 imagens são, portanto, os de todas as
  imagens.

**Parte B (auditoria do E4):** 62 de 200 negativos com RFI visível no consenso (0,31 [0,25; 0,38]),
74 para o A e 117 para o B, κ = 0,36. Os detalhes estão em `results_v4/e4/README.md`.
