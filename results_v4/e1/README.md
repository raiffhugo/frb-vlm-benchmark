# E1 — conjunto simulado held-out

Protocolo: `article/plano_revisao_referee_v4.md`, §10.3. Gerado em 13/10/2026.

Mesma configuração do benchmark (`config.yaml`), mudando só a semente-mestre (**20260927**,
contra 42) e o diretório de saída: `config_e1.yaml`. São 1000 amostras por classe, e o
subconjunto binário sai do mesmo `select-binary-benchmark` (1000 FRB, 100 por subtipo de RFI,
500 NOISE; seed 20260927). Os FITS (~59 GB) ficam em `/var/tmp/frb_svlm_e1/fits` via o symlink
`dataset_e1/fits`. São regeneráveis pela semente e saem do disco no fim da revisão.

## Resultado

- 3000 amostras (1000 por classe; 200 por subtipo de RFI). O binário tem 1000 FRB, 100 por
  subtipo de RFI e 500 NOISE.
- Sementes por amostra: nenhuma repetida, **nenhuma igual às do benchmark**, e todas reproduzidas
  a partir da semente-mestre 20260927. A mesma conta a partir da 42 reproduz as 3000 do benchmark.
- Simulação 18:56–20:09 de 13/10 (1h13); render 20:09–20:16 (32 processos).

## Comandos

```bash
mkdir -p /var/tmp/frb_svlm_e1/fits dataset_e1
ln -s /var/tmp/frb_svlm_e1/fits dataset_e1/fits
uv run python run_pipeline.py simulate --config config_e1.yaml --n-per-class 1000
uv run python run_pipeline.py plot --config config_e1.yaml --normalization percentile \
  --percentile-low 1 --percentile-high 99 --cmap viridis --width 1024 --height 768 --workers 32
uv run python run_pipeline.py select-binary-benchmark --config config_e1.yaml \
  --output-dir dataset_e1_binary --seed 20260927
uv run python -m revision_v4.e1_heldout check
```

## Arquivos

- `e1_manifest.json`: proveniência, config, hashes, composição e checagem das sementes.
- `labels.jsonl` e `image_manifest.jsonl`: cópias versionadas das 3000 amostras. O binário fica
  em `dataset_e1_binary/metadata/`, versionado como o do benchmark.
- `png_sha256.csv`: hash de cada PNG.
- `simulate_log.txt`: as linhas INFO da simulação, tiradas do log de console (as do
  `logs/simulation.log` se perderam; ver o commit `440b009`).

## SwinYNet no binário (19/10)

Export das 2000 amostras pelo `export-benchmark` (`/var/tmp/frb_svlm_e1_swinynet`), SwinYNet
reinstalado (commit `2ad67fa`, `results_v4/swinynet/`) na GPU 3, com 10 469 s no total, e
importação como no artigo (limiar 0,5, máximo entre candidatos):
`uv run python -m revision_v4.e1_analysis import-swinynet --workdir /var/tmp/frb_svlm_e1_swinynet
--external /var/tmp/frb_svlm_e1_swinynet/output/manifest.json`.

- `swinynet/predictions.jsonl`: as 2000 predições, com os ids do binário do E1.
- `swinynet/swinynet_output_manifest.json`: a saída bruta do SwinYNet. O SHA-256 é o mesmo
  registrado em `swinynet/import_manifest.json`.

## Rodadas dos VLMs (19/10)

Os dois modelos com os quatro prompts congelados (binário completo, lean, lean+reason e
multiclasse), pelo `revision_v4/gpu_runs.py`: flags do artigo, revisão HF fixada, E2B na GPU 2 e
E4B na GPU 0. As 18 000 respostas foram válidas (0 falhas). Cada braço grava o commit em que
começou (`77edcc1` a `340fcf0`), todos com a árvore limpa, e o código de inferência é idêntico
entre esses commits.

```bash
uv run python -m revision_v4.gpu_runs e1 --model e2b --gpu 2
uv run python -m revision_v4.gpu_runs e1 --model e4b --gpu 0
uv run python check_burst_visibility_sim.py --manifest dataset_e1/metadata/image_manifest.jsonl \
  --out results_v4/e1/burst_visibility_e1.csv --only-frb
uv run python -m revision_v4.e1_analysis analyze
```

- `<modelo>_<braço>/predictions.jsonl`: as respostas brutas e a proveniência de cada uma.
- `burst_visibility_e1.csv`: o σ dos 1000 bursts do E1, pelo mesmo diagnóstico do benchmark.
- `e1_results.json`: todas as tabelas abaixo, com intervalos e testes.

## Resultado (19/10)

**Binário a 0,5** (Wilson 95%):

| | Benchmark | E1 | E1 − benchmark (Newcombe, amostras independentes) |
|---|---|---|---|
| E2B, binário completo | 0,9405 | 0,9410 [0,930; 0,951] | +0,05 pp [−1,42; +1,52] |
| E4B, binário completo | 0,9005 | 0,9120 [0,899; 0,924] | +1,15 pp [−0,66; +2,96] |
| SwinYNet | 0,9285 | 0,9200 [0,907; 0,931] | −0,85 pp [−2,50; +0,79] |

No E1, o recall é 0,910 (E2B), 0,843 (E4B) e 1,000 (SwinYNet); a FPR é 0,028, 0,019 e 0,160.
Nenhum modelo muda de forma significativa do benchmark para o E1.

**C2** (VLM − SwinYNet, pareado): no E1, E2B +2,10 pp [+0,64; +3,57], McNemar exato p = 0,006;
E4B −0,80 pp [−2,50; +0,90], p = 0,38. No benchmark eram +1,20 pp (p = 0,12) e −2,80 pp.

**C1** (limiar do SwinYNet escolhido num conjunto e aplicado no outro): o limiar que casa a FPR
do E2B no benchmark (0,024), aplicado no E1, dá ao SwinYNet FPR 0,028, recall 1,000 e acurácia
0,986 (VLM − SwinYNet = −4,50 pp, p < 10⁻¹⁵). No limiar de máxima acurácia, ele chega a 0,999 no E1, e
nenhum híbrido o supera (OR −1,30 pp; AND −4,50 pp). O sentido inverso dá o mesmo quadro.

**C9** (variantes de prompt nas 2000 imagens):

| | Completo | Lean | Lean+reason | Multiclasse como binário |
|---|---|---|---|---|
| E2B, acurácia | 0,941 | 0,799 | 0,939 | 0,939 |
| E2B, recall de FRB | 0,910 | 0,601 | 0,891 | 0,877 |
| E4B, acurácia | 0,912 | 0,921 | 0,929 | 0,960 |
| E4B, recall de FRB | 0,843 | 0,852 | 0,860 | 0,920 |

No E2B, o lean+reason recupera 29,0 pp do recall que o lean perde (290 FRB detectadas só pelo
lean+reason, 0 só pelo lean; p < 10⁻⁸⁰). No E4B, a diferença é de 0,8 pp (p = 0,18). Isso
confirma no held-out o padrão da Tabela 10, mas não separa a justificativa da ordem dos campos
de decisão, que também muda entre os dois prompts. Quem decide a frase causal é o E7.

**C7** (multiclasse nas 3000): E2B 0,803 (benchmark 0,801), E4B 0,849 (0,860). Os recalls por
classe são FRB 0,877 e 0,920, RFI 0,531 e 0,626, e NOISE 1,000 nos dois.

**σ:** 408 dos 1000 bursts do E1 ficam abaixo de 5σ, como no benchmark (434), e só 8 ficam entre
5 e 30σ.
