# SwinYNet — reinstalação e verificação

Protocolo: `article/plano_revisao_referee_v4.md`, §10.3. Rodado em 13/10/2026.

## Resultado

100 amostras do benchmark binário (seed 42), comparadas com `results_external_binary`:
**100/100 scores idênticos (|Δ| máximo = 0)**, com o mesmo número de candidatos e o mesmo rótulo
em todas. A instalação abaixo reproduz a rodada do artigo e serve para o E1.

## Versão usada (para citar na v4 e no Zenodo)

- Código: <https://github.com/expnn/SwinYNet>, commit **`2ad67fa91d5cd5fdc43bd3b75134863cfcea5333`**
  (12/08/2025). É o HEAD desde então e era o HEAD no clone de 18/05/2026 usado no artigo
  (histórico do shell; não houve `git pull` depois).
- Pesos: `simu_burst-swin_ynet+full.tar.gz` (Google Drive `1S2a1siGzVTAXpDZCZ01hhC_r2mR6a2NE`),
  SHA-256 `57f61a56…762382d`. O modelo é `te8hjj4j/model-0094-678585.pt`, SHA-256
  `f521120a…5ceefc77`, o mesmo que as predições originais registram.
- Ambiente: `uv venv --python 3.11` e `uv sync` com o `uv.lock` do repositório (torch
  2.4.1+cu124), **com uma exceção**: o lock aponta `pyerfa==2.0.1.6` para o índice
  `pypi.anaconda.org/liberfa`, que hoje só serve builds noturnos (404 para essa versão).
  Instalamos `pyerfa==2.0.1.5`, a última do PyPI. O pyerfa só entra nas conversões de tempo e
  coordenadas do astropy, e a comparação acima mostra que os scores não mudam.

## Comandos

```bash
git clone https://github.com/expnn/SwinYNet.git ~/SwinYNet && cd ~/SwinYNet
uv venv --python 3.11
uv sync --no-install-package pyerfa
uv pip install --python .venv/bin/python pyerfa==2.0.1.5
tar -xzf simu_burst-swin_ynet+full.tar.gz -C cache/

cd ~/frb_svlm
uv run python -m revision_v4.swinynet_check export --workdir /var/tmp/frb_svlm_swinynet_check
cd ~/SwinYNet && CUDA_VISIBLE_DEVICES=2 uv run --no-sync swinynet -c frbd/cfgs/te8hjj4j.yaml \
  -p /var/tmp/frb_svlm_swinynet_check/fits \
  -o /var/tmp/frb_svlm_swinynet_check/output/manifest.json -x run.num_postprocessor=0 -I none
cd ~/frb_svlm && uv run python -m revision_v4.swinynet_check compare \
  --workdir /var/tmp/frb_svlm_swinynet_check \
  --external /var/tmp/frb_svlm_swinynet_check/output/manifest.json
```

Tempo: 384 s para 100 arquivos numa RTX 4500 Ada (~3,8 s por arquivo).
