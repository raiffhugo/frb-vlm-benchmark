# E0 — reconstrução do benchmark simulado e verificação ponta a ponta

Protocolo: `article/plano_revisao_referee_v4.md`, §10.3. Rodado em 13/10/2026.

## Resultado

| Verificação | Resultado |
|---|---|
| Manifesto reconstruído × `dataset/metadata/image_manifest.jsonl` de 23/06 | **3000/3000 registros idênticos** |
| 50 imagens do binário (seed 42; 28 FRB, 12 RFI, 10 NOISE) reclassificadas com o E2B × `results_vlm2b_0t_binary` | **50/50 `raw_model_response` idênticos byte a byte** |
| Revisões do Gemma em cache (8 por modelo, abr–ago/2026) | **mesmos pesos e mesma entrada tokenizada** nos dois prompts |

Com isso, as imagens reconstruídas, o código atual e a revisão HF fixada para a v4
(`3e22461f…` no E2B, `ee0ef602…` no E4B) reproduzem a rodada do artigo. E6–E8 podem
usar estes PNGs. `png_sha256.csv` guarda o hash de cada PNG como referência.

## Comandos

```bash
uv run python -m revision_v4.e0_rebuild render --workers 32
uv run python -m revision_v4.e0_rebuild select-check          # 50 imagens, seed 42
HF_HUB_OFFLINE=1 uv run python -m revision_v4.e0_rebuild revisions
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 uv run python run_pipeline.py classify \
  --model gemma4 --task frb-binary --model-id google/gemma-4-E2B-it \
  --model-revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --device-map cuda --dtype auto --max-new-tokens 1024 --cache-implementation static \
  --config config_fast_lband.yaml --generation-seed 42 \
  --manifest results_v4/e0/check50_manifest.jsonl \
  --output results_v4/e0/check50_predictions.jsonl
uv run python -m revision_v4.e0_rebuild compare
```

O `classify` repete as flags da rodada de 11–12/07 (histórico do shell), acrescentando só a
revisão fixada. O `dpi` do render é o padrão de 192: o manifesto de 23/06 não o grava, e o
comando do histórico não passava `--dpi`.

## Arquivos

- `render_manifest.json`: proveniência do render e checagem do manifesto.
- `image_manifest_rebuilt.jsonl`: cópia versionada do manifesto das 3000 imagens (o original
  fica em `dataset/metadata/`, que o git ignora).
- `png_sha256.csv`: tamanho e SHA-256 de cada PNG reconstruído.
- `check50_*`: amostra, predições (com proveniência por registro) e comparação.
- `model_revisions.json`: blobs e hash da entrada tokenizada por revisão HF em cache.

Nota de proveniência: os scripts de `revision_v4/` rodaram antes de entrar no git; o commit
que os adiciona é o mesmo que adiciona estes resultados, sem mudança de conteúdo. O `git`
gravado nos manifestos é o HEAD de então (`b9d71d0` no render, `8b1fe3d` no classify).
