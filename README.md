# frb-vlm-benchmark

Code and evaluation artifacts for the paper:

> **Generalist Vision-Language Models for Fast Radio Burst detection: a zero-shot benchmark against a specialized detector**
> Raiff H. Santos, Amílcar R. Queiroz, Tharcísyo S. S. Duarte, K. E. L. de Farias, Rafael A. Batista
> [arXiv:2607.07382](https://arxiv.org/abs/2607.07382)

The pipeline simulates PSRFITS dynamic spectra containing Fast Radio Bursts (FRB), structured Radio Frequency Interference (RFI), and noise; renders them as anonymized PNG images; classifies the images with small, open-weight Vision-Language Models (Gemma 4 E2B/E4B) in a zero-shot, prompt-only regime; and evaluates the predictions, including a paired, sample-by-sample comparison against the specialized detector [SwinYNet](https://github.com/expnn/SwinYNet). The same models, prompts, and thresholds are then applied unchanged to the 1600 real FAST observations of [FAST-FREX](https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0), together with a burst-visibility diagnostic that separates classifier error from absence of signal in the rendered image.

## What is included

- The full pipeline code (`run_pipeline.py` plus the packages below).
- The exact configuration files used in the paper (`config.yaml`, `config_binary.yaml`).
- The label and image manifests of the simulated multiclass dataset (`dataset/metadata/`) and of the derived binary benchmark (`dataset_binary/metadata/`).
- The real-data manifests: the FAST-FREX input manifest and burst-parameter tables (`fast_frex/`), the ScienceDB file index, and the image manifest documenting the preprocessing of all 1600 rendered observations (`dataset_real_full/metadata/`).
- The burst-visibility diagnostics behind the real-vs-simulated comparison (`fast_frex/burst_visibility_full.csv`, `dataset/metadata/burst_visibility_sim.csv`).
- The per-sample predictions and evaluation outputs of the three systems, exactly as reported in the paper:

| Directory | Content |
|---|---|
| `results_vlm2b_0t_binary/` | Gemma 4 E2B, binary FRB vs NON_FRB (2000 samples) |
| `results_vlm_0t_binary/` | Gemma 4 E4B, binary FRB vs NON_FRB |
| `results_external_binary/` | SwinYNet predictions, imported into the pipeline format |
| `results_comparison_2b_0t/` | Paired comparison Gemma 4 E2B vs SwinYNet |
| `results_comparison_4b_0t/` | Paired comparison Gemma 4 E4B vs SwinYNet |
| `results_vlm2b_0t/` | Gemma 4 E2B, multiclass FRB/RFI/NOISE (3000 samples) |
| `results_vlm_0t/` | Gemma 4 E4B, multiclass FRB/RFI/NOISE |
| `results_prompt_timing_2b/` | Prompt-latency audit, Gemma 4 E2B (200-image subsample, four prompt variants) |
| `results_prompt_timing_4b/` | Prompt-latency audit, Gemma 4 E4B |
| `results_real_binary_full_2b/` | Gemma 4 E2B on the 1600 real FAST-FREX observations |
| `results_real_binary_full_4b/` | Gemma 4 E4B on the 1600 real FAST-FREX observations |

The simulated PSRFITS files and rendered images (about 60 GB) are not hosted in this repository. They are regenerated exactly by the pinned configurations and seeds, following the steps below. The real FAST-FREX observations (about 412 GB of PSRFITS) are not hosted either: they are downloaded from ScienceDB by `fetch_fast_frex.py`, and the manifests and diagnostics tracked here pin exactly which files were used.

## Modules

- `simulate_dataset/` — generates synthetic PSRFITS files for FRB, RFI, and NOISE via simulateSearch, and derives the binary benchmark subset (`select-binary-benchmark`).
- `plot_dataset/` — converts PSRFITS into anonymized dynamic-spectrum PNGs.
- `vlm_classifier/` — classifies images with Gemma 4 through Hugging Face Transformers (or a seeded `--dry-run` mode).
- `evaluation/` — computes discrete and continuous metrics from `predictions.jsonl`.
- `benchmark_export/` — exports anonymized FITS for external detectors.
- `benchmark_predictions/` — imports external predictions and compares models.
- `prepare_real/` — reads real search-mode PSRFITS, applies the preprocessing a real spectrum needs, and renders it through the same image path as the simulated benchmark.
- `run_pipeline.py` — the CLI that integrates all stages.
- `prompt_timing_experiment.py` — standalone paired prompt-latency audit (the computational-cost analysis of the paper).
- `fetch_fast_frex.py` — selects and downloads the FAST-FREX subset from ScienceDB and writes the `prepare-real` input manifest.
- `check_burst_visibility.py`, `check_burst_visibility_sim.py`, `analyze_visibility.py`, `summarize_real_run.py` — read-only diagnostics measuring the dedispersed peak of every sample, on a grid shared by the real and simulated sets.
- `figures_src/` — regenerates the article figures from the tracked artifacts.

## Requirements

- Python >= 3.10 and [uv](https://docs.astral.sh/uv/).
- [simulateSearch](https://ascl.net/2205.025) installed, with these executables on `PATH`: `simulateSystemNoise`, `simulateBurst`, `simulateRFI`, `simulateRFI_far_sidelobes`, `createSearchFile`. Set `PSRFITS_HEADER_DIR` to the directory containing `psrheader_simulate.fits`.
- An NVIDIA GPU with CUDA for real VLM inference (the paper runs used `--device-map cuda`).
- A Hugging Face token if the Gemma checkpoints require authentication: `export HF_TOKEN="hf_..."`.
- For the external-detector comparison: [SwinYNet](https://github.com/expnn/SwinYNet) v1.0.0 with the official pretrained weights.

## Installation

```bash
uv venv
uv pip install -e .
uv run python run_pipeline.py --help
```

Check the simulateSearch environment:

```bash
command -v simulateSystemNoise simulateBurst simulateRFI simulateRFI_far_sidelobes createSearchFile
printenv PSRFITS_HEADER_DIR
```

## Configuration

Both configuration files describe the same 2-second FAST L-band observation (1000–1500 MHz, 2048 channels, `tsamp = 1.96608e-4` s, 8-bit quantization, global seed 42) and differ only in the output directory:

- `config.yaml` → `dataset/` — the simulated multiclass set (FRB/RFI/NOISE).
- `config_binary.yaml` → `dataset_binary/` — the binary benchmark (FRB vs NON_FRB), a subset of the multiclass set derived with `select-binary-benchmark`; this configuration is not used to simulate, it only tells `classify` and `export-benchmark` where the benchmark lives.

During RFI simulation, each sample receives a primary category chosen reproducibly among `persistent_narrowband`, `impulsive_broadband`, `wifi_multiband`, `point_to_point_microwave`, and `satellite_like_inband`. The background noise level also varies per sample through a seeded `tsys_scale`, recorded in the metadata.

## Reproducing the paper

The commands below are the exact commands used to produce the results reported in the paper. Generation is deterministic: greedy decoding (`do_sample=False`) with `transformers.set_seed(42)` before each generation, fully seeded simulation, and a seeded benchmark selection. Rerunning overwrites the shipped `results_*` files in place, so `git diff` shows any deviation from the published runs.

### 1. Simulate the multiclass dataset (1000 per class = 3000 samples)

```bash
uv run python run_pipeline.py simulate \
  --n-per-class 1000 \
  --config config.yaml \
  --overwrite

uv run python run_pipeline.py plot \
  --config config.yaml \
  --normalization percentile \
  --percentile-low 1 \
  --percentile-high 99 \
  --cmap viridis \
  --width 1024 \
  --height 768 \
  --overwrite
```

> **Pass the geometry explicitly.** `plot` defaults to 1536x1152 at 192 dpi, whereas the paper renders at **1024x768**. Omitting `--width`/`--height` produces images at a different resolution from the ones the published predictions were made on, and the classification does depend on the rendered image. Every image manifest records the geometry actually used under `plot`, so `grep -m1 '"plot"' dataset/metadata/image_manifest.jsonl` confirms what a run produced.

### 2. Select the binary benchmark (2000-sample subset of the multiclass set)

The binary benchmark contains all 1000 FRB samples, 100 samples of each of the five RFI subtypes, and 500 NOISE samples, drawn deterministically (seed 42) from the multiclass set. No new simulation or rendering happens: the derived manifests written to `dataset_binary/metadata/` point to the FITS files and images already generated in step 1.

```bash
uv run python run_pipeline.py select-binary-benchmark \
  --config config.yaml \
  --output-dir dataset_binary \
  --n-frb 1000 \
  --per-rfi-subtype 100 \
  --n-noise 500 \
  --seed 42 \
  --overwrite
```

### 3. Binary task: FRB vs NON_FRB with Gemma 4

Gemma 4 E2B:

```bash
uv run python run_pipeline.py classify \
  --model gemma4 \
  --task frb-binary \
  --model-id google/gemma-4-E2B-it \
  --device-map cuda \
  --dtype auto \
  --max-new-tokens 1024 \
  --cache-implementation static \
  --config config_binary.yaml \
  --output results_vlm2b_0t_binary/predictions.jsonl \
  --generation-seed 42 \
  --overwrite

uv run python run_pipeline.py evaluate \
  --task frb-binary \
  --predictions results_vlm2b_0t_binary/predictions.jsonl \
  --output-dir results_vlm2b_0t_binary/ \
  --overwrite
```

Gemma 4 E4B:

```bash
uv run python run_pipeline.py classify \
  --model gemma4 \
  --task frb-binary \
  --model-id google/gemma-4-E4B-it \
  --device-map cuda \
  --dtype auto \
  --max-new-tokens 1024 \
  --cache-implementation static \
  --config config_binary.yaml \
  --output results_vlm_0t_binary/predictions.jsonl \
  --generation-seed 42 \
  --overwrite

uv run python run_pipeline.py evaluate \
  --task frb-binary \
  --predictions results_vlm_0t_binary/predictions.jsonl \
  --output-dir results_vlm_0t_binary/ \
  --overwrite
```

In this task the VLM must return `frb_probability`, a continuous estimate of P(FRB | image); the discrete label is derived from `--decision-threshold` (default 0.5).

### 4. External detector: SwinYNet

Export anonymized FITS for the external model (replace `/path/to/SwinYNet` with your SwinYNet checkout):

```bash
uv run python run_pipeline.py export-benchmark \
  --config config_binary.yaml \
  --output-dir /path/to/SwinYNet/data \
  --overwrite
```

Run SwinYNet v1.0.0 with the official pretrained weights and released inference configuration on the exported `fits/` directory. Its detections (a `manifest.json` with `id`, `prob`, and `label` per detection) are then imported back, aggregating multiple detections per sample by maximum probability:

```bash
uv run python run_pipeline.py import-benchmark-predictions \
  --benchmark-manifest /path/to/SwinYNet/data/metadata/benchmark_manifest.jsonl \
  --external-predictions /path/to/SwinYNet/data/output/manifest.json \
  --output results_external_binary/predictions.jsonl \
  --threshold 0.5 \
  --aggregation max \
  --overwrite

uv run python run_pipeline.py evaluate \
  --task frb-binary \
  --predictions results_external_binary/predictions.jsonl \
  --output-dir results_external_binary \
  --overwrite
```

Paired comparisons (accuracy, McNemar, ROC/PR with paired bootstrap, calibration, stratified FPR):

```bash
uv run python run_pipeline.py compare-models \
  --vlm-metrics results_vlm2b_0t_binary/metrics.json \
  --external-metrics results_external_binary/metrics.json \
  --vlm-predictions results_vlm2b_0t_binary/predictions.jsonl \
  --external-predictions results_external_binary/predictions.jsonl \
  --output-dir results_comparison_2b_0t \
  --overwrite

uv run python run_pipeline.py compare-models \
  --vlm-metrics results_vlm_0t_binary/metrics.json \
  --external-metrics results_external_binary/metrics.json \
  --vlm-predictions results_vlm_0t_binary/predictions.jsonl \
  --external-predictions results_external_binary/predictions.jsonl \
  --output-dir results_comparison_4b_0t \
  --overwrite
```

### 5. Multiclass task: FRB/RFI/NOISE with Gemma 4

Gemma 4 E2B:

```bash
uv run python run_pipeline.py classify \
  --model gemma4 \
  --model-id google/gemma-4-E2B-it \
  --device-map cuda \
  --dtype auto \
  --max-new-tokens 1024 \
  --cache-implementation static \
  --config config.yaml \
  --output results_vlm2b_0t/predictions.jsonl \
  --generation-seed 42 \
  --overwrite

uv run python run_pipeline.py evaluate \
  --predictions results_vlm2b_0t/predictions.jsonl \
  --output-dir results_vlm2b_0t/ \
  --overwrite
```

Gemma 4 E4B:

```bash
uv run python run_pipeline.py classify \
  --model gemma4 \
  --model-id google/gemma-4-E4B-it \
  --device-map cuda \
  --dtype auto \
  --max-new-tokens 1024 \
  --cache-implementation static \
  --config config.yaml \
  --output results_vlm_0t/predictions.jsonl \
  --generation-seed 42 \
  --overwrite

uv run python run_pipeline.py evaluate \
  --predictions results_vlm_0t/predictions.jsonl \
  --output-dir results_vlm_0t/ \
  --overwrite
```

### 6. Prompt-latency audit: full vs lean vs lean+reason vs multiclass

`prompt_timing_experiment.py` measures, on a stratified 200-image subsample of the binary benchmark (100 FRB, 10 per RFI subtype, 50 NOISE, fixed seed), the per-inference wall-clock time of four prompt variants applied to the same images: the full binary prompt of the paper, a lean output that keeps only the label and the continuous probability, the lean variant plus a one-sentence justification generated before the decision fields, and the multiclass prompt. It also reports the subsample accuracy of each variant and its label agreement with the full prompt, which is how the paper audits whether a cheaper prompt preserves the reference behavior without re-running the complete benchmark. It requires the rendered benchmark images (steps 1–3 above).

```bash
uv run python prompt_timing_experiment.py \
  --model google/gemma-4-E2B-it \
  --output-dir results_prompt_timing_2b \
  --overwrite

uv run python prompt_timing_experiment.py \
  --model google/gemma-4-E4B-it \
  --output-dir results_prompt_timing_4b \
  --overwrite
```

Each run writes `predictions.jsonl` (one line per sample with the timed response of every variant), `summary.json`, and `report.txt` (latency mean/median/percentiles, mean response length, accuracy at the 0.5 threshold, and agreement with the full prompt). The script loads the model on a single device by default (`--device-map cuda`); a `--dry-run` mode exercises the whole flow with seeded random predictions and no GPU.

### 7. Real data: download the FAST-FREX observations

The real-data validation of the paper runs on [FAST-FREX](https://www.scidb.cn/en/detail?dataSetId=3b3cf2f75a74419b89a56cc9626af2a0), a public set of 1600 FAST L-band observations: 600 catalogued bursts from three repeaters (470 from FRB20121102, 125 from FRB20201124, 5 from FRB20180301) and 1000 negatives. The archive provides, for every positive, the time of arrival within the file and the dispersion measure.

`fetch_fast_frex.py` builds the file index from the public dataset page, draws a stratified sample, downloads the PSRFITS with MD5 verification, and writes the input manifest consumed by `prepare-real`:

```bash
uv run python fetch_fast_frex.py \
  --n-frb 600 --n-negative 1000 --seed 42 \
  --out-manifest fast_frex/real_manifest_full.csv \
  --workers 6
```

Three things are worth knowing before starting the download:

- **The files are large.** Each PSRFITS is 244–488 MB, so the full set is about 412 GB. By default the script **deletes each file once it has been used**, to keep the working set small; pass `--keep-used` to retain them.
- **The server rate-limits.** Above roughly 16 concurrent connections ScienceDB starts answering 429; the script applies a global backoff when that happens.
- **The index is cached.** `fast_frex/scidb_index.json` is a snapshot of the catalogue (1603 entries) taken on 2026-08-10 and is tracked in this repository, so the exact selection of the paper stays reproducible even if the archive is reorganized. Delete it or pass `--refresh-index` to query the live catalogue instead.

The manifest used in the paper is included as `fast_frex/real_manifest_full.csv`, together with the burst-parameter tables of the three sources (`FRB*_summary.csv`). Its format is:

```csv
fits_path,label,toa,dm,source
fast_frex/FRB20121102_0001.fits,FRB,3.092631209,567.3,FRB20121102
fast_frex/00001_neg_sample.fits,RFI,,,
```

`fits_path` and `label` (`FRB`, `RFI`, or `NOISE`) are required; `toa` (seconds within the file) and `dm` (pc cm^-3) are required for `FRB` and optional for negatives. Since FAST-FREX does not separate interference from noise among its negatives, they are labelled `RFI` — the binary mapping to `NON_FRB` is the same either way.

### 8. Real data: render the observations with `prepare-real`

`prepare-real` converts real PSRFITS into PNGs under the same visual protocol as the benchmark (percentile 1–99, viridis, 1024x768, anonymized), inserting first the preprocessing that a real spectrum requires and a simulation does not:

1. extraction of a 2 s window centred on the burst (from the labelled `toa` and `dm`) or placed reproducibly in negative samples;
2. search-mode PSRFITS reading with `DAT_SCL`/`DAT_OFFS`/`ZERO_OFF` applied, polarizations averaged, and the band reordered to increasing frequency;
3. robust per-channel normalization (median and clipped standard deviation; a plain MAD is locked to quantization steps on 8-bit data), which removes the bandpass;
4. masking of dead channels and of channels with persistent RFI (`--zap-sigma`, plus `DAT_WTS` and `--edge-channels`);
5. block-averaging to `--time-bins x --freq-bins` (default 1024x512), which raises the per-pixel S/N by about sqrt(n) while keeping the dispersive sweep visible;
6. a second, per-row normalization pass on the decimated image, which flattens variance correlated between neighbouring channels (the FAST polyphase ripple) and erases residual persistent-RFI lines while preserving time-localized transients.

The burst is deliberately **not** dedispersed: the sweep is the visual signature the prompts define as an FRB, and a vertical impulse at zero DM is classified as `NON_FRB`.

In the FAST-FREX files the `toa` is the arrival at the top of the band (`--toa-ref top`, the default), the recorded band is 1000–1500 MHz in increasing order, and the usable band is 1050–1450 MHz (`--edge-channels 410`).

```bash
uv run python run_pipeline.py prepare-real \
  --input-manifest fast_frex/real_manifest_full.csv \
  --output-dir dataset_real_full \
  --edge-channels 410 \
  --overwrite
```

Every record of the output manifest documents, under `preprocessing`, the window used, the averaging factors, and how many channels were masked. The manifest of the paper run is included as `dataset_real_full/metadata/image_manifest.jsonl`, so the diagnostics of step 10 can be reproduced without re-downloading the archive.

`--reader astropy|your` selects the PSRFITS backend. The default `astropy` is the built-in reader; `your` uses the [your](https://github.com/thepetabyteproject/your) library — the one behind the FETCH / `your_candmaker` pipelines — as an independent cross-check of the decode. Both backends produce identical spectra, and `tests/test_prepare_real.py` asserts that parity whenever the optional extra is installed:

```bash
uv pip install -e '.[real]'
```

Before spending GPU time, inspect the PNGs: if a bright burst is not visible by eye, adjust the averaging factors or the zap first.

### 9. Real data: classify and evaluate

The models are applied unchanged — same weights, same prompt, same threshold as the simulated benchmark. Only the manifest differs:

```bash
for M in E2B:2b E4B:4b; do
  MODEL="${M%%:*}"; KEY="${M##*:}"
  uv run python run_pipeline.py classify \
    --model gemma4 --model-id "google/gemma-4-${MODEL}-it" \
    --task frb-binary \
    --manifest dataset_real_full/metadata/image_manifest.jsonl \
    --output "results_real_binary_full_${KEY}/predictions.jsonl" \
    --overwrite

  uv run python run_pipeline.py evaluate \
    --task frb-binary \
    --predictions "results_real_binary_full_${KEY}/predictions.jsonl" \
    --output-dir "results_real_binary_full_${KEY}" \
    --overwrite
done
```

The predictions and evaluation outputs of the paper are included in `results_real_binary_full_2b/` and `results_real_binary_full_4b/` (1600 samples each).

### 10. Real data: burst-visibility diagnostic

The central question of the real-data section is whether a missed burst was a classifier error or an image with no signal in it. `check_burst_visibility.py` answers it by rebuilding the exact array that became each PNG, dedispersing it at the catalogued DM, and measuring the peak of the time profile in units of sigma. It is read-only and never re-runs the pipeline. It needs the downloaded PSRFITS, so the resulting CSV is tracked here.

```bash
# Real observations. 'both' also measures the negatives with a DM drawn from
# the distribution of the positives, so that both classes get the same number
# of trials -- the maximum over a 24-DM scan is biased upwards against a
# single measurement, and comparing the distributions requires the symmetric one.
uv run python check_burst_visibility.py \
  --manifest dataset_real_full/metadata/image_manifest.jsonl \
  --negative-dm both \
  --out fast_frex/burst_visibility_full.csv

# Simulated benchmark, measured on the SAME time-frequency grid, which is what
# makes "recall on sharp real bursts vs recall on the simulated set" a
# controlled comparison rather than an assumption.
uv run python check_burst_visibility_sim.py \
  --manifest dataset/metadata/image_manifest.jsonl \
  --out dataset/metadata/burst_visibility_sim.csv
```

`analyze_visibility.py` consolidates both CSVs with the two prediction files and prints every number quoted in the real-data section — the peak distributions, the recall per sigma band, the breakdown per source, and the controlled real-vs-simulated comparison. Both CSVs are tracked, so this runs directly on a fresh clone:

```bash
uv run python analyze_visibility.py \
  --real fast_frex/burst_visibility_full.csv \
  --sim dataset/metadata/burst_visibility_sim.csv
```

`summarize_real_run.py` gives a shorter per-model, per-source, per-visibility summary of a single run.

### 11. Article figures

`figures_src/` regenerates the figures from the artifacts tracked here, writing vector PDFs and 300-dpi PNGs into `figures/`:

```bash
uv run python figures_src/make_article_figs.py        # ROC and score histograms (simulated)
uv run python figures_src/make_real_figs.py           # real examples, recall vs sigma, sigma histogram
uv run python figures_src/characterize_disagreements.py  # VLM-vs-SwinYNet disagreement breakdown
uv run python figures_src/make_examples_fig.py        # one example per class (needs steps 1-2)
```

`make_article_figs.py` asserts that the ROC-AUC it recomputes from `paired_scores.csv` matches the value published in the paper, so it doubles as a check that the shipped scores are the ones behind the figures. All of these run on a fresh clone except `make_examples_fig.py`, which needs the rendered simulated images from steps 1–2.

## Outputs

`classify` writes one JSON line per sample with `sample_id`, `image_path`, `true_label`, `predicted_label`, `confidence`, `raw_model_response`, `parsed_response`, `error`, and `inference_seconds` (wall-clock seconds of the successful model call). `evaluate` writes `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, `summary.csv`, and `evaluation_report.pdf`. `compare-models` writes paired reports, statistical tests, and ROC/PR/calibration/threshold plots.

Samples with execution errors are retried; content problems (invalid JSON, missing fields) are never resampled — the raw response is kept and the issue recorded in `content_warning`.

## Leakage control

Images given to the VLM are anonymized: no title is drawn on the PNG and files use neutral names (`sample_000000.png`). Class names appear only in metadata manifests and evaluation artifacts. The FITS files exported to the external detector carry the same neutral identifiers.

## Notes

- `--dry-run` on `classify` produces seeded random predictions without loading a VLM, useful to test the setup end to end.
- With `--overwrite`, each stage first removes its own previous artifacts; without it, the pipeline refuses to clobber existing outputs.
- If your PyTorch/Transformers build has issues with the attention or cache backends, add `--attn-implementation none` and/or `--cache-implementation none`. For precise CUDA errors, prepend `CUDA_LAUNCH_BLOCKING=1` and restart the process after any device-side assert.

## Tests

The unit tests do not require simulateSearch or a GPU:

```bash
uv run python -m unittest discover -s tests
```

## Citation

If you use this code or the benchmark artifacts, please cite:

```bibtex
@misc{santos2026generalist,
  title         = {Generalist Vision-Language Models for Fast Radio Burst detection:
                   a zero-shot benchmark against a specialized detector},
  author        = {Santos, Raiff H. and Queiroz, Am{\'i}lcar R. and
                   Duarte, Tharc{\'i}syo S. S. and de Farias, K. E. L. and
                   Batista, Rafael A.},
  year          = {2026},
  eprint        = {2607.07382},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.07382}
}
```

## License

MIT — see [LICENSE](LICENSE).
