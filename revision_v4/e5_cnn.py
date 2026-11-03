"""E5: CNN de referencia, "a imagem contem o sinal?" (plano, secao 10.7; regras na 10.12).

  1. ``plan``: folds por arquivo entre os 1000 negativos (seed 20261013). Em cada fold, 5000
     janelas de fundo de arquivos do fold, cada uma com e sem injecao (pares, para impedir
     atalhos de fundo): sigma alvo log-uniforme em [3, 60], largura log-uniforme em [1, 30] ms,
     metade em banda larga e metade com banda limitada (FWHM uniforme em 100-400 MHz),
     espalhamento tau_1250 uniforme em [0, 5] ms e DM uniforme em [350, 650]. 10% dos pares vao
     para a validacao. Os 600 positivos do artigo sao repartidos entre os dois modelos,
     alternando dentro de cada faixa de sigma embaralhada; nenhum burst real entra no treino.
  2. ``render --fold k``: os pares pelo P0, com o mesmo codigo do E2.
  3. ``train --fold k``: ResNet-18 do torchvision com pesos ImageNet, Adam (lr 1e-4), ate 10
     epocas, batch 16, precisao mista; fica o checkpoint de menor perda na validacao, e o treino
     para depois de 3 epocas sem melhora.
  4. ``score --fold k``: probabilidades nos PNGs do artigo (os positivos do modelo k e os
     negativos do outro fold) e nos controles da validacao.
  5. ``analyze``: recall por faixa de sigma (Wilson), nos limiares 0,5 e na FPR do VLM
     (escolhido nos controles da validacao), McNemar contra o E2B e o E4B nas mesmas imagens e
     a regra de decisao do C4.
  6. ``score-offburst --fold k``: controle acrescentado depois do resultado da CNN (secao 10.12):
     probabilidades nas janelas do E3 fora do burst (as de arquivos de burst que nao tocam o
     sweep da banda inteira, com folga de 0,5 s) e nas janelas dos arquivos negativos, com o
     mesmo modelo de cada arquivo no teste.
  7. ``analyze-offburst``: fracao de janelas sinalizadas fora do burst e nas negativas, com IC
     por bootstrap de arquivos, e o recall da CNN nos bins de 5-15 sigma descontado dessa taxa.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from prepare_real.reader import read_layout
from revision_v4.common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    REAL_SEED,
    RESULTS_ROOT,
    read_jsonl,
    run_manifest,
    sha256_file,
    write_json,
    write_jsonl,
)
from revision_v4.e2_injection import (
    BAND_FWHM_RANGE_MHZ,
    DM_RANGE,
    EDGE_MARGIN_S,
    _layout_summary,
    negative_files,
    redraw_corrupted,
    render_injection,
)
from revision_v4.e3_tiling import Sweep, catalog_sweeps
from revision_v4.injection import P0, Pulse
from revision_v4.selections import STRATA, stratum_of
from revision_v4.stats import PairedCounts, mcnemar_exact, newcombe_paired_difference, wilson


OUTPUT_DIR = RESULTS_ROOT / "e5"
PLAN_PATH = OUTPUT_DIR / "cnn_plan.jsonl"
POSITIVES_PATH = OUTPUT_DIR / "cnn_positive_assignment.jsonl"
WORKDIR = Path("/var/tmp/frb_svlm_e5_cnn")
PAPER_MANIFEST = Path("dataset_real_full/metadata/image_manifest.jsonl")
E3_MANIFEST = Path("dataset_real_e3/metadata/image_manifest.jsonl")
VISIBILITY = Path("fast_frex/burst_visibility_full.csv")
VLM_PREDICTIONS = {
    "e2b": Path("results_real_binary_full_2b/predictions.jsonl"),
    "e4b": Path("results_real_binary_full_4b/predictions.jsonl"),
}
WINDOWS_PER_FOLD = 5000
VALIDATION_FRACTION = 0.10
SIGMA_RANGE = (3.0, 60.0)
WIDTH_RANGE_MS = (1.0, 30.0)
TAU_RANGE_S = (0.0, 0.005)
BATCH_SIZE = 16
EPOCHS = 10
PATIENCE = 3
LEARNING_RATE = 1e-4
C4_BINS = ("[5, 8)", "[8, 15)")
C4_MIN_DELTA = 0.10
OFFBURST_MARGIN_S = 0.5
_STREAMS = {"folds": 10, "windows": 11, "pulses": 12, "validation": 13, "positives": 14, "names": 15, "torch": 16, "redraw": 17}


def _rng(stream: str, *extra: int) -> np.random.Generator:
    return np.random.default_rng([REAL_SEED, _STREAMS[stream], *extra])


# ---------------------------------------------------------------------------
# plan


def file_folds(files: list[str]) -> dict[str, int]:
    order = _rng("folds").permutation(len(files))
    return {files[int(i)]: int(position >= len(files) // 2) for position, i in enumerate(order)}


def fold_plan(fold: int, files: list[str], layouts: dict[str, Any], p0: P0) -> list[dict[str, Any]]:
    windows, pulses = _rng("windows", fold), _rng("pulses", fold)
    validation = set(
        int(i)
        for i in _rng("validation", fold).choice(
            WINDOWS_PER_FOLD, size=int(VALIDATION_FRACTION * WINDOWS_PER_FOLD), replace=False
        )
    )
    names = _rng("names", fold).permutation(2 * WINDOWS_PER_FOLD)
    plan = []
    for index in range(WINDOWS_PER_FOLD):
        fits_path = files[int(windows.integers(len(files)))]
        layout = layouts[fits_path]
        start = int(windows.integers(0, layout["nsamp_total"] - p0.window_samples + 1))
        sigma, width_ms, broadband, band_fwhm, band_center, tau, dm, arrival = pulses.uniform(size=8)
        dm = DM_RANGE[0] + dm * (DM_RANGE[1] - DM_RANGE[0])
        window_s = p0.window_samples * layout["tsamp_s"]
        span = Pulse(t_top_s=0.0, dm=dm, fwhm_s=0.0).sweep_s(layout["f_low_mhz"], layout["f_high_mhz"])[1]
        limited = broadband >= 0.5
        plan.append(
            {
                "injection_id": f"cnn{fold}_{index:04d}",
                "fold": fold,
                "split": "validation" if index in validation else "train",
                "fits_path": fits_path,
                "start_sample": start,
                "target_sigma": float(np.exp(np.log(SIGMA_RANGE[0]) + sigma * np.log(SIGMA_RANGE[1] / SIGMA_RANGE[0]))),
                "fwhm_s": float(np.exp(np.log(WIDTH_RANGE_MS[0]) + width_ms * np.log(WIDTH_RANGE_MS[1] / WIDTH_RANGE_MS[0])) / 1e3),
                "band_fwhm_mhz": float(BAND_FWHM_RANGE_MHZ[0] + band_fwhm * np.diff(BAND_FWHM_RANGE_MHZ)[0]) if limited else None,
                "band_center_mhz": float(layout["visible_low_mhz"] + band_center * (layout["visible_high_mhz"] - layout["visible_low_mhz"])) if limited else None,
                "tau_1250_s": float(TAU_RANGE_S[0] + tau * (TAU_RANGE_S[1] - TAU_RANGE_S[0])),
                "dm": float(dm),
                "t_top_s": float(EDGE_MARGIN_S + arrival * (window_s - span - 2 * EDGE_MARGIN_S)),
                "injected_sample_id": f"train_{int(names[2 * index]):05d}",
                "control_sample_id": f"train_{int(names[2 * index + 1]):05d}",
            }
        )
    return plan


def assign_positives(sigma_by_id: dict[str, float]) -> dict[str, int]:
    """Cada positivo vai para um dos dois modelos, alternando dentro da faixa embaralhada."""
    rng = _rng("positives")
    assignment = {}
    for low, high in STRATA:
        members = sorted(i for i, s in sigma_by_id.items() if low <= s < high)
        for position, index in enumerate(rng.permutation(len(members))):
            assignment[members[int(index)]] = position % 2
    return assignment


def _plan(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    p0 = P0()
    files = negative_files()
    folds = file_folds(files)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        layouts = dict(zip(files, pool.map(_layout_summary, files, itertools.repeat(p0), chunksize=16)))
    plan = []
    for fold in (0, 1):
        plan += redraw_corrupted(
            fold_plan(fold, sorted(f for f, k in folds.items() if k == fold), layouts, p0),
            layouts,
            p0,
            stream=(_STREAMS["redraw"], fold),
            workers=args.workers,
        )
    write_jsonl(PLAN_PATH, plan)

    manifest = read_jsonl(PAPER_MANIFEST)
    with VISIBILITY.open(encoding="utf-8") as handle:
        sigma = {row["sample_id"]: float(row["peak_sigma"]) for row in csv.DictReader(handle)}
    positives = {r["sample_id"]: sigma[r["sample_id"]] for r in manifest if r["true_label"] == "FRB"}
    assignment = assign_positives(positives)
    write_jsonl(
        POSITIVES_PATH,
        [
            {"sample_id": sample_id, "model_fold": assignment[sample_id], "peak_sigma": positives[sample_id],
             "stratum": stratum_of(positives[sample_id])}
            for sample_id in sorted(assignment)
        ],
    )
    write_json(
        OUTPUT_DIR / "cnn_plan_manifest.json",
        run_manifest(
            experiment="E5-CNN-plan",
            project_root=project_root,
            seed=REAL_SEED,
            streams=_STREAMS,
            fold_files={str(k): sum(1 for v in folds.values() if v == k) for k in (0, 1)},
            fold_of_file=folds,
            windows_per_fold=WINDOWS_PER_FOLD,
            validation_fraction=VALIDATION_FRACTION,
            sigma_range=SIGMA_RANGE,
            width_range_ms=WIDTH_RANGE_MS,
            tau_range_s=TAU_RANGE_S,
            redrawn={e["injection_id"]: e["redrawn"] for e in plan if "redrawn" in e},
            plan={"path": PLAN_PATH.as_posix(), "sha256": sha256_file(PLAN_PATH)},
            positives={"path": POSITIVES_PATH.as_posix(), "sha256": sha256_file(POSITIVES_PATH)},
        ),
    )
    print(f"{len(plan)} pares de treino; {len(assignment)} positivos repartidos")


# ---------------------------------------------------------------------------
# render


def _render(args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    plan = [entry for entry in read_jsonl(PLAN_PATH) if entry["fold"] == args.fold]
    dataset_dir = WORKDIR / f"fold{args.fold}"
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = pool.map(render_injection, plan, itertools.repeat(dataset_dir), chunksize=1)
        for entry, result in tqdm(zip(plan, futures), total=len(plan), desc=f"Fold {args.fold}"):
            for record in result["records"]:
                records.append({**record, "split": entry["split"], "fold": entry["fold"],
                                "target_sigma": entry["target_sigma"]})
    manifest_path = dataset_dir / "metadata" / "image_manifest.jsonl"
    write_jsonl(manifest_path, records)
    write_json(
        OUTPUT_DIR / f"cnn_render_fold{args.fold}.json",
        run_manifest(
            experiment=f"E5-CNN-render-fold{args.fold}",
            project_root=project_root,
            plan={"path": PLAN_PATH.as_posix(), "sha256": sha256_file(PLAN_PATH)},
            manifest={"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            n_images=len(records),
        ),
    )
    print(f"{len(records)} PNGs em {dataset_dir}")


# ---------------------------------------------------------------------------
# train / score


def _torch_parts():
    import torch
    import torchvision
    from PIL import Image

    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
    std = torch.tensor(weights.transforms().std).view(3, 1, 1)

    class Images(torch.utils.data.Dataset):
        def __init__(self, paths: list[str], labels: list[int]) -> None:
            self.paths, self.labels = paths, labels

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            with Image.open(self.paths[index]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = (torch.from_numpy(array).permute(2, 0, 1) - mean) / std
            return tensor, torch.tensor(float(self.labels[index]))

    def model():
        net = torchvision.models.resnet18(weights=weights)
        net.fc = torch.nn.Linear(net.fc.in_features, 1)
        return net

    return torch, Images, model, weights


def _split_records(fold: int, split: str) -> list[dict[str, Any]]:
    manifest = read_jsonl(WORKDIR / f"fold{fold}" / "metadata" / "image_manifest.jsonl")
    return [r for r in manifest if r["split"] == split]


def _train(args: argparse.Namespace) -> None:
    torch, Images, make_model, weights = _torch_parts()
    project_root = Path.cwd().resolve()
    seed = int(_rng("torch", args.fold).integers(2**31))
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    def loader(split: str, shuffle: bool):
        records = _split_records(args.fold, split)
        data = Images([r["image_path"] for r in records], [int(r["true_label"] == "FRB") for r in records])
        generator = torch.Generator().manual_seed(seed)
        return torch.utils.data.DataLoader(
            data, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=args.loader_workers,
            generator=generator, pin_memory=True,
        )

    train, validation = loader("train", True), loader("validation", False)
    model = make_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    checkpoint = WORKDIR / f"resnet18_fold{args.fold}.pt"
    history, best, stale = [], float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train, desc=f"epoca {epoch}", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(images).squeeze(1), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(labels)
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for images, labels in validation:
                images, labels = images.to(device), labels.to(device)
                logits = model(images).squeeze(1).float()
                val_loss += float(loss_fn(logits, labels)) * len(labels)
                correct += int(((logits > 0) == (labels > 0.5)).sum())
        entry = {
            "epoch": epoch,
            "train_loss": train_loss / len(train.dataset),
            "validation_loss": val_loss / len(validation.dataset),
            "validation_accuracy": correct / len(validation.dataset),
        }
        history.append(entry)
        print(entry, flush=True)
        if entry["validation_loss"] < best:
            best, stale = entry["validation_loss"], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    write_json(
        OUTPUT_DIR / f"cnn_train_fold{args.fold}.json",
        run_manifest(
            experiment=f"E5-CNN-train-fold{args.fold}",
            project_root=project_root,
            seed=seed,
            weights=str(weights),
            hyperparameters={"batch_size": BATCH_SIZE, "epochs": EPOCHS, "patience": PATIENCE,
                             "learning_rate": LEARNING_RATE, "optimizer": "Adam", "precision": "float16 autocast"},
            n_train=len(train.dataset),
            n_validation=len(validation.dataset),
            history=history,
            best_validation_loss=best,
            checkpoint={"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        ),
    )


def _score(args: argparse.Namespace) -> None:
    torch, Images, make_model, _ = _torch_parts()
    project_root = Path.cwd().resolve()
    device = torch.device("cuda")
    model = make_model()
    checkpoint = WORKDIR / f"resnet18_fold{args.fold}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()

    folds = read_json_field(OUTPUT_DIR / "cnn_plan_manifest.json", "fold_of_file")
    positives = {r["sample_id"]: r["model_fold"] for r in read_jsonl(POSITIVES_PATH)}
    test = [
        r for r in read_jsonl(PAPER_MANIFEST)
        if (r["true_label"] == "FRB" and positives[r["sample_id"]] == args.fold)
        or (r["true_label"] != "FRB" and folds[r["fits_path"]] != args.fold)
    ]
    controls = [r for r in _split_records(args.fold, "validation") if r["role"] == "control"]
    rows = [("test", r["sample_id"], r["image_path"], r["true_label"]) for r in test]
    rows += [("validation_control", r["sample_id"], r["image_path"], r["true_label"]) for r in controls]
    data = Images([row[2] for row in rows], [int(row[3] == "FRB") for row in rows])
    scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for images, _ in torch.utils.data.DataLoader(data, batch_size=32, num_workers=args.loader_workers):
            scores += torch.sigmoid(model(images.to(device)).squeeze(1).float()).cpu().tolist()
    output = OUTPUT_DIR / f"cnn_scores_fold{args.fold}.jsonl"
    write_jsonl(
        output,
        [
            {"set": s, "sample_id": i, "true_label": label, "model_fold": args.fold, "cnn_probability": p}
            for (s, i, _, label), p in zip(rows, scores)
        ],
    )
    write_json(
        OUTPUT_DIR / f"cnn_score_fold{args.fold}.json",
        run_manifest(
            experiment=f"E5-CNN-score-fold{args.fold}",
            project_root=project_root,
            checkpoint={"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            scores={"path": output.as_posix(), "sha256": sha256_file(output)},
            n_test=len(test),
            n_validation_controls=len(controls),
        ),
    )
    print(f"fold {args.fold}: {len(test)} imagens de teste, {len(controls)} controles de validacao")


def read_json_field(path: Path, field: str) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))[field]


# ---------------------------------------------------------------------------
# analyze


def threshold_at_fpr(control_scores: np.ndarray, target_fpr: float) -> float:
    """Menor limiar cuja FPR nos controles da validacao nao passa da FPR alvo."""
    for threshold in np.append(np.unique(control_scores), np.inf):
        if np.mean(control_scores >= threshold) <= target_fpr + 1e-12:
            return float(threshold)
    return float("inf")


def c4_rule(reference_hits: list[bool], vlm_hits: list[bool]) -> dict[str, Any]:
    """Delta = recall da referencia - recall do VLM nos bins de 5-15 sigma, nas mesmas imagens."""
    counts = PairedCounts.from_outcomes(reference_hits, vlm_hits)
    test = mcnemar_exact(counts)
    delta = newcombe_paired_difference(counts)
    return {
        "n": counts.n,
        "delta_recall": delta.as_dict(),
        "mcnemar": test,
        "passes": delta.estimate >= C4_MIN_DELTA and test["p_value_exact"] < 0.05,
    }


def cnn_scores() -> tuple[dict[str, dict[str, Any]], dict[int, np.ndarray]]:
    """Scores de teste por sample_id e os scores dos controles da validacao de cada modelo."""
    scores = read_jsonl(OUTPUT_DIR / "cnn_scores_fold0.jsonl") + read_jsonl(OUTPUT_DIR / "cnn_scores_fold1.jsonl")
    test = {r["sample_id"]: r for r in scores if r["set"] == "test"}
    controls = {
        fold: np.array([r["cnn_probability"] for r in scores if r["set"] == "validation_control" and r["model_fold"] == fold])
        for fold in (0, 1)
    }
    return test, controls


def vlm_decisions(path: Path) -> tuple[dict[str, bool], float]:
    """Decisao do VLM a 0,5 por sample_id e a FPR dele nos negativos reais."""
    records = {r["sample_id"]: r for r in read_jsonl(path)}
    detect = {i: float(r["parsed_response"]["frb_probability"]) >= 0.5 for i, r in records.items()}
    negatives = [i for i, r in records.items() if r["true_label"] != "FRB"]
    return detect, float(np.mean([detect[i] for i in negatives]))


def cnn_thresholds(controls: dict[int, np.ndarray], vlm_fpr: float) -> dict[str, dict[int, float]]:
    """Limiar 0,5 e o limiar casado com a FPR do VLM, escolhido nos controles de cada modelo."""
    return {"0.5": {0: 0.5, 1: 0.5},
            "vlm_fpr": {fold: threshold_at_fpr(controls[fold], vlm_fpr) for fold in (0, 1)}}


def _analyze(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    test, controls = cnn_scores()
    positives = {r["sample_id"]: r for r in read_jsonl(POSITIVES_PATH)}
    results: dict[str, Any] = {}
    for vlm, path in VLM_PREDICTIONS.items():
        vlm_detect, vlm_fpr = vlm_decisions(path)
        entry: dict[str, Any] = {"vlm_fpr_real": vlm_fpr}
        for name, by_fold in cnn_thresholds(controls, vlm_fpr).items():
            detect = {i: r["cnn_probability"] >= by_fold[r["model_fold"]] for i, r in test.items()}
            by_bin = {}
            for low, high in STRATA:
                label = stratum_of(low)
                members = sorted(i for i, p in positives.items() if p["stratum"] == label)
                cnn = [detect[i] for i in members]
                model = [vlm_detect[i] for i in members]
                by_bin[label] = {
                    "n": len(members),
                    "cnn": wilson(sum(cnn), len(members)).as_dict(),
                    vlm: wilson(sum(model), len(members)).as_dict(),
                    "mcnemar": mcnemar_exact(PairedCounts.from_outcomes(cnn, model)),
                }
            faint = sorted(i for i, p in positives.items() if p["stratum"] in C4_BINS)
            negatives_test = [i for i in test if test[i]["true_label"] != "FRB"]
            entry[name] = {
                "thresholds_by_model_fold": by_fold,
                "cnn_fpr_real": wilson(sum(detect[i] for i in negatives_test), len(negatives_test)).as_dict(),
                "recall_by_sigma_bin": by_bin,
                "c4_rule_5_to_15_sigma": c4_rule([detect[i] for i in faint], [vlm_detect[i] for i in faint]),
            }
        results[vlm] = entry
    write_json(
        OUTPUT_DIR / "cnn_results.json",
        run_manifest(experiment="E5-CNN-analysis", project_root=project_root, results=results),
    )
    for vlm, entry in results.items():
        for name in ("0.5", "vlm_fpr"):
            rule = entry[name]["c4_rule_5_to_15_sigma"]
            print(f"{vlm} limiar {name}: delta {rule['delta_recall']['estimate']:+.3f}, "
                  f"p = {rule['mcnemar']['p_value_exact']:.3g}, passa = {rule['passes']}")


# ---------------------------------------------------------------------------
# controle fora do burst (secao 10.12, acrescentado depois do resultado da CNN)


def off_burst(record: dict[str, Any], sweep: Sweep, margin: float = OFFBURST_MARGIN_S) -> bool:
    """A janela nao toca o sweep, com folga de ``margin`` segundos dos dois lados."""
    return (
        float(record["window_stop_seconds"]) < sweep.start - margin
        or float(record["window_start_seconds"]) > sweep.stop + margin
    )


def control_windows(fold: int, project_root: Path) -> list[tuple[str, dict[str, Any]]]:
    """(papel, registro) das janelas do E3 que o modelo ``fold`` pontua no controle.

    Fora do burst: as janelas dos arquivos de burst atribuidos a esse modelo que nao tocam o
    sweep da banda inteira do catalogo (o mesmo do E3). Negativas: todas as janelas dos
    arquivos negativos do outro fold, como no teste.
    """
    folds = read_json_field(OUTPUT_DIR / "cnn_plan_manifest.json", "fold_of_file")
    model_of = {r["sample_id"]: r["model_fold"] for r in read_jsonl(POSITIVES_PATH)}
    paper_id = {r["source_sample_id"]: r["sample_id"] for r in read_jsonl(PAPER_MANIFEST)}
    sweeps: dict[str, Sweep] = {}
    rows = []
    for record in read_jsonl(E3_MANIFEST):
        source = record["source_sample_id"]
        if record["true_label"] != "FRB":
            if folds[record["fits_path"]] != fold:
                rows.append(("negative", record))
            continue
        if model_of[paper_id[source]] != fold:
            continue
        if source not in sweeps:
            layout = read_layout(project_root / record["fits_path"])
            parameters = record["real_parameters"]
            sweeps[source] = catalog_sweeps(
                toa_s=float(parameters["toa_s"]),
                dm=float(parameters["dm"]),
                frequencies_mhz=np.asarray(layout.frequencies_mhz),
                edge_channels=int(record["preprocessing"]["edge_channels"]),
            )["full_band"]
        if off_burst(record, sweeps[source]):
            rows.append(("off_burst", record))
    return rows


def _score_offburst(args: argparse.Namespace) -> None:
    torch, Images, make_model, _ = _torch_parts()
    project_root = Path.cwd().resolve()
    device = torch.device("cuda")
    model = make_model()
    checkpoint = WORKDIR / f"resnet18_fold{args.fold}.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device).eval()

    rows = control_windows(args.fold, project_root)
    data = Images([r["image_path"] for _, r in rows], [0] * len(rows))
    scores = []
    # a mesma pontuacao do ``score``
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for images, _ in torch.utils.data.DataLoader(data, batch_size=32, num_workers=args.loader_workers):
            scores += torch.sigmoid(model(images.to(device)).squeeze(1).float()).cpu().tolist()
    output = OUTPUT_DIR / f"cnn_offburst_scores_fold{args.fold}.jsonl"
    write_jsonl(
        output,
        [
            {"role": role, "sample_id": r["sample_id"], "file": r["source_sample_id"], "model_fold": args.fold,
             "cnn_probability": p}
            for (role, r), p in zip(rows, scores)
        ],
    )
    counts = {role: sum(row[0] == role for row in rows) for role in ("off_burst", "negative")}
    write_json(
        OUTPUT_DIR / f"cnn_offburst_score_fold{args.fold}.json",
        run_manifest(
            experiment=f"E5-CNN-offburst-score-fold{args.fold}",
            project_root=project_root,
            checkpoint={"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            e3_manifest={"path": E3_MANIFEST.as_posix(), "sha256": sha256_file(E3_MANIFEST)},
            margin_s=OFFBURST_MARGIN_S,
            scores={"path": output.as_posix(), "sha256": sha256_file(output)},
            windows=counts,
        ),
    )
    print(f"fold {args.fold}: {counts['off_burst']} janelas fora do burst, {counts['negative']} negativas")


def _interval(estimate: float, draws: np.ndarray) -> dict[str, float]:
    return {"estimate": float(estimate), "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5))}


def window_rates(
    off_burst_hits: list[tuple[str, bool]],
    negative_hits: list[tuple[str, bool]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fracao de janelas sinalizadas fora do burst e nas negativas, e a diferenca.

    Cada item e (arquivo, sinalizada). As janelas de um arquivo nao sao independentes, entao o
    IC 95% vem de um bootstrap percentil que reamostra arquivos dentro de cada grupo.
    """
    groups = {}
    for name, hits in (("off_burst", off_burst_hits), ("negative", negative_hits)):
        flagged: dict[str, int] = {}
        windows: dict[str, int] = {}
        for file, hit in hits:
            flagged[file] = flagged.get(file, 0) + int(hit)
            windows[file] = windows.get(file, 0) + 1
        files = sorted(windows)
        groups[name] = (np.array([flagged[f] for f in files]), np.array([windows[f] for f in files]))
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(iterations) for name in groups}
    for iteration in range(iterations):
        for name, (k, n) in groups.items():
            index = rng.integers(0, k.size, k.size)
            draws[name][iteration] = k[index].sum() / n[index].sum()
    result: dict[str, Any] = {}
    for name, (k, n) in groups.items():
        result[name] = {"files": int(k.size), "windows": int(n.sum()), "flagged": int(k.sum()),
                        **_interval(k.sum() / n.sum(), draws[name])}
    result["off_burst_minus_negative"] = _interval(
        result["off_burst"]["estimate"] - result["negative"]["estimate"], draws["off_burst"] - draws["negative"]
    )
    return result


def discounted_c4(cnn_hits: list[bool], vlm_hits: list[bool], off_burst_rate: float) -> dict[str, Any]:
    """Recall da CNN descontado de uma taxa f de disparo fora do burst, (r - f) / (1 - f)."""
    cnn_recall, vlm_recall = float(np.mean(cnn_hits)), float(np.mean(vlm_hits))
    discounted = (cnn_recall - off_burst_rate) / (1.0 - off_burst_rate)
    return {
        "n": len(cnn_hits),
        "cnn_recall": cnn_recall,
        "vlm_recall": vlm_recall,
        "off_burst_rate_used": off_burst_rate,
        "cnn_recall_discounted": discounted,
        "delta_discounted": discounted - vlm_recall,
        "holds": discounted - vlm_recall >= C4_MIN_DELTA,
    }


def _analyze_offburst(_args: argparse.Namespace) -> None:
    project_root = Path.cwd().resolve()
    test, controls = cnn_scores()
    positives = {r["sample_id"]: r for r in read_jsonl(POSITIVES_PATH)}
    faint = sorted(i for i, p in positives.items() if p["stratum"] in C4_BINS)
    windows = read_jsonl(OUTPUT_DIR / "cnn_offburst_scores_fold0.jsonl") + read_jsonl(
        OUTPUT_DIR / "cnn_offburst_scores_fold1.jsonl"
    )
    results: dict[str, Any] = {}
    for vlm, path in VLM_PREDICTIONS.items():
        vlm_detect, vlm_fpr = vlm_decisions(path)
        entry: dict[str, Any] = {"vlm_fpr_real": vlm_fpr}
        for name, by_fold in cnn_thresholds(controls, vlm_fpr).items():
            hits = {
                role: [(w["file"], w["cnn_probability"] >= by_fold[w["model_fold"]]) for w in windows if w["role"] == role]
                for role in ("off_burst", "negative")
            }
            rates = window_rates(hits["off_burst"], hits["negative"])
            entry[name] = {
                "thresholds_by_model_fold": by_fold,
                "window_rates": rates,
                # o limite superior do IC da taxa fora do burst, o desconto mais severo
                "c4_discounted": discounted_c4(
                    [test[i]["cnn_probability"] >= by_fold[test[i]["model_fold"]] for i in faint],
                    [vlm_detect[i] for i in faint],
                    rates["off_burst"]["high"],
                ),
            }
        results[vlm] = entry
    write_json(
        OUTPUT_DIR / "cnn_offburst_results.json",
        run_manifest(
            experiment="E5-CNN-offburst-analysis", project_root=project_root, margin_s=OFFBURST_MARGIN_S,
            bootstrap={"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED}, results=results,
        ),
    )
    for vlm, entry in results.items():
        for name in ("0.5", "vlm_fpr"):
            rates, c4 = entry[name]["window_rates"], entry[name]["c4_discounted"]
            print(f"{vlm} limiar {name}: fora do burst {rates['off_burst']['estimate']:.4f}, negativas "
                  f"{rates['negative']['estimate']:.4f}; delta descontado {c4['delta_discounted']:+.3f}, "
                  f"mantem = {c4['holds']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--workers", type=int, default=16)
    plan.set_defaults(func=_plan)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    render_parser.add_argument("--workers", type=int, default=16)
    render_parser.set_defaults(func=_render)
    for name, func in (("train", _train), ("score", _score), ("score-offburst", _score_offburst)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--fold", type=int, choices=(0, 1), required=True)
        sub.add_argument("--loader-workers", type=int, default=8)
        sub.set_defaults(func=func)
    subparsers.add_parser("analyze").set_defaults(func=_analyze)
    subparsers.add_parser("analyze-offburst").set_defaults(func=_analyze_offburst)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
