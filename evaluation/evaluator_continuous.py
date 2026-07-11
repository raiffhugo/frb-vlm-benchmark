from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from benchmark_predictions.compare import (
    extract_frb_probability,
    mcnemar_summary,
    score_metric_summary,
    threshold_curve,
)
from vlm_classifier.parser import TASK_FRB_BINARY, normalize_label, normalize_task


LOGGER = logging.getLogger(__name__)
DEFAULT_PRIORS = (1e-2, 1e-3, 1e-4)
POSITIVE_LABEL = "FRB"


@dataclass(frozen=True)
class ContinuousEvaluationConfig:
    predictions: tuple[Path, ...]
    output_dir: Path
    task: str = TASK_FRB_BINARY
    names: tuple[str, ...] | None = None
    threshold: float = 0.5
    priors: tuple[float, ...] = DEFAULT_PRIORS
    threshold_steps: int = 101
    calibration_bins: int = 10

    def validate(self) -> None:
        task = normalize_task(self.task)
        if task != TASK_FRB_BINARY:
            raise ValueError("Continuous evaluation only supports the frb-binary task.")
        if not 1 <= len(self.predictions) <= 2:
            raise ValueError("Provide 1 or 2 prediction files.")
        if self.names is not None and len(self.names) != len(self.predictions):
            raise ValueError("names must have the same length as predictions.")
        if self.names is not None and len(set(self.names)) != len(self.names):
            raise ValueError("names must not contain duplicated values.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if self.threshold_steps < 2:
            raise ValueError("threshold_steps must be >= 2.")
        for prior in self.priors:
            if not 0.0 < prior < 1.0:
                raise ValueError("priors must be in (0, 1).")


class ContinuousPredictionEvaluator:
    def __init__(
        self,
        *,
        config: ContinuousEvaluationConfig,
        project_root: Path,
    ) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.prediction_paths = tuple(self._resolve(path) for path in config.predictions)
        self.output_dir = self._resolve(config.output_dir)
        self.names = config.names or tuple(path.stem for path in self.prediction_paths)

    def evaluate(self, *, overwrite: bool = False) -> dict[str, Path]:
        datasets = [
            self._score_dataset(path, name)
            for path, name in zip(self.prediction_paths, self.names)
        ]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = self._output_paths()
        if overwrite:
            self._reset_outputs(outputs)
        else:
            self._check_outputs(outputs)

        metrics = self._build_metrics(datasets)
        outputs["metrics_continuous"].write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._write_roc_plot(outputs["roc_curve"], datasets)
        self._write_pr_plot(outputs["precision_recall_curve"], datasets)
        self._write_threshold_plot(outputs["threshold_curves"], datasets)
        LOGGER.info("Continuous evaluation finished in %s", self.output_dir)
        return outputs

    def _score_dataset(self, path: Path, name: str) -> dict[str, Any]:
        records = _load_prediction_jsonl(path)
        rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for record in records:
            sample_id = str(record.get("sample_id"))
            if record.get("error"):
                exclusions.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"model_error:{record.get('error')}",
                    }
                )
                continue
            score = extract_frb_probability(record)
            if score is None:
                exclusions.append(
                    {
                        "sample_id": sample_id,
                        "reason": "missing_or_invalid_frb_probability",
                    }
                )
                continue
            try:
                true_label = normalize_label(
                    record.get("true_label"),
                    task=TASK_FRB_BINARY,
                )
            except ValueError as exc:
                exclusions.append({"sample_id": sample_id, "reason": str(exc)})
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_true_label": record.get("source_true_label") or true_label,
                    "true_label": true_label,
                    "y_true": 1 if true_label == POSITIVE_LABEL else 0,
                    "score": score,
                }
            )
        if not rows:
            raise ValueError(f"No valid score in {path}")
        return {"name": name, "path": path, "rows": rows, "exclusions": exclusions}

    def _build_metrics(self, datasets: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task": TASK_FRB_BINARY,
            "score_source": "parsed_response.frb_probability",
            "threshold": self.config.threshold,
            "priors": list(self.config.priors),
            "models": {},
        }
        for dataset in datasets:
            y_true = np.asarray([row["y_true"] for row in dataset["rows"]], dtype=int)
            scores = np.asarray([row["score"] for row in dataset["rows"]], dtype=float)
            threshold_rows = _enriched_threshold_curve(
                y_true,
                scores,
                steps=self.config.threshold_steps,
            )
            f1_optimal = max(threshold_rows, key=lambda row: (row["f1"], row["threshold"]))
            youden_optimal = max(
                threshold_rows,
                key=lambda row: (row["youden_j"], row["threshold"]),
            )
            result["models"][dataset["name"]] = {
                "path": str(dataset["path"]),
                "evaluated_count": int(len(y_true)),
                "exclusion_count": len(dataset["exclusions"]),
                "class_balance": {
                    "positive_count": int(y_true.sum()),
                    "negative_count": int(len(y_true) - y_true.sum()),
                    "positive_fraction": float(y_true.mean()),
                },
                "score_metrics": score_metric_summary(
                    y_true,
                    scores,
                    bootstrap_iterations=0,
                    bootstrap_seed=42,
                    calibration_bins=self.config.calibration_bins,
                ),
                "operating_points": {
                    "threshold_current": _single_threshold_metrics(
                        y_true,
                        scores,
                        self.config.threshold,
                    ),
                    "f1_optimal": f1_optimal,
                    "youden_j_optimal": youden_optimal,
                },
                "prior_adjusted": {
                    str(prior): _prior_adjusted_metrics(
                        y_true,
                        scores,
                        prior=prior,
                        threshold=self.config.threshold,
                    )
                    for prior in self.config.priors
                },
                "threshold_curve": threshold_rows,
                "exclusions": dataset["exclusions"],
            }
        if len(datasets) == 2:
            result["paired_comparison"] = self._paired_comparison(datasets)
        return result

    def _paired_comparison(self, datasets: list[dict[str, Any]]) -> dict[str, Any]:
        left, right = datasets
        left_rows = {row["sample_id"]: row for row in left["rows"]}
        right_rows = {row["sample_id"]: row for row in right["rows"]}
        sample_ids = sorted(set(left_rows) & set(right_rows))
        paired: list[dict[str, Any]] = []
        left_correct: list[bool] = []
        right_correct: list[bool] = []
        for sample_id in sample_ids:
            a = left_rows[sample_id]
            b = right_rows[sample_id]
            if a["y_true"] != b["y_true"]:
                continue
            pred_a = a["score"] >= self.config.threshold
            pred_b = b["score"] >= self.config.threshold
            truth = bool(a["y_true"])
            left_correct.append(pred_a == truth)
            right_correct.append(pred_b == truth)
            paired.append(
                {
                    "sample_id": sample_id,
                    "y_true": a["y_true"],
                    left["name"]: a["score"],
                    right["name"]: b["score"],
                }
            )
        return {
            "left": left["name"],
            "right": right["name"],
            "paired_count": len(paired),
            "mcnemar_at_threshold": mcnemar_summary(left_correct, right_correct),
            "paired_scores": paired,
            "left_only_sample_ids": sorted(set(left_rows) - set(right_rows)),
            "right_only_sample_ids": sorted(set(right_rows) - set(left_rows)),
        }

    def _write_roc_plot(self, path: Path, datasets: list[dict[str, Any]]) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        for dataset in datasets:
            y_true, scores = _arrays(dataset)
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _thresholds = roc_curve(y_true, scores)
            auc = score_metric_summary(
                y_true,
                scores,
                bootstrap_iterations=0,
                bootstrap_seed=42,
                calibration_bins=self.config.calibration_bins,
            )["roc_auc"]["estimate"]
            ax.plot(fpr, tpr, label=f"{dataset['name']} AUC={_fmt(auc)}")
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC over frb_probability")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_pr_plot(self, path: Path, datasets: list[dict[str, Any]]) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        for dataset in datasets:
            y_true, scores = _arrays(dataset)
            if len(np.unique(y_true)) < 2:
                continue
            precision, recall, _thresholds = precision_recall_curve(y_true, scores)
            ap = score_metric_summary(
                y_true,
                scores,
                bootstrap_iterations=0,
                bootstrap_seed=42,
                calibration_bins=self.config.calibration_bins,
            )["average_precision"]["estimate"]
            ax.plot(recall, precision, label=f"{dataset['name']} AP={_fmt(ap)}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall over frb_probability")
        ax.legend(loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_threshold_plot(self, path: Path, datasets: list[dict[str, Any]]) -> None:
        fig, axes = plt.subplots(
            1,
            len(datasets),
            figsize=(6 * len(datasets), 4.5),
            dpi=150,
            squeeze=False,
        )
        for ax, dataset in zip(axes[0], datasets):
            y_true, scores = _arrays(dataset)
            rows = _enriched_threshold_curve(
                y_true,
                scores,
                steps=self.config.threshold_steps,
            )
            thresholds = [row["threshold"] for row in rows]
            for metric in ("precision", "recall", "f1", "specificity"):
                ax.plot(thresholds, [row[metric] for row in rows], label=metric)
            ax.axvline(self.config.threshold, color="0.35", linestyle="--", linewidth=1)
            ax.set_title(dataset["name"])
            ax.set_xlabel("Threshold")
            ax.grid(alpha=0.25)
        axes[0][0].set_ylabel("Metric")
        axes[0][-1].legend(loc="lower left")
        fig.suptitle("Varredura de threshold")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _output_paths(self) -> dict[str, Path]:
        return {
            "metrics_continuous": self.output_dir / "metrics_continuous.json",
            "roc_curve": self.output_dir / "roc_curve.png",
            "precision_recall_curve": self.output_dir / "precision_recall_curve.png",
            "threshold_curves": self.output_dir / "threshold_curves.png",
        }

    def _reset_outputs(self, outputs: dict[str, Path]) -> None:
        for path in outputs.values():
            if path.exists():
                path.unlink()
                LOGGER.info("Old artifact removed: %s", path)

    def _check_outputs(self, outputs: dict[str, Path]) -> None:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Continuous evaluation files already exist: {joined}. Use --overwrite."
            )

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path


def _load_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"Line {line_number} is not a JSON object: {path}")
            records.append(parsed)
    if not records:
        raise ValueError(f"No predictions found in {path}")
    return records


def _arrays(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray([row["y_true"] for row in dataset["rows"]], dtype=int)
    scores = np.asarray([row["score"] for row in dataset["rows"]], dtype=float)
    return y_true, scores


def _enriched_threshold_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    steps: int,
) -> list[dict[str, Any]]:
    rows = threshold_curve(y_true, scores, steps=steps)
    for row in rows:
        total = row["tp"] + row["fp"] + row["tn"] + row["fn"]
        row["accuracy"] = float((row["tp"] + row["tn"]) / total) if total else 0.0
        row["specificity"] = (
            float(row["tn"] / (row["tn"] + row["fp"]))
            if (row["tn"] + row["fp"])
            else 0.0
        )
        row["youden_j"] = row["recall"] + row["specificity"] - 1.0
    return rows


def _single_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = scores >= threshold
    tp = int(np.sum(y_pred & (y_true == 1)))
    fp = int(np.sum(y_pred & (y_true == 0)))
    tn = int(np.sum(~y_pred & (y_true == 0)))
    fn = int(np.sum(~y_pred & (y_true == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    total = tp + fp + tn + fn
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "youden_j": recall + specificity - 1.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _prior_adjusted_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    prior: float,
    threshold: float,
) -> dict[str, Any]:
    rows = _enriched_threshold_curve(y_true, scores, steps=101)
    adjusted_rows = []
    for row in rows:
        fpr = (
            float(row["fp"] / (row["fp"] + row["tn"]))
            if (row["fp"] + row["tn"])
            else 0.0
        )
        tpr = row["recall"]
        denominator = prior * tpr + (1.0 - prior) * fpr
        precision = float((prior * tpr) / denominator) if denominator else 0.0
        adjusted_rows.append({**row, "precision_at_prior": precision, "fpr": fpr})
    recall = np.asarray([row["recall"] for row in adjusted_rows], dtype=float)
    precision = np.asarray([row["precision_at_prior"] for row in adjusted_rows], dtype=float)
    order = np.argsort(recall)
    pr_auc = float(np.trapezoid(precision[order], recall[order]))
    at_threshold = min(adjusted_rows, key=lambda row: abs(row["threshold"] - threshold))
    return {
        "prior": prior,
        "threshold": threshold,
        "approx_pr_auc": pr_auc,
        "precision_at_threshold": at_threshold["precision_at_prior"],
        "recall_at_threshold": at_threshold["recall"],
        "note": "Precision e PR-AUC reescaladas a partir da curva empirica de TPR/FPR.",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"
