from __future__ import annotations

import csv
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from vlm_classifier.parser import (
    TASK_MULTICLASS,
    classes_for_task,
    normalize_label,
    normalize_task,
)


LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class EvaluationConfig:
    predictions_path: Path
    output_dir: Path
    task: str = TASK_MULTICLASS


class PredictionEvaluator:
    def __init__(self, *, config: EvaluationConfig, project_root: Path) -> None:
        self.config = config
        self.task = normalize_task(config.task)
        self.classes = classes_for_task(self.task)
        self.project_root = project_root.resolve()
        self.predictions_path = self._resolve(config.predictions_path)
        self.output_dir = self._resolve(config.output_dir)
        self.metrics_json = self.output_dir / "metrics.json"
        self.classification_report_txt = self.output_dir / "classification_report.txt"
        self.confusion_matrix_png = self.output_dir / "confusion_matrix.png"
        self.summary_csv = self.output_dir / "summary.csv"
        self.evaluation_report_pdf = self.output_dir / "evaluation_report.pdf"

    def evaluate(self, *, overwrite: bool = False) -> dict[str, Path]:
        predictions = self._load_predictions()
        valid_records, skipped_records = self._split_valid_and_skipped(predictions)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self._reset_outputs()
        else:
            self._check_outputs()

        y_true = [record["true_label"] for record in valid_records]
        y_pred = [record["predicted_label"] for record in valid_records]
        metrics = self._compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            total_count=len(predictions),
            skipped_records=skipped_records,
        )

        self._write_metrics(metrics)
        self._write_classification_report(
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            skipped_records=skipped_records,
        )
        self._write_confusion_matrix_png(metrics["confusion_matrix"])
        self._write_summary_csv(metrics)
        self._write_evaluation_report_pdf(metrics)

        LOGGER.info(
            "Evaluation finished: %d valid, %d skipped.",
            metrics["evaluated_count"],
            metrics["skipped_count"],
        )
        return {
            "metrics_json": self.metrics_json,
            "classification_report": self.classification_report_txt,
            "confusion_matrix_png": self.confusion_matrix_png,
            "summary_csv": self.summary_csv,
            "evaluation_report_pdf": self.evaluation_report_pdf,
        }

    def _load_predictions(self) -> list[dict[str, Any]]:
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")

        records: list[dict[str, Any]] = []
        with self.predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.predictions_path}:{line_number}"
                    ) from exc
                self._validate_prediction_record(record, line_number)
                records.append(record)

        if not records:
            raise ValueError(f"No predictions found in {self.predictions_path}")
        return records

    def _validate_prediction_record(self, record: dict[str, Any], line_number: int) -> None:
        required = {
            "sample_id",
            "image_path",
            "true_label",
            "predicted_label",
            "confidence",
            "raw_model_response",
            "parsed_response",
            "error",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Line {line_number} is missing required keys: {', '.join(missing)}"
            )

    def _split_valid_and_skipped(
        self,
        predictions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        valid_records: list[dict[str, Any]] = []
        skipped_records: list[dict[str, Any]] = []

        for record in predictions:
            error = record.get("error")
            if error:
                skipped_records.append(
                    self._skipped_record(record, reason=f"model_error: {error}")
                )
                continue

            try:
                true_label = normalize_label(record.get("true_label"), task=self.task)
                predicted_label = normalize_label(
                    record.get("predicted_label"),
                    task=self.task,
                )
            except ValueError as exc:
                skipped_records.append(self._skipped_record(record, reason=str(exc)))
                continue

            valid = dict(record)
            valid["true_label"] = true_label
            valid["predicted_label"] = predicted_label
            valid_records.append(valid)

        return valid_records, skipped_records

    def _skipped_record(self, record: dict[str, Any], *, reason: str) -> dict[str, Any]:
        return {
            "sample_id": record.get("sample_id"),
            "image_path": record.get("image_path"),
            "true_label": record.get("true_label"),
            "predicted_label": record.get("predicted_label"),
            "reason": reason,
        }

    def _compute_metrics(
        self,
        *,
        y_true: list[str],
        y_pred: list[str],
        total_count: int,
        skipped_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if y_true:
            accuracy = float(accuracy_score(y_true, y_pred))
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(self.classes),
                zero_division=0,
            )
            macro_f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(self.classes),
                    average="macro",
                    zero_division=0,
                )
            )
            weighted_f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=list(self.classes),
                    average="weighted",
                    zero_division=0,
                )
            )
            matrix = confusion_matrix(y_true, y_pred, labels=list(self.classes))
        else:
            accuracy = None
            precision = np.zeros(len(self.classes), dtype=float)
            recall = np.zeros(len(self.classes), dtype=float)
            f1 = np.zeros(len(self.classes), dtype=float)
            support = np.zeros(len(self.classes), dtype=int)
            macro_f1 = None
            weighted_f1 = None
            matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)

        per_class = {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(self.classes)
        }

        error_count = sum(1 for record in skipped_records if str(record["reason"]).startswith("model_error:"))
        invalid_count = len(skipped_records) - error_count
        return {
            "task": self.task,
            "classes": list(self.classes),
            "total_count": total_count,
            "evaluated_count": len(y_true),
            "skipped_count": len(skipped_records),
            "error_count": error_count,
            "invalid_count": invalid_count,
            "accuracy": accuracy,
            "precision_per_class": {
                label: per_class[label]["precision"] for label in self.classes
            },
            "recall_per_class": {
                label: per_class[label]["recall"] for label in self.classes
            },
            "f1_per_class": {
                label: per_class[label]["f1_score"] for label in self.classes
            },
            "per_class": per_class,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "confusion_matrix": matrix.astype(int).tolist(),
            "skipped_samples": skipped_records,
        }

    def _write_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics_json.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_classification_report(
        self,
        *,
        y_true: list[str],
        y_pred: list[str],
        metrics: dict[str, Any],
        skipped_records: list[dict[str, Any]],
    ) -> None:
        lines = [
            "FRB SVLM classification report",
            "",
            f"Total samples: {metrics['total_count']}",
            f"Evaluated samples: {metrics['evaluated_count']}",
            f"Skipped samples: {metrics['skipped_count']}",
            f"Samples with model error: {metrics['error_count']}",
            f"Samples with invalid labels: {metrics['invalid_count']}",
            "",
        ]
        if y_true:
            lines.append(
                classification_report(
                    y_true,
                    y_pred,
                    labels=list(self.classes),
                    target_names=list(self.classes),
                    digits=4,
                    zero_division=0,
                )
            )
        else:
            lines.append("No valid predictions available for metric computation.")

        if skipped_records:
            lines.extend(["", "Skipped samples:"])
            for record in skipped_records:
                lines.append(
                    f"- {record['sample_id']}: {record['reason']}"
                )

        self.classification_report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_confusion_matrix_png(self, matrix: list[list[int]]) -> None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix, dtype=int),
            display_labels=list(self.classes),
        )
        display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.confusion_matrix_png)
        plt.close(fig)

    def _write_summary_csv(self, metrics: dict[str, Any]) -> None:
        fieldnames = ["row", "precision", "recall", "f1_score", "support"]
        with self.summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label in self.classes:
                row = metrics["per_class"][label]
                writer.writerow(
                    {
                        "row": label,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1_score": row["f1_score"],
                        "support": row["support"],
                    }
                )
            writer.writerow(
                {
                    "row": "accuracy",
                    "precision": "",
                    "recall": "",
                    "f1_score": metrics["accuracy"],
                    "support": metrics["evaluated_count"],
                }
            )
            writer.writerow(
                {
                    "row": "macro_avg",
                    "precision": "",
                    "recall": "",
                    "f1_score": metrics["macro_f1"],
                    "support": metrics["evaluated_count"],
                }
            )
            writer.writerow(
                {
                    "row": "weighted_avg",
                    "precision": "",
                    "recall": "",
                    "f1_score": metrics["weighted_f1"],
                    "support": metrics["evaluated_count"],
                }
            )
            writer.writerow(
                {
                    "row": "skipped",
                    "precision": "",
                    "recall": "",
                    "f1_score": "",
                    "support": metrics["skipped_count"],
                }
            )

    _PDF_PAGE_SIZE = (8.27, 11.69)
    _PDF_MARGIN = 0.08
    _PDF_INK = "#1A1A1A"
    _PDF_MUTED = "#555555"
    _PDF_ACCENT = "#1F4E79"
    _PDF_TABLE_HEADER_BG = "#EAF2F8"
    _PDF_TABLE_ROW_BG = "#F5F7FA"
    _PDF_TABLE_EDGE = "#D7DEE6"

    def _write_evaluation_report_pdf(self, metrics: dict[str, Any]) -> None:
        with PdfPages(self.evaluation_report_pdf) as pdf:
            self._add_pdf_summary_page(pdf, metrics)
            self._add_pdf_reading_guide_page(pdf, metrics["confusion_matrix"])

    def _add_pdf_summary_page(self, pdf: PdfPages, metrics: dict[str, Any]) -> None:
        fig = plt.figure(figsize=self._PDF_PAGE_SIZE)
        fig.patch.set_facecolor("white")
        left = self._PDF_MARGIN
        width = 1.0 - 2 * self._PDF_MARGIN

        fig.text(
            left, 0.945, "Evaluation Report",
            fontsize=20, weight="bold", color=self._PDF_INK,
        )
        fig.text(
            left, 0.922, "Prediction summary",
            fontsize=10.5, color=self._PDF_MUTED,
        )
        fig.add_artist(
            plt.Line2D(
                [left, left + width], [0.912, 0.912],
                transform=fig.transFigure,
                color=self._PDF_ACCENT, linewidth=1.6,
            )
        )

        y = 0.885
        summary_rows = [
            ["Total samples", str(metrics["total_count"])],
            ["Evaluated samples", str(metrics["evaluated_count"])],
            ["Skipped samples", str(metrics["skipped_count"])],
            ["Model errors", str(metrics["error_count"])],
            ["Invalid labels", str(metrics["invalid_count"])],
            ["Accuracy", _format_metric(metrics["accuracy"])],
            ["Macro-F1", _format_metric(metrics["macro_f1"])],
            ["Weighted-F1", _format_metric(metrics["weighted_f1"])],
        ]
        y = self._pdf_section(
            fig, y, "Summary", ["Indicator", "Value"],
            summary_rows, col_widths=(0.60, 0.40),
        )

        per_class_rows = [
            [
                label,
                _format_metric(metrics["per_class"][label]["precision"]),
                _format_metric(metrics["per_class"][label]["recall"]),
                _format_metric(metrics["per_class"][label]["f1_score"]),
                str(metrics["per_class"][label]["support"]),
            ]
            for label in self.classes
        ]
        y = self._pdf_section(
            fig, y, "Per-class metrics",
            ["Class", "Precision", "Recall", "F1-score", "Support"],
            per_class_rows, col_widths=(0.28, 0.18, 0.18, 0.18, 0.18),
        )

        interpretation = _build_performance_interpretation(
            metrics,
            classes=self.classes,
        )
        y = self._pdf_paragraph(fig, y, "Performance interpretation", interpretation)

        if metrics["skipped_count"]:
            skipped_text = (
                f"{metrics['skipped_count']} sample(s) were skipped in the metrics. "
                "These cases do not enter the denominator of accuracy, precision, "
                "recall, or F1."
            )
            self._pdf_paragraph(fig, y, "Skipped samples", skipped_text)

        pdf.savefig(fig)
        plt.close(fig)

    def _add_pdf_reading_guide_page(
        self, pdf: PdfPages, matrix: list[list[int]]
    ) -> None:
        fig = plt.figure(figsize=self._PDF_PAGE_SIZE)
        fig.patch.set_facecolor("white")
        left = self._PDF_MARGIN
        width = 1.0 - 2 * self._PDF_MARGIN

        y = 0.955
        fig.text(
            left, y, "How to read the metrics",
            fontsize=12.5, weight="bold", color=self._PDF_ACCENT,
        )
        fig.add_artist(
            plt.Line2D(
                [left, left + width], [y - 0.008, y - 0.008],
                transform=fig.transFigure,
                color=self._PDF_TABLE_EDGE, linewidth=1.0,
            )
        )
        classes_text = ", ".join(self.classes)
        explanations = [
            "Accuracy: fraction of evaluated samples whose predicted class matches the true label.",
            "Per-class precision: among all images predicted as a class, measures how many actually belonged to that class.",
            "Per-class recall: among all images that actually belonged to a class, measures how many the model recovered.",
            "Per-class F1-score: harmonic mean of precision and recall; penalizes scenarios where either metric is low.",
            f"Macro-F1: simple mean of the F1-scores of the classes {classes_text}; every class has the same weight.",
            "Weighted-F1: mean of the F1-scores weighted by the number of samples of each class.",
            "Support: number of evaluated samples of each true class.",
            "Confusion matrix: rows represent the true class and columns represent the predicted class.",
        ]
        y -= 0.032
        for explanation in explanations:
            wrapped = textwrap.wrap(
                f"- {explanation}", width=108, subsequent_indent="  "
            )
            fig.text(
                left, y, "\n".join(wrapped),
                fontsize=9.5, color=self._PDF_INK, va="top", linespacing=1.35,
            )
            y -= 0.0225 * len(wrapped) + 0.012

        y -= 0.02
        fig.text(
            left, y, "Confusion Matrix",
            fontsize=12.5, weight="bold", color=self._PDF_ACCENT,
        )
        fig.add_artist(
            plt.Line2D(
                [left, left + width], [y - 0.008, y - 0.008],
                transform=fig.transFigure,
                color=self._PDF_TABLE_EDGE, linewidth=1.0,
            )
        )
        page_width, page_height = self._PDF_PAGE_SIZE
        ax_width = 0.52
        ax_height = ax_width * page_width / page_height
        ax = fig.add_axes(
            [left + (width - ax_width) / 2, y - 0.035 - ax_height, ax_width, ax_height]
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(matrix, dtype=int),
            display_labels=list(self.classes),
        )
        display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
        ax.set_title(
            "Rows: true class | Columns: predicted class",
            fontsize=10, color=self._PDF_MUTED,
        )
        pdf.savefig(fig)
        plt.close(fig)

    def _pdf_section(
        self,
        fig: Any,
        y_top: float,
        title: str,
        col_labels: list[str],
        rows: list[list[str]],
        *,
        col_widths: tuple[float, ...],
    ) -> float:
        left = self._PDF_MARGIN
        width = 1.0 - 2 * self._PDF_MARGIN
        fig.text(
            left, y_top, title,
            fontsize=12.5, weight="bold", color=self._PDF_ACCENT,
        )
        row_height = 0.0235
        table_height = row_height * (len(rows) + 1)
        y_bottom = y_top - 0.016 - table_height
        ax = fig.add_axes([left, y_bottom, width, table_height])
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="left",
            colLoc="left",
            colWidths=list(col_widths),
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(self._PDF_TABLE_EDGE)
            cell.set_linewidth(0.6)
            cell.get_text().set_color(self._PDF_INK)
            if row == 0:
                cell.set_facecolor(self._PDF_TABLE_HEADER_BG)
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(self._PDF_TABLE_ROW_BG)
        return y_bottom - 0.035

    def _pdf_paragraph(self, fig: Any, y_top: float, title: str, text: str) -> float:
        left = self._PDF_MARGIN
        fig.text(
            left, y_top, title,
            fontsize=12.5, weight="bold", color=self._PDF_ACCENT,
        )
        lines = textwrap.wrap(text, width=100)
        fig.text(
            left, y_top - 0.026, "\n".join(lines),
            fontsize=9.5, color=self._PDF_INK, va="top", linespacing=1.4,
        )
        return y_top - 0.026 - 0.0205 * len(lines) - 0.035

    def _outputs(self) -> list[Path]:
        return [
            self.metrics_json,
            self.classification_report_txt,
            self.confusion_matrix_png,
            self.summary_csv,
            self.evaluation_report_pdf,
        ]

    def _reset_outputs(self) -> None:
        for path in self._outputs():
            if path.exists():
                path.unlink()
                LOGGER.info("Old artifact removed: %s", path)

    def _check_outputs(self) -> None:
        existing = [path for path in self._outputs() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Evaluation files already exist: {joined}. Use --overwrite.")

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _build_performance_interpretation(
    metrics: dict[str, Any],
    *,
    classes: tuple[str, ...],
) -> str:
    if metrics["evaluated_count"] == 0:
        return (
            "There are no valid predictions to evaluate. Check the model errors in "
            "results/predictions.jsonl before interpreting the performance."
        )

    macro_f1 = metrics["macro_f1"]
    accuracy = metrics["accuracy"]
    weighted_f1 = metrics["weighted_f1"]
    per_class = metrics["per_class"]

    if macro_f1 is None:
        level = "undefined"
    elif macro_f1 >= 0.85:
        level = "high"
    elif macro_f1 >= 0.70:
        level = "good"
    elif macro_f1 >= 0.50:
        level = "moderate"
    else:
        level = "low"

    best_class = max(classes, key=lambda label: per_class[label]["f1_score"])
    worst_class = min(classes, key=lambda label: per_class[label]["f1_score"])
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    confusion_text = _largest_confusion_text(matrix, classes=classes)

    parts = [
        (
            f"The model showed {level} performance, with accuracy "
            f"{_format_metric(accuracy)} and macro-F1 {_format_metric(macro_f1)} "
            f"over the {metrics['evaluated_count']} evaluated samples."
        ),
        (
            f"The best class by F1-score was {best_class} "
            f"({_format_metric(per_class[best_class]['f1_score'])}), while the weakest "
            f"class was {worst_class} "
            f"({_format_metric(per_class[worst_class]['f1_score'])})."
        ),
    ]

    if weighted_f1 is not None and macro_f1 is not None:
        delta = float(weighted_f1) - float(macro_f1)
        if abs(delta) >= 0.10:
            parts.append(
                "The gap between weighted-F1 and macro-F1 indicates that the average result "
                "is being influenced by the class distribution or by uneven performance "
                "across classes."
            )
        else:
            parts.append(
                "Macro-F1 and weighted-F1 are close, suggesting relatively balanced "
                "performance across the evaluated classes."
            )

    if confusion_text:
        parts.append(confusion_text)

    if metrics["skipped_count"]:
        parts.append(
            f"The interpretation must take into account that {metrics['skipped_count']} sample(s) "
            "were skipped due to model error or invalid label."
        )

    return " ".join(parts)


def _largest_confusion_text(matrix: np.ndarray, *, classes: tuple[str, ...]) -> str:
    if matrix.size == 0:
        return ""

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    max_value = int(off_diagonal.max())
    if max_value == 0:
        return "There were no confusions outside the main diagonal of the confusion matrix."

    true_index, pred_index = np.unravel_index(int(off_diagonal.argmax()), off_diagonal.shape)
    return (
        f"The most frequent confusion was {classes[true_index]} classified as "
        f"{classes[pred_index]}, with {max_value} occurrence(s)."
    )
