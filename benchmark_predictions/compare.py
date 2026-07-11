from __future__ import annotations

import csv
import json
import logging
import math
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
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from vlm_classifier.parser import TASK_FRB_BINARY, normalize_label


LOGGER = logging.getLogger(__name__)

COMPARISON_FILES = (
    "comparison.json",
    "comparison.csv",
    "comparison_report.txt",
    "comparison_report.pdf",
    "disagreement_samples.csv",
    "paired_scores.csv",
    "statistical_tests.json",
    "roc_curve.png",
    "precision_recall_curve.png",
    "calibration_reliability.png",
    "score_histogram.png",
    "threshold_curves.png",
)
KEY_METRICS = ("accuracy", "macro_f1", "weighted_f1")
BINARY_POSITIVE_LABEL = "FRB"
CALIBRATION_EPS = 1e-7


@dataclass(frozen=True)
class ModelComparisonConfig:
    vlm_metrics: Path
    external_metrics: Path
    vlm_predictions: Path
    external_predictions: Path
    output_dir: Path
    bootstrap_iterations: int = 1000
    bootstrap_seed: int = 42
    calibration_bins: int = 10

    def validate(self) -> None:
        if self.bootstrap_iterations < 0:
            raise ValueError("bootstrap_iterations must be >= 0.")
        if self.calibration_bins <= 0:
            raise ValueError("calibration_bins must be > 0.")


class ModelComparator:
    def __init__(self, *, config: ModelComparisonConfig, project_root: Path) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.vlm_metrics = self._resolve(config.vlm_metrics)
        self.external_metrics = self._resolve(config.external_metrics)
        self.vlm_predictions = self._resolve(config.vlm_predictions)
        self.external_predictions = self._resolve(config.external_predictions)
        self.output_dir = self._resolve(config.output_dir)

    def compare(self, *, overwrite: bool = False) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            self._reset_outputs()
        else:
            self._check_outputs()

        vlm_metrics = self._load_json(self.vlm_metrics)
        external_metrics = self._load_json(self.external_metrics)
        vlm_predictions = self._load_predictions(self.vlm_predictions)
        external_predictions = self._load_predictions(self.external_predictions)

        vlm_ids = {str(record["sample_id"]) for record in vlm_predictions}
        external_ids = {str(record["sample_id"]) for record in external_predictions}
        comparison_rows = self._compare_predictions(vlm_predictions, external_predictions)
        task = vlm_metrics.get("task") or external_metrics.get("task")
        binary_analysis = self._build_binary_score_analysis(
            vlm_predictions,
            external_predictions,
            task=task,
        )
        summary = self._build_summary(
            vlm_metrics,
            external_metrics,
            comparison_rows,
            vlm_only_sample_ids=sorted(vlm_ids - external_ids),
            external_only_sample_ids=sorted(external_ids - vlm_ids),
            binary_analysis=binary_analysis,
        )
        paths = self._output_paths()
        self._write_comparison_json(paths["comparison_json"], summary)
        self._write_comparison_csv(paths["comparison_csv"], summary)
        self._write_disagreements_csv(paths["disagreement_samples"], comparison_rows)
        if binary_analysis.get("binary_task"):
            self._write_paired_scores_csv(paths["paired_scores"], binary_analysis)
            self._write_statistical_tests_json(
                paths["statistical_tests"],
                binary_analysis,
            )
            self._write_binary_plots(paths, binary_analysis)
        self._write_report_txt(paths["comparison_report_txt"], summary)
        self._write_report_pdf(paths["comparison_report_pdf"], summary, paths)

        LOGGER.info("Model comparison written to %s", self.output_dir)
        return paths

    def _compare_predictions(
        self,
        vlm_predictions: list[dict[str, Any]],
        external_predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        vlm_by_id = {str(record["sample_id"]): record for record in vlm_predictions}
        external_by_id = {
            str(record["sample_id"]): record for record in external_predictions
        }
        sample_ids = sorted(set(vlm_by_id) & set(external_by_id))
        rows: list[dict[str, Any]] = []
        for sample_id in sample_ids:
            vlm = vlm_by_id[sample_id]
            external = external_by_id[sample_id]
            true_label = vlm.get("true_label") or external.get("true_label")
            if vlm.get("true_label") and external.get("true_label"):
                if vlm["true_label"] != external["true_label"]:
                    true_label = f"{vlm['true_label']}|{external['true_label']}"

            vlm_error = vlm.get("error")
            external_error = external.get("error")
            vlm_pred = vlm.get("predicted_label")
            external_pred = external.get("predicted_label")
            vlm_correct = bool(not vlm_error and vlm_pred == vlm.get("true_label"))
            external_correct = bool(
                not external_error and external_pred == external.get("true_label")
            )
            vlm_score = extract_frb_probability(vlm)
            external_score = extract_frb_probability(external)
            external_parsed = external.get("parsed_response")
            if not isinstance(external_parsed, dict):
                external_parsed = {}
            vlm_parsed = vlm.get("parsed_response")
            if not isinstance(vlm_parsed, dict):
                vlm_parsed = {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "source_true_label": (
                        vlm.get("source_true_label")
                        or external.get("source_true_label")
                        or true_label
                    ),
                    "vlm_predicted_label": vlm_pred,
                    "external_predicted_label": external_pred,
                    "vlm_correct": vlm_correct,
                    "external_correct": external_correct,
                    "vlm_confidence": vlm.get("confidence"),
                    "external_confidence": external.get("confidence"),
                    "vlm_frb_probability": vlm_score,
                    "external_frb_probability": external_score,
                    "external_n_candidates": external_parsed.get(
                        "n_candidates",
                        external_parsed.get("external_detection_count"),
                    ),
                    "external_aggregation": external_parsed.get("aggregation"),
                    "vlm_features": _json_dumps_compact(vlm_parsed.get("features")),
                    "vlm_error": vlm_error,
                    "external_error": external_error,
                    "agreement": vlm_pred == external_pred,
                }
            )
        return rows

    def _build_summary(
        self,
        vlm_metrics: dict[str, Any],
        external_metrics: dict[str, Any],
        comparison_rows: list[dict[str, Any]],
        vlm_only_sample_ids: list[str],
        external_only_sample_ids: list[str],
        binary_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        metric_comparison = {
            metric: {
                "vlm": vlm_metrics.get(metric),
                "external": external_metrics.get(metric),
                "delta_vlm_minus_external": _metric_delta(
                    vlm_metrics.get(metric),
                    external_metrics.get(metric),
                ),
            }
            for metric in KEY_METRICS
        }

        classes = sorted(
            set(vlm_metrics.get("classes", [])) | set(external_metrics.get("classes", []))
        )
        per_class: dict[str, dict[str, Any]] = {}
        for label in classes:
            per_class[label] = {}
            for metric_name in ("precision", "recall", "f1_score"):
                vlm_value = vlm_metrics.get("per_class", {}).get(label, {}).get(metric_name)
                external_value = (
                    external_metrics.get("per_class", {}).get(label, {}).get(metric_name)
                )
                per_class[label][metric_name] = {
                    "vlm": vlm_value,
                    "external": external_value,
                    "delta_vlm_minus_external": _metric_delta(vlm_value, external_value),
                }

        agreement_count = sum(1 for row in comparison_rows if row["agreement"])
        disagreement_rows = [row for row in comparison_rows if not row["agreement"]]
        vlm_only_correct = [
            row
            for row in comparison_rows
            if row["vlm_correct"] and not row["external_correct"]
        ]
        external_only_correct = [
            row
            for row in comparison_rows
            if row["external_correct"] and not row["vlm_correct"]
        ]
        both_correct = [
            row
            for row in comparison_rows
            if row["vlm_correct"] and row["external_correct"]
        ]
        both_wrong = [
            row
            for row in comparison_rows
            if not row["vlm_correct"] and not row["external_correct"]
        ]

        return {
            "task": vlm_metrics.get("task") or external_metrics.get("task"),
            "metric_comparison": metric_comparison,
            "per_class_comparison": per_class,
            "sample_comparison": {
                "common_count": len(comparison_rows),
                "agreement_count": agreement_count,
                "disagreement_count": len(disagreement_rows),
                "vlm_only_correct_count": len(vlm_only_correct),
                "external_only_correct_count": len(external_only_correct),
                "both_correct_count": len(both_correct),
                "both_wrong_count": len(both_wrong),
                "vlm_only_sample_count": len(vlm_only_sample_ids),
                "external_only_sample_count": len(external_only_sample_ids),
            },
            "vlm_only_sample_ids": vlm_only_sample_ids,
            "external_only_sample_ids": external_only_sample_ids,
            "disagreement_samples": disagreement_rows,
            "sample_rows": comparison_rows,
            "binary_score_analysis": binary_analysis,
        }

    def _build_binary_score_analysis(
        self,
        vlm_predictions: list[dict[str, Any]],
        external_predictions: list[dict[str, Any]],
        *,
        task: Any,
    ) -> dict[str, Any]:
        if task != TASK_FRB_BINARY:
            return {
                "available": False,
                "binary_task": False,
                "reason": "probabilistic_analysis_only_for_frb_binary",
                "paired_rows": [],
                "exclusions": [],
            }

        paired_rows, exclusions = build_paired_score_rows(
            vlm_predictions,
            external_predictions,
        )
        if not paired_rows:
            return {
                "available": False,
                "binary_task": True,
                "reason": "no_paired_binary_scores",
                "paired_rows": [],
                "exclusions": exclusions,
            }

        y_true = np.asarray([row["y_true"] for row in paired_rows], dtype=int)
        score_vlm = np.asarray([row["score_vlm"] for row in paired_rows], dtype=float)
        score_external = np.asarray(
            [row["score_external"] for row in paired_rows],
            dtype=float,
        )
        source_true_labels = [row["source_true_label"] for row in paired_rows]
        bootstrap_iterations = self.config.bootstrap_iterations
        bootstrap_seed = self.config.bootstrap_seed
        calibration_bins = self.config.calibration_bins

        vlm_metrics = score_metric_summary(
            y_true,
            score_vlm,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            calibration_bins=calibration_bins,
        )
        external_metrics = score_metric_summary(
            y_true,
            score_external,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            calibration_bins=calibration_bins,
        )
        deltas = {
            "roc_auc": bootstrap_metric_delta_ci(
                y_true,
                score_vlm,
                score_external,
                metric="roc_auc",
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
            "average_precision": bootstrap_metric_delta_ci(
                y_true,
                score_vlm,
                score_external,
                metric="average_precision",
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
        }

        threshold_analysis = {
            "vlm": operating_point_summary(y_true, score_vlm),
            "external": operating_point_summary(y_true, score_external),
        }
        vlm_correct = score_vlm >= 0.5
        external_correct = score_external >= 0.5
        vlm_correct = vlm_correct == y_true.astype(bool)
        external_correct = external_correct == y_true.astype(bool)

        return {
            "available": True,
            "binary_task": True,
            "score_source": "parsed_response.frb_probability",
            "clipping": {
                "eps": CALIBRATION_EPS,
                "applied_to": ["log_loss", "brier_score", "ece"],
            },
            "bootstrap": {
                "iterations": bootstrap_iterations,
                "seed": bootstrap_seed,
            },
            "calibration_bins": calibration_bins,
            "paired_count": len(paired_rows),
            "class_balance": {
                "positive_count": int(y_true.sum()),
                "negative_count": int(len(y_true) - y_true.sum()),
                "positive_fraction": float(y_true.mean()) if len(y_true) else None,
            },
            "metrics": {
                "vlm": vlm_metrics,
                "external": external_metrics,
                "delta_vlm_minus_external": deltas,
            },
            "operating_points": threshold_analysis,
            "mcnemar_at_threshold_0_5": mcnemar_summary(
                vlm_correct,
                external_correct,
            ),
            "fpr_by_source_true_label_at_threshold_0_5": fpr_by_source_true_label(
                y_true,
                source_true_labels,
                score_vlm,
                score_external,
                threshold=0.5,
            ),
            "exclusions": exclusions,
            "paired_rows": paired_rows,
        }

    def _write_comparison_json(self, path: Path, summary: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_comparison_csv(self, path: Path, summary: dict[str, Any]) -> None:
        fieldnames = ["metric", "class", "vlm", "external", "delta_vlm_minus_external"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for metric, values in summary["metric_comparison"].items():
                writer.writerow({"metric": metric, "class": "", **values})
            for label, metrics in summary["per_class_comparison"].items():
                for metric, values in metrics.items():
                    writer.writerow({"metric": metric, "class": label, **values})
            binary = summary.get("binary_score_analysis", {})
            if binary.get("available"):
                metric_summary = binary["metrics"]
                for metric_name in (
                    "roc_auc",
                    "average_precision",
                    "brier_score",
                    "log_loss",
                    "ece",
                ):
                    vlm_value = metric_summary["vlm"].get(metric_name, {}).get("estimate")
                    external_value = (
                        metric_summary["external"].get(metric_name, {}).get("estimate")
                    )
                    writer.writerow(
                        {
                            "metric": f"binary_{metric_name}",
                            "class": "",
                            "vlm": vlm_value,
                            "external": external_value,
                            "delta_vlm_minus_external": _metric_delta(
                                vlm_value,
                                external_value,
                            ),
                        }
                    )

    def _write_disagreements_csv(
        self,
        path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        fieldnames = [
            "sample_id",
            "true_label",
            "source_true_label",
            "vlm_predicted_label",
            "external_predicted_label",
            "vlm_correct",
            "external_correct",
            "vlm_confidence",
            "external_confidence",
            "vlm_frb_probability",
            "external_frb_probability",
            "external_n_candidates",
            "external_aggregation",
            "vlm_features",
            "vlm_error",
            "external_error",
            "agreement",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                if not row["agreement"]:
                    writer.writerow(row)

    def _write_paired_scores_csv(
        self,
        path: Path,
        binary_analysis: dict[str, Any],
    ) -> None:
        fieldnames = [
            "sample_id",
            "true_label",
            "source_true_label",
            "y_true",
            "score_vlm",
            "score_external",
            "vlm_predicted_label",
            "external_predicted_label",
            "external_n_candidates",
            "external_aggregation",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            if not binary_analysis.get("available"):
                return
            writer.writerows(binary_analysis.get("paired_rows", []))

    def _write_statistical_tests_json(
        self,
        path: Path,
        binary_analysis: dict[str, Any],
    ) -> None:
        payload: dict[str, Any]
        if binary_analysis.get("available"):
            payload = {
                "bootstrap": binary_analysis["bootstrap"],
                "metrics": binary_analysis["metrics"],
                "mcnemar_at_threshold_0_5": binary_analysis[
                    "mcnemar_at_threshold_0_5"
                ],
                "exclusion_count": len(binary_analysis.get("exclusions", [])),
            }
        else:
            payload = {
                "available": False,
                "reason": binary_analysis.get("reason"),
                "exclusion_count": len(binary_analysis.get("exclusions", [])),
            }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_report_txt(self, path: Path, summary: dict[str, Any]) -> None:
        lines = ["Model comparison report", ""]
        for metric, values in summary["metric_comparison"].items():
            lines.append(
                f"{metric}: VLM={_fmt(values['vlm'])} | "
                f"external={_fmt(values['external'])} | "
                f"delta={_fmt(values['delta_vlm_minus_external'])}"
            )
        lines.extend(["", "Sample-level comparison:"])
        for key, value in summary["sample_comparison"].items():
            lines.append(f"- {key}: {value}")
        binary = summary.get("binary_score_analysis", {})
        if binary.get("available"):
            lines.extend(["", "Binary probabilistic analysis:"])
            lines.append(f"- paired_score_count: {binary['paired_count']}")
            balance = binary["class_balance"]
            lines.append(
                "- class_balance: "
                f"positive={balance['positive_count']} | "
                f"negative={balance['negative_count']} | "
                f"positive_fraction={_fmt(balance['positive_fraction'])}"
            )
            for metric_name in ("average_precision", "roc_auc", "brier_score", "ece"):
                vlm_metric = binary["metrics"]["vlm"].get(metric_name, {})
                external_metric = binary["metrics"]["external"].get(metric_name, {})
                lines.append(
                    f"- {metric_name}: VLM={_fmt(vlm_metric.get('estimate'))} | "
                    f"external={_fmt(external_metric.get('estimate'))}"
                )
            delta_ap = binary["metrics"]["delta_vlm_minus_external"][
                "average_precision"
            ]
            lines.append(
                "- delta_average_precision_vlm_minus_external: "
                f"{_fmt(delta_ap.get('estimate'))} "
                f"CI95=[{_fmt(delta_ap.get('ci_low'))}, {_fmt(delta_ap.get('ci_high'))}]"
            )
            lines.append(
                "- score_source: parsed_response.frb_probability; "
                "the root-level confidence was not used for ROC/PR."
            )
        elif binary.get("binary_task"):
            lines.extend(["", "Binary probabilistic analysis:"])
            lines.append(f"- unavailable: {binary.get('reason')}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _PDF_PAGE_SIZE = (8.27, 11.69)
    _PDF_MARGIN = 0.08
    _PDF_INK = "#1A1A1A"
    _PDF_MUTED = "#555555"
    _PDF_ACCENT = "#1F4E79"
    _PDF_TABLE_HEADER_BG = "#EAF2F8"
    _PDF_TABLE_ROW_BG = "#F5F7FA"
    _PDF_TABLE_EDGE = "#D7DEE6"

    def _write_report_pdf(
        self,
        path: Path,
        summary: dict[str, Any],
        paths: dict[str, Path],
    ) -> None:
        with PdfPages(path) as pdf:
            self._add_summary_pdf_page(pdf, summary)
            binary = summary.get("binary_score_analysis", {})
            if binary.get("available"):
                plot_pages = [
                    [
                        ("ROC Curve", paths["roc_curve"]),
                        ("Precision-Recall Curve", paths["precision_recall_curve"]),
                    ],
                    [
                        ("Score Histogram", paths["score_histogram"]),
                        ("Operating Point", paths["threshold_curves"]),
                    ],
                    [("Calibration", paths["calibration_reliability"])],
                ]
                for panels in plot_pages:
                    available = [
                        (title, plot_path)
                        for title, plot_path in panels
                        if plot_path.exists()
                    ]
                    if available:
                        self._add_plot_pdf_page(pdf, available)

    def _add_summary_pdf_page(self, pdf: PdfPages, summary: dict[str, Any]) -> None:
        fig = plt.figure(figsize=self._PDF_PAGE_SIZE)
        fig.patch.set_facecolor("white")
        left = self._PDF_MARGIN
        width = 1.0 - 2 * self._PDF_MARGIN

        fig.text(
            left, 0.945, "Model Comparison",
            fontsize=20, weight="bold", color=self._PDF_INK,
        )
        fig.text(
            left, 0.922, "VLM vs external model on the binary task",
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
        metric_rows = [
            [
                metric,
                _fmt(values["vlm"]),
                _fmt(values["external"]),
                _fmt(values["delta_vlm_minus_external"]),
            ]
            for metric, values in summary["metric_comparison"].items()
        ]
        y = self._pdf_section(
            fig, y, "Discrete metrics at the 0.5 threshold",
            ["Metric", "VLM", "External", "Delta"],
            metric_rows, col_widths=(0.34, 0.22, 0.22, 0.22),
        )

        sample = summary["sample_comparison"]
        sample_rows = [
            [label, str(sample[key])]
            for key, label in _SAMPLE_SUMMARY_LABELS
            if key in sample
        ]
        y = self._pdf_section(
            fig, y, "Per-sample summary",
            ["Indicator", "Value"],
            sample_rows, col_widths=(0.66, 0.34),
        )

        binary = summary.get("binary_score_analysis", {})
        if binary.get("available"):
            balance = binary["class_balance"]
            metrics = binary["metrics"]
            binary_rows = [
                ["Paired samples", str(binary["paired_count"])],
                ["FRB", str(balance["positive_count"])],
                ["NON_FRB", str(balance["negative_count"])],
                ["AP VLM", _metric_with_ci(metrics["vlm"]["average_precision"])],
                ["AP external", _metric_with_ci(metrics["external"]["average_precision"])],
                [
                    "Delta AP",
                    _metric_with_ci(
                        metrics["delta_vlm_minus_external"]["average_precision"]
                    ),
                ],
                ["AUC VLM", _metric_with_ci(metrics["vlm"]["roc_auc"])],
                ["AUC external", _metric_with_ci(metrics["external"]["roc_auc"])],
            ]
            y = self._pdf_section(
                fig, y, "Binary probabilistic analysis",
                ["Indicator", "Value"],
                binary_rows, col_widths=(0.40, 0.60),
            )
            note = (
                "The ROC/PR curves use parsed_response.frb_probability. "
                "The root-level confidence is only the confidence in the discrete label. "
                "Log-loss, Brier, and ECE use clipping to [1e-7, 1-1e-7]."
            )
            fig.text(
                left, max(y, 0.05),
                "\n".join(textwrap.wrap(note, width=110)),
                fontsize=8.5, color=self._PDF_MUTED, va="top", linespacing=1.35,
            )
        elif binary.get("binary_task"):
            fig.text(
                left, y,
                f"Binary probabilistic analysis unavailable: {binary.get('reason')}",
                fontsize=9.5, color=self._PDF_MUTED, va="top",
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

    def _add_plot_pdf_page(
        self,
        pdf: PdfPages,
        panels: list[tuple[str, Path]],
    ) -> None:
        fig = plt.figure(figsize=self._PDF_PAGE_SIZE)
        fig.patch.set_facecolor("white")
        left = self._PDF_MARGIN
        width = 1.0 - 2 * self._PDF_MARGIN
        page_width, page_height = self._PDF_PAGE_SIZE
        y = 0.955
        slot_height = (0.88 / len(panels)) - 0.03
        for title, plot_path in panels:
            fig.text(
                left, y, title,
                fontsize=12.5, weight="bold", color=self._PDF_ACCENT,
            )
            fig.add_artist(
                plt.Line2D(
                    [left, left + width], [y - 0.008, y - 0.008],
                    transform=fig.transFigure,
                    color=self._PDF_TABLE_EDGE, linewidth=1.0,
                )
            )
            image = plt.imread(plot_path)
            aspect = image.shape[1] / image.shape[0]
            image_width = width
            image_height = image_width * page_width / (aspect * page_height)
            if image_height > slot_height:
                image_height = slot_height
                image_width = image_height * aspect * page_height / page_width
            x = left + (width - image_width) / 2
            ax = fig.add_axes([x, y - 0.02 - image_height, image_width, image_height])
            ax.imshow(image)
            ax.set_axis_off()
            y = y - 0.02 - image_height - 0.045
        pdf.savefig(fig)
        plt.close(fig)

    def _write_binary_plots(
        self,
        paths: dict[str, Path],
        binary_analysis: dict[str, Any],
    ) -> None:
        if not binary_analysis.get("available"):
            return
        rows = binary_analysis["paired_rows"]
        y_true = np.asarray([row["y_true"] for row in rows], dtype=int)
        score_vlm = np.asarray([row["score_vlm"] for row in rows], dtype=float)
        score_external = np.asarray([row["score_external"] for row in rows], dtype=float)

        self._write_roc_plot(paths["roc_curve"], y_true, score_vlm, score_external)
        self._write_pr_plot(
            paths["precision_recall_curve"],
            y_true,
            score_vlm,
            score_external,
        )
        self._write_calibration_plot(
            paths["calibration_reliability"],
            binary_analysis,
        )
        self._write_score_histogram(
            paths["score_histogram"],
            y_true,
            score_vlm,
            score_external,
        )
        self._write_threshold_plot(
            paths["threshold_curves"],
            y_true,
            score_vlm,
            score_external,
        )

    def _write_roc_plot(
        self,
        path: Path,
        y_true: np.ndarray,
        score_vlm: np.ndarray,
        score_external: np.ndarray,
    ) -> None:
        if len(np.unique(y_true)) < 2:
            return
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        for label, scores in (("VLM", score_vlm), ("External", score_external)):
            fpr, tpr, _thresholds = roc_curve(y_true, scores)
            auc = _safe_score_metric(y_true, scores, metric="roc_auc")
            ax.plot(fpr, tpr, label=f"{label} AUC={_fmt(auc)}")
        ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(_plot_title("ROC", y_true))
        ax.legend(loc="lower right")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_pr_plot(
        self,
        path: Path,
        y_true: np.ndarray,
        score_vlm: np.ndarray,
        score_external: np.ndarray,
    ) -> None:
        if len(np.unique(y_true)) < 2:
            return
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        prevalence = float(np.mean(y_true))
        for label, scores in (("VLM", score_vlm), ("External", score_external)):
            precision, recall, _thresholds = precision_recall_curve(y_true, scores)
            ap = _safe_score_metric(y_true, scores, metric="average_precision")
            ax.plot(recall, precision, label=f"{label} AP={_fmt(ap)}")
        ax.axhline(prevalence, color="0.5", linestyle="--", linewidth=1, label="Prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(_plot_title("Precision-Recall", y_true))
        ax.legend(loc="lower left")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_calibration_plot(
        self,
        path: Path,
        binary_analysis: dict[str, Any],
    ) -> None:
        fig, (ax_rel, ax_hist) = plt.subplots(
            2,
            1,
            figsize=(6.5, 7.0),
            dpi=150,
            gridspec_kw={"height_ratios": [3, 1.2]},
        )
        ax_rel.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
        for label, color in (("vlm", "tab:blue"), ("external", "tab:orange")):
            metric = binary_analysis["metrics"][label]["ece"]
            bins = metric["bins"]
            centers = [
                (bin_row["lower"] + bin_row["upper"]) / 2.0
                for bin_row in bins
                if bin_row["count"] > 0
            ]
            accuracies = [
                bin_row["accuracy"]
                for bin_row in bins
                if bin_row["count"] > 0
            ]
            ax_rel.plot(
                centers,
                accuracies,
                marker="o",
                label=f"{label.upper()} ECE={_fmt(metric['estimate'])}",
                color=color,
            )
            counts = [bin_row["count"] for bin_row in bins]
            edges = [bins[0]["lower"], *[bin_row["upper"] for bin_row in bins]]
            ax_hist.stairs(counts, edges, label=label.upper(), color=color, linewidth=1.4)
        ax_rel.set_xlabel("Mean predicted probability")
        ax_rel.set_ylabel("Empirical FRB fraction")
        ax_rel.set_title("Reliability diagram")
        ax_rel.legend(loc="upper left")
        ax_rel.grid(alpha=0.25)
        ax_hist.set_xlabel("Predicted FRB probability")
        ax_hist.set_ylabel("Count")
        ax_hist.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_score_histogram(
        self,
        path: Path,
        y_true: np.ndarray,
        score_vlm: np.ndarray,
        score_external: np.ndarray,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=150)
        bins = np.linspace(0.0, 1.0, 21)
        ax.hist(score_vlm, bins=bins, alpha=0.55, label="VLM", color="tab:blue")
        ax.hist(
            score_external,
            bins=bins,
            alpha=0.55,
            label="External",
            color="tab:orange",
        )
        ax.set_xlabel("Predicted FRB probability")
        ax.set_ylabel("Samples")
        ax.set_title(_plot_title("Score histogram", y_true))
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _write_threshold_plot(
        self,
        path: Path,
        y_true: np.ndarray,
        score_vlm: np.ndarray,
        score_external: np.ndarray,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharey=True)
        for ax, label, scores in (
            (axes[0], "VLM", score_vlm),
            (axes[1], "External", score_external),
        ):
            rows = threshold_curve(y_true, scores)
            thresholds = [row["threshold"] for row in rows]
            for metric in ("precision", "recall", "f1"):
                ax.plot(
                    thresholds,
                    [row[metric] for row in rows],
                    label=metric,
                )
            ax.set_title(label)
            ax.set_xlabel("Threshold")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Metric")
        axes[1].legend(loc="lower left")
        fig.suptitle("Operating point curves")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError(f"JSON file must contain an object: {path}")
        return parsed

    def _load_predictions(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Predictions not found: {path}")
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid line in {path}:{line_number}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object.")
                records.append(parsed)
        return records

    def _output_paths(self) -> dict[str, Path]:
        return {
            "comparison_json": self.output_dir / "comparison.json",
            "comparison_csv": self.output_dir / "comparison.csv",
            "comparison_report_txt": self.output_dir / "comparison_report.txt",
            "comparison_report_pdf": self.output_dir / "comparison_report.pdf",
            "disagreement_samples": self.output_dir / "disagreement_samples.csv",
            "paired_scores": self.output_dir / "paired_scores.csv",
            "statistical_tests": self.output_dir / "statistical_tests.json",
            "roc_curve": self.output_dir / "roc_curve.png",
            "precision_recall_curve": self.output_dir / "precision_recall_curve.png",
            "calibration_reliability": self.output_dir / "calibration_reliability.png",
            "score_histogram": self.output_dir / "score_histogram.png",
            "threshold_curves": self.output_dir / "threshold_curves.png",
        }

    def _reset_outputs(self) -> None:
        for path in self._output_paths().values():
            if path.exists():
                path.unlink()
                LOGGER.info("Old artifact removed: %s", path)

    def _check_outputs(self) -> None:
        existing = [path for path in self._output_paths().values() if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"Comparison files already exist: {joined}. Use --overwrite."
            )

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path


def build_paired_score_rows(
    vlm_predictions: list[dict[str, Any]],
    external_predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vlm_by_id = {str(record.get("sample_id")): record for record in vlm_predictions}
    external_by_id = {
        str(record.get("sample_id")): record for record in external_predictions
    }
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for sample_id in sorted(set(vlm_by_id) & set(external_by_id)):
        vlm = vlm_by_id[sample_id]
        external = external_by_id[sample_id]
        reason = _score_row_exclusion_reason(vlm, external)
        if reason is not None:
            exclusions.append({"sample_id": sample_id, "reason": reason})
            continue

        try:
            vlm_true = normalize_label(vlm.get("true_label"), task=TASK_FRB_BINARY)
            external_true = normalize_label(
                external.get("true_label"),
                task=TASK_FRB_BINARY,
            )
        except ValueError as exc:
            exclusions.append({"sample_id": sample_id, "reason": str(exc)})
            continue
        if vlm_true != external_true:
            exclusions.append(
                {
                    "sample_id": sample_id,
                    "reason": f"true_label_mismatch:{vlm_true}|{external_true}",
                }
            )
            continue

        score_vlm = extract_frb_probability(vlm)
        score_external = extract_frb_probability(external)
        if score_vlm is None or score_external is None:
            exclusions.append(
                {"sample_id": sample_id, "reason": "missing_or_invalid_frb_probability"}
            )
            continue

        external_parsed = external.get("parsed_response")
        if not isinstance(external_parsed, dict):
            external_parsed = {}

        rows.append(
            {
                "sample_id": sample_id,
                "true_label": vlm_true,
                "source_true_label": (
                    vlm.get("source_true_label")
                    or external.get("source_true_label")
                    or vlm_true
                ),
                "y_true": 1 if vlm_true == BINARY_POSITIVE_LABEL else 0,
                "score_vlm": score_vlm,
                "score_external": score_external,
                "vlm_predicted_label": vlm.get("predicted_label"),
                "external_predicted_label": external.get("predicted_label"),
                "external_n_candidates": external_parsed.get(
                    "n_candidates",
                    external_parsed.get("external_detection_count"),
                ),
                "external_aggregation": external_parsed.get("aggregation"),
            }
        )
    return rows, exclusions


def _score_row_exclusion_reason(
    vlm: dict[str, Any],
    external: dict[str, Any],
) -> str | None:
    if vlm.get("error"):
        return f"vlm_error:{vlm.get('error')}"
    if external.get("error"):
        return f"external_error:{external.get('error')}"
    if not isinstance(vlm.get("parsed_response"), dict):
        return "vlm_missing_parsed_response"
    if not isinstance(external.get("parsed_response"), dict):
        return "external_missing_parsed_response"
    return None


def extract_frb_probability(record: dict[str, Any]) -> float | None:
    parsed_response = record.get("parsed_response")
    if not isinstance(parsed_response, dict):
        return None
    value = parsed_response.get("frb_probability")
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability):
        return None
    return min(1.0, max(0.0, probability))


def score_metric_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    calibration_bins: int,
) -> dict[str, Any]:
    return {
        "roc_auc": bootstrap_metric_ci(
            y_true,
            scores,
            metric="roc_auc",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "average_precision": bootstrap_metric_ci(
            y_true,
            scores,
            metric="average_precision",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "brier_score": {
            "estimate": brier_score(y_true, scores),
            "ci_low": None,
            "ci_high": None,
        },
        "log_loss": {
            "estimate": binary_log_loss(y_true, scores),
            "ci_low": None,
            "ci_high": None,
        },
        "ece": expected_calibration_error(
            y_true,
            scores,
            n_bins=calibration_bins,
        ),
        "probability_summary": probability_summary(scores),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    metric: str,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    estimate = _safe_score_metric(y_arr, score_arr, metric=metric)
    values: list[float] = []
    if iterations > 0 and len(y_arr) > 1:
        rng = np.random.default_rng(seed)
        for _ in range(iterations):
            indices = rng.integers(0, len(y_arr), size=len(y_arr))
            value = _safe_score_metric(
                y_arr[indices],
                score_arr[indices],
                metric=metric,
            )
            if value is not None:
                values.append(value)
    ci_low = float(np.percentile(values, 2.5)) if values else None
    ci_high = float(np.percentile(values, 97.5)) if values else None
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "valid_bootstrap_iterations": len(values),
    }


def bootstrap_metric_delta_ci(
    y_true: np.ndarray | list[int],
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    *,
    metric: str,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    estimate_a = _safe_score_metric(y_arr, a_arr, metric=metric)
    estimate_b = _safe_score_metric(y_arr, b_arr, metric=metric)
    estimate = (
        None
        if estimate_a is None or estimate_b is None
        else float(estimate_a - estimate_b)
    )
    values: list[float] = []
    if iterations > 0 and len(y_arr) > 1:
        rng = np.random.default_rng(seed)
        for _ in range(iterations):
            indices = rng.integers(0, len(y_arr), size=len(y_arr))
            value_a = _safe_score_metric(y_arr[indices], a_arr[indices], metric=metric)
            value_b = _safe_score_metric(y_arr[indices], b_arr[indices], metric=metric)
            if value_a is not None and value_b is not None:
                values.append(float(value_a - value_b))
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(values, 2.5)) if values else None,
        "ci_high": float(np.percentile(values, 97.5)) if values else None,
        "valid_bootstrap_iterations": len(values),
    }


def _safe_score_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str,
) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, scores))
    if metric == "average_precision":
        return float(average_precision_score(y_true, scores))
    raise ValueError(f"Unknown binary metric: {metric}")


def brier_score(y_true: np.ndarray | list[int], scores: np.ndarray | list[float]) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    clipped = clip_probabilities(scores)
    return float(np.mean((clipped - y_arr) ** 2))


def binary_log_loss(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    clipped = clip_probabilities(scores)
    losses = -(y_arr * np.log(clipped) + (1.0 - y_arr) * np.log(1.0 - clipped))
    return float(np.mean(losses))


def expected_calibration_error(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0.")
    y_arr = np.asarray(y_true, dtype=float)
    clipped = clip_probabilities(scores)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_arr)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (clipped >= lower) & (clipped <= upper)
        else:
            mask = (clipped >= lower) & (clipped < upper)
        count = int(mask.sum())
        if count:
            confidence = float(clipped[mask].mean())
            accuracy = float(y_arr[mask].mean())
            gap = abs(accuracy - confidence)
            ece += (count / total) * gap
        else:
            confidence = None
            accuracy = None
            gap = None
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
    return {"estimate": float(ece), "bins": bins}


def clip_probabilities(scores: np.ndarray | list[float]) -> np.ndarray:
    return np.clip(np.asarray(scores, dtype=float), CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)


def probability_summary(scores: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(scores, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "exact_zero_count": int(np.sum(arr == 0.0)),
        "exact_one_count": int(np.sum(arr == 1.0)),
    }


def threshold_curve(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    *,
    steps: int = 101,
) -> list[dict[str, Any]]:
    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, steps):
        rows.append(_threshold_metrics(y_arr, score_arr, float(threshold)))
    return rows


def operating_point_summary(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    curve = threshold_curve(y_arr, score_arr)
    f1_optimal = max(curve, key=lambda row: (row["f1"], row["threshold"]))
    recall_candidates = [row for row in curve if row["recall"] >= 0.9]
    recall_0_9 = (
        max(recall_candidates, key=lambda row: (row["precision"], row["threshold"]))
        if recall_candidates
        else None
    )
    return {
        "threshold_0_5": _threshold_metrics(y_arr, score_arr, 0.5),
        "f1_optimal": f1_optimal,
        "recall_0_9": recall_0_9,
    }


def _threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def mcnemar_summary(
    vlm_correct: np.ndarray | list[bool],
    external_correct: np.ndarray | list[bool],
) -> dict[str, Any]:
    vlm_arr = np.asarray(vlm_correct, dtype=bool)
    external_arr = np.asarray(external_correct, dtype=bool)
    vlm_only = int(np.sum(vlm_arr & ~external_arr))
    external_only = int(np.sum(~vlm_arr & external_arr))
    discordant = vlm_only + external_only
    if discordant == 0:
        statistic = None
        p_value = 1.0
    else:
        statistic = ((abs(vlm_only - external_only) - 1) ** 2) / discordant
        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "vlm_correct_external_wrong": vlm_only,
        "external_correct_vlm_wrong": external_only,
        "discordant_count": discordant,
        "statistic_continuity_corrected": statistic,
        "p_value_chi_square_approx": p_value,
    }


def fpr_by_source_true_label(
    y_true: np.ndarray | list[int],
    source_true_labels: list[Any],
    score_vlm: np.ndarray | list[float],
    score_external: np.ndarray | list[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    y_arr = np.asarray(y_true, dtype=int)
    vlm_pred = np.asarray(score_vlm, dtype=float) >= threshold
    external_pred = np.asarray(score_external, dtype=float) >= threshold
    groups = sorted(
        {
            str(label or "UNKNOWN").upper()
            for label, y_value in zip(source_true_labels, y_arr)
            if y_value == 0
        }
    )
    result: dict[str, Any] = {}
    for group in groups:
        mask = np.asarray(
            [
                y_value == 0 and str(label or "UNKNOWN").upper() == group
                for label, y_value in zip(source_true_labels, y_arr)
            ],
            dtype=bool,
        )
        n = int(mask.sum())
        vlm_fp = int(np.sum(vlm_pred[mask]))
        external_fp = int(np.sum(external_pred[mask]))
        result[group] = {
            "n": n,
            "vlm_false_positives": vlm_fp,
            "external_false_positives": external_fp,
            "vlm_fpr": float(vlm_fp / n) if n else None,
            "external_fpr": float(external_fp / n) if n else None,
        }
    return result


def _metric_delta(vlm_value: Any, external_value: Any) -> float | None:
    if vlm_value is None or external_value is None:
        return None
    return float(vlm_value) - float(external_value)


_SAMPLE_SUMMARY_LABELS = (
    ("common_count", "Common samples"),
    ("agreement_count", "Agreements"),
    ("disagreement_count", "Disagreements"),
    ("vlm_only_correct_count", "Only the VLM correct"),
    ("external_only_correct_count", "Only the external model correct"),
    ("both_correct_count", "Both correct"),
    ("both_wrong_count", "Both wrong"),
    ("vlm_only_sample_count", "Samples only in the VLM file"),
    ("external_only_sample_count", "Samples only in the external file"),
)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _metric_with_ci(metric: dict[str, Any]) -> str:
    estimate = _fmt(metric.get("estimate"))
    if metric.get("ci_low") is None or metric.get("ci_high") is None:
        return estimate
    return f"{estimate} [{_fmt(metric.get('ci_low'))}, {_fmt(metric.get('ci_high'))}]"


def _plot_title(prefix: str, y_true: np.ndarray) -> str:
    positives = int(np.sum(y_true))
    negatives = int(len(y_true) - positives)
    return f"{prefix} (N={len(y_true)}, FRB={positives}, NON_FRB={negatives})"


def _json_dumps_compact(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
