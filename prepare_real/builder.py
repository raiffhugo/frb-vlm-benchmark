from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from plot_dataset.plotter import (
    ANONYMIZED_IMAGE_DIR,
    ANONYMIZED_IMAGE_PREFIX,
    DEFAULT_IMAGE_DPI,
    DynamicSpectrum,
    render_spectrum_png,
)
from prepare_real.preprocess import (
    TOA_REFERENCES,
    averaged_frequency_axis,
    block_average,
    burst_center_s,
    channel_keep_mask,
    flatten_rows,
    normalize_per_channel,
    plan_window,
    robust_channel_stats,
)
from prepare_real import reader as astropy_reader


LOGGER = logging.getLogger(__name__)

VALID_LABELS = ("FRB", "RFI", "NOISE")
NEGATIVE_WINDOW_MODES = ("random", "center", "start")
READER_BACKENDS = ("astropy", "your")


@dataclass(frozen=True)
class RealPrepConfig:
    input_manifest: Path
    output_dir: Path
    window_seconds: float = 2.0
    time_bins: int = 1024
    freq_bins: int = 512
    zap_sigma: float = 5.0
    edge_channels: int = 0
    toa_reference: str = "top"
    negative_window: str = "random"
    seed: int = 42
    cmap: str = "viridis"
    normalization: str = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    width: int = 1024
    height: int = 768
    dpi: int = DEFAULT_IMAGE_DPI
    include_title: bool = False
    reader: str = "astropy"

    def validate(self) -> None:
        if self.reader not in READER_BACKENDS:
            raise ValueError(f"invalid reader: {self.reader!r}")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if self.time_bins <= 0 or self.freq_bins <= 0:
            raise ValueError("time_bins and freq_bins must be positive.")
        if self.zap_sigma < 0:
            raise ValueError("zap_sigma must be >= 0 (0 disables the zap).")
        if self.edge_channels < 0:
            raise ValueError("edge_channels must be >= 0.")
        if self.toa_reference not in TOA_REFERENCES:
            raise ValueError(f"invalid toa_reference: {self.toa_reference!r}")
        if self.negative_window not in NEGATIVE_WINDOW_MODES:
            raise ValueError(f"invalid negative_window: {self.negative_window!r}")
        if self.normalization not in {"linear", "minmax", "zscore", "percentile"}:
            raise ValueError(f"Invalid normalization: {self.normalization}")
        if not 0 <= self.percentile_low < self.percentile_high <= 100:
            raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")
        if self.width <= 0 or self.height <= 0 or self.dpi <= 0:
            raise ValueError("width, height and dpi must be positive.")


class RealDatasetPreparer:
    def __init__(self, *, config: RealPrepConfig, project_root: Path) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.input_manifest = self._resolve(config.input_manifest)
        self.output_dir = self._resolve(config.output_dir)
        self.images_dir = self.output_dir / "images"
        self.manifest_path = self.output_dir / "metadata" / "image_manifest.jsonl"
        if config.reader == "your":
            from prepare_real import reader_your

            self._reader = reader_your
        else:
            self._reader = astropy_reader

    def prepare(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        entries = self._load_input_manifest()
        if overwrite:
            self._reset_outputs()
        if self.manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Manifest already exists: {self.manifest_path}. Use --overwrite."
            )
        (self.images_dir / ANONYMIZED_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for index, entry in enumerate(
            tqdm(entries, desc="Preparing real spectra", unit="image")
        ):
            records.append(self._prepare_one(entry, image_index=index, overwrite=overwrite))

        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        LOGGER.info("Image manifest written to %s", self.manifest_path)
        return records

    def _load_input_manifest(self) -> list[dict[str, Any]]:
        if not self.input_manifest.exists():
            raise FileNotFoundError(
                f"Input manifest not found: {self.input_manifest}"
            )

        suffix = self.input_manifest.suffix.lower()
        if suffix == ".csv":
            with self.input_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        elif suffix in {".jsonl", ".json"}:
            rows = []
            with self.input_manifest.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid line in {self.input_manifest}:{line_number}"
                        ) from exc
        else:
            raise ValueError(
                f"Unsupported manifest format: {self.input_manifest.name}. "
                "Use .csv or .jsonl."
            )

        if not rows:
            raise ValueError(f"No sample found in {self.input_manifest}")
        return [self._validate_entry(row, position) for position, row in enumerate(rows, start=1)]

    def _validate_entry(self, row: dict[str, Any], position: int) -> dict[str, Any]:
        fits_path = str(row.get("fits_path") or "").strip()
        if not fits_path:
            raise ValueError(f"Sample {position} has no fits_path in the input manifest.")

        label = str(row.get("label") or "").strip().upper()
        if label not in VALID_LABELS:
            raise ValueError(
                f"Sample {position} has an invalid label: {row.get('label')!r}. "
                f"Use one of {', '.join(VALID_LABELS)}."
            )

        toa = _optional_float(row.get("toa"), field="toa", position=position)
        dm = _optional_float(row.get("dm"), field="dm", position=position)
        if label == "FRB" and (toa is None or dm is None):
            raise ValueError(
                f"Sample {position} is an FRB but does not provide toa and dm; both "
                "are required to centre the window on the burst."
            )

        source_id = str(row.get("id") or "").strip() or Path(fits_path).stem
        source = str(row.get("source") or "").strip() or None
        extras = {
            key: value
            for key, value in row.items()
            if key not in {"fits_path", "label", "toa", "dm", "id", "source"}
            and value not in (None, "")
        }
        return {
            "fits_path": fits_path,
            "label": label,
            "toa": toa,
            "dm": dm,
            "source_id": source_id,
            "source": source,
            "extras": extras,
        }

    def _prepare_one(
        self,
        entry: dict[str, Any],
        *,
        image_index: int,
        overwrite: bool,
    ) -> dict[str, Any]:
        fits_path = self._resolve(Path(entry["fits_path"]))
        if not fits_path.exists():
            raise FileNotFoundError(f"PSRFITS file not found: {fits_path}")

        image_id = f"{ANONYMIZED_IMAGE_PREFIX}_{image_index:06d}"
        image_path = self.images_dir / ANONYMIZED_IMAGE_DIR / f"{image_id}.png"
        if image_path.exists() and not overwrite:
            raise FileExistsError(f"Image already exists: {image_path}. Use --overwrite.")

        layout = self._reader.read_layout(fits_path)
        center_s, window_mode = self._window_center(entry, layout, image_index)
        plan = plan_window(
            center_s=center_s,
            window_seconds=self.config.window_seconds,
            tsamp_s=layout.tsamp_s,
            nsamp_total=layout.nsamp_total,
            time_bins=self.config.time_bins,
        )
        block = self._reader.read_block(
            fits_path,
            start_sample=plan.start_sample,
            stop_sample=plan.stop_sample,
            layout=layout,
        )

        median, sigma = robust_channel_stats(block.data)
        keep = channel_keep_mask(
            median=median,
            sigma=sigma,
            weights=block.channel_weights,
            zap_sigma=self.config.zap_sigma,
            edge_channels=self.config.edge_channels,
        )
        zscored = normalize_per_channel(block.data, median=median, sigma=sigma, keep=keep)
        averaged, freq_factor, trim_low = block_average(
            zscored,
            keep=keep,
            time_factor=plan.time_factor,
            freq_bins=self.config.freq_bins,
        )
        averaged = flatten_rows(averaged)
        frequency_axis = averaged_frequency_axis(
            block.frequencies_mhz,
            freq_bins=self.config.freq_bins,
            freq_factor=freq_factor,
            trim_low=trim_low,
        )
        bin_seconds = plan.time_factor * layout.tsamp_s
        time_axis = np.arange(averaged.shape[1], dtype=float) * bin_seconds

        spectrum = DynamicSpectrum(
            data=averaged,
            time_axis=time_axis,
            frequency_axis=frequency_axis,
        )
        render_spectrum_png(
            spectrum,
            image_path=image_path,
            cmap=self.config.cmap,
            normalization=self.config.normalization,
            percentile_low=self.config.percentile_low,
            percentile_high=self.config.percentile_high,
            width=self.config.width,
            height=self.config.height,
            dpi=self.config.dpi,
            title=image_id if self.config.include_title else None,
        )
        LOGGER.info("Imagem gerada: %s", image_path)

        parameters: dict[str, Any] = {
            "origin": "real",
            "source": entry["source"],
            "dm": entry["dm"],
            "toa_s": entry["toa"],
            **entry["extras"],
        }
        preprocessing = {
            "window_seconds_requested": self.config.window_seconds,
            "window_seconds_actual": plan.n_samples * layout.tsamp_s,
            "window_start_seconds": plan.start_sample * layout.tsamp_s,
            "window_mode": window_mode,
            "toa_reference": self.config.toa_reference if entry["label"] == "FRB" else None,
            "time_bins": self.config.time_bins,
            "freq_bins": self.config.freq_bins,
            "time_factor": plan.time_factor,
            "freq_factor": freq_factor,
            "channels_total": layout.nchan,
            "channels_zapped": int(np.count_nonzero(~keep)),
            "zap_sigma": self.config.zap_sigma,
            "edge_channels": self.config.edge_channels,
            "per_channel_normalization": "median-clipped-std-zscore+row-flatten",
            "reader": self.config.reader,
            "seed": self.config.seed,
        }
        return {
            "sample_id": image_id,
            "source_sample_id": entry["source_id"],
            "image_path": self._relative_path(image_path),
            "fits_path": self._relative_path(fits_path),
            "true_label": entry["label"],
            "simulation_parameters": parameters,
            "real_parameters": parameters,
            "preprocessing": preprocessing,
            "plot": {
                "backend": "prepare-real",
                "cmap": self.config.cmap,
                "normalization": self.config.normalization,
                "width": self.config.width,
                "height": self.config.height,
                "title": self.config.include_title,
                "anonymized_image_path": True,
            },
        }

    def _window_center(
        self,
        entry: dict[str, Any],
        layout: Any,
        image_index: int,
    ) -> tuple[float, str]:
        duration = layout.duration_s
        if entry["label"] == "FRB":
            center = burst_center_s(
                toa_s=float(entry["toa"]),
                dm=float(entry["dm"]),
                f_low_mhz=float(layout.frequencies_mhz[0]),
                f_high_mhz=float(layout.frequencies_mhz[-1]),
                toa_reference=self.config.toa_reference,
            )
            return center, "burst"

        if entry["toa"] is not None:
            return float(entry["toa"]), "toa"

        mode = self.config.negative_window
        half = min(self.config.window_seconds / 2.0, duration / 2.0)
        if mode == "center":
            return duration / 2.0, mode
        if mode == "start":
            return half, mode
        rng = np.random.default_rng([self.config.seed, image_index])
        low, high = half, duration - half
        if high <= low:
            return duration / 2.0, "center"
        return float(rng.uniform(low, high)), mode

    def _reset_outputs(self) -> None:
        for target in (self.images_dir, self.manifest_path):
            if target.is_dir():
                shutil.rmtree(target)
                LOGGER.info("Artefato antigo removido: %s", target)
            elif target.exists():
                target.unlink()
                LOGGER.info("Artefato antigo removido: %s", target)

    def _resolve(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()


def _optional_float(value: Any, *, field: str, position: int) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Sample {position} has an invalid value for {field}: {value!r}"
        ) from exc
