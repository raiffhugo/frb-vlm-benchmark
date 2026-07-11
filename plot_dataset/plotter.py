from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)

CLASSES = ("frb", "rfi", "noise")
ANONYMIZED_IMAGE_DIR = "samples"
ANONYMIZED_IMAGE_PREFIX = "sample"
DEFAULT_IMAGE_WIDTH = 1536
DEFAULT_IMAGE_HEIGHT = 1152
DEFAULT_IMAGE_DPI = 192


@dataclass(frozen=True)
class PlotConfig:
    labels_path: Path
    output_dir: Path
    manifest_path: Path
    cmap: str = "viridis"
    normalization: str = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    width: int = DEFAULT_IMAGE_WIDTH
    height: int = DEFAULT_IMAGE_HEIGHT
    dpi: int = DEFAULT_IMAGE_DPI
    include_title: bool = False

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        if self.dpi <= 0:
            raise ValueError("dpi must be positive.")
        if self.normalization not in {"linear", "minmax", "zscore", "percentile"}:
            raise ValueError(f"Invalid normalization: {self.normalization}")
        if not 0 <= self.percentile_low < self.percentile_high <= 100:
            raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")


@dataclass(frozen=True)
class DynamicSpectrum:
    data: np.ndarray
    time_axis: np.ndarray
    frequency_axis: np.ndarray


def normalize_for_display(
    data: np.ndarray,
    *,
    normalization: str,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    clean = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(clean)
    if not finite.any():
        raise ValueError("Spectrum has no finite values.")

    values = clean[finite]
    if normalization == "linear":
        normalized = clean
    elif normalization == "minmax":
        low = float(np.min(values))
        high = float(np.max(values))
        normalized = _scale_to_unit(clean, low, high)
    elif normalization == "zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std == 0.0:
            std = 1.0
        normalized = (clean - mean) / std
    elif normalization == "percentile":
        low, high = np.percentile(values, [percentile_low, percentile_high])
        normalized = _scale_to_unit(clean, float(low), float(high))
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    return np.nan_to_num(normalized, copy=False)


def _scale_to_unit(data: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (data - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def render_spectrum_png(
    spectrum: DynamicSpectrum,
    *,
    image_path: Path,
    cmap: str,
    normalization: str,
    percentile_low: float,
    percentile_high: float,
    width: int,
    height: int,
    dpi: int,
    title: str | None = None,
) -> None:
    data = normalize_for_display(
        spectrum.data,
        normalization=normalization,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    fig_size = (width / dpi, height / dpi)
    time_end = float(spectrum.time_axis[-1])
    if spectrum.time_axis.size > 1:
        time_end += float(spectrum.time_axis[1] - spectrum.time_axis[0])

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    extent = [
        float(spectrum.time_axis[0]),
        time_end,
        float(spectrum.frequency_axis[0]),
        float(spectrum.frequency_axis[-1]),
    ]
    ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        extent=extent,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (MHz)")
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)


class DatasetPlotter:
    def __init__(self, *, config: PlotConfig, project_root: Path) -> None:
        config.validate()
        self.config = config
        self.project_root = project_root.resolve()
        self.labels_path = self._resolve(config.labels_path)
        self.output_dir = self._resolve(config.output_dir)
        self.manifest_path = self._resolve(config.manifest_path)

    def plot_all(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        records = self._load_labels()
        if overwrite:
            self._reset_outputs()
        self._create_directories()

        if self.manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Manifest already exists: {self.manifest_path}. Use --overwrite."
            )

        manifest_records: list[dict[str, Any]] = []
        for image_index, record in enumerate(
            tqdm(records, desc="Plotting PSRFITS", unit="image")
        ):
            manifest_records.append(
                self._plot_one(record, image_index=image_index, overwrite=overwrite)
            )

        self._write_manifest(manifest_records)
        LOGGER.info("Image manifest written to %s", self.manifest_path)
        return manifest_records

    def _load_labels(self) -> list[dict[str, Any]]:
        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels.jsonl not found: {self.labels_path}")

        records: list[dict[str, Any]] = []
        with self.labels_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid line in {self.labels_path}:{line_number}"
                    ) from exc
                self._validate_label_record(record, line_number)
                records.append(record)

        if not records:
            raise ValueError(f"No samples found in {self.labels_path}")
        return records

    def _validate_label_record(self, record: dict[str, Any], line_number: int) -> None:
        required = {"sample_id", "fits_path", "label", "parameters"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Line {line_number} is missing required keys: {', '.join(missing)}"
            )

    def _create_directories(self) -> None:
        (self.output_dir / ANONYMIZED_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _reset_outputs(self) -> None:
        targets: list[Path] = [self.output_dir, self.manifest_path]

        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
                LOGGER.info("Old artifact removed: %s", target)
            elif target.exists():
                target.unlink()
                LOGGER.info("Old artifact removed: %s", target)

    def _plot_one(
        self,
        record: dict[str, Any],
        *,
        image_index: int,
        overwrite: bool,
    ) -> dict[str, Any]:
        label = str(record["label"]).lower()
        if label not in CLASSES:
            raise ValueError(f"Classe desconhecida no labels.jsonl: {record['label']}")

        fits_path = self._resolve(Path(record["fits_path"]))
        if not fits_path.exists():
            raise FileNotFoundError(f"PSRFITS file not found: {fits_path}")
        if fits_path.suffix.lower() not in {".fits", ".sf"}:
            raise ValueError(f"Unsupported PSRFITS extension: {fits_path}")

        image_id = self._image_id(image_index)
        image_path = self._image_path_for_index(image_index)
        if image_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Image already exists: {image_path}. Use --overwrite."
                )
            image_path.unlink()

        try:
            spectrum = self._read_with_astropy(fits_path)
            self._plot_spectrum(spectrum, image_id, image_path)
            plot_backend = "astropy"
        except Exception as exc:
            LOGGER.warning(
                "Failed to plot %s with astropy: %s. Trying pfits_plot.",
                fits_path,
                exc,
            )
            self._plot_with_pfits(fits_path, image_path)
            plot_backend = "pfits_plot"

        manifest_record = {
            "sample_id": image_id,
            "source_sample_id": record["sample_id"],
            "image_path": self._relative_path(image_path),
            "fits_path": self._relative_path(fits_path),
            "true_label": record["label"],
            "simulation_parameters": record["parameters"],
            "plot": {
                "backend": plot_backend,
                "cmap": self.config.cmap,
                "normalization": self.config.normalization,
                "width": self.config.width,
                "height": self.config.height,
                "title": self.config.include_title,
                "anonymized_image_path": True,
            },
        }
        LOGGER.info("Image generated: %s", image_path)
        return manifest_record

    def _image_id(self, image_index: int) -> str:
        if image_index < 0:
            raise ValueError("image_index must be >= 0.")
        return f"{ANONYMIZED_IMAGE_PREFIX}_{image_index:06d}"

    def _image_path_for_index(self, image_index: int) -> Path:
        return (
            self.output_dir
            / ANONYMIZED_IMAGE_DIR
            / f"{self._image_id(image_index)}.png"
        )

    def _read_with_astropy(self, path: Path) -> DynamicSpectrum:
        with fits.open(path, memmap=False) as hdul:
            primary_header = hdul[0].header
            subint = self._find_subint_hdu(hdul)
            header = subint.header

            raw_data = np.asarray(subint.data["DATA"])
            nchan = int(header.get("NCHAN", primary_header.get("OBSNCHAN", 0)))
            nsblk = int(header.get("NSBLK", 0))
            nbits = int(header.get("NBITS", 8))
            if nchan <= 0 or nsblk <= 0:
                raise ValueError("PSRFITS header has no valid NCHAN/NSBLK.")

            spectrum = self._decode_psrfits_data(
                raw_data=raw_data,
                nchan=nchan,
                nsblk=nsblk,
                nbits=nbits,
            )
            time_axis = self._time_axis(header, nsblk, spectrum.shape[1])
            frequency_axis = self._frequency_axis(primary_header, header, nchan)
            return DynamicSpectrum(
                data=spectrum,
                time_axis=time_axis,
                frequency_axis=frequency_axis,
            )

    def _find_subint_hdu(self, hdul: fits.HDUList) -> fits.BinTableHDU:
        if "SUBINT" in hdul and "DATA" in hdul["SUBINT"].columns.names:
            return hdul["SUBINT"]

        for hdu in hdul:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            if hdu.columns is not None and "DATA" in hdu.columns.names:
                return hdu
        raise ValueError("No SUBINT/DATA extension found.")

    def _decode_psrfits_data(
        self,
        *,
        raw_data: np.ndarray,
        nchan: int,
        nsblk: int,
        nbits: int,
    ) -> np.ndarray:
        values = np.asarray(raw_data)
        if values.ndim >= 3 and values.shape[-1] == nchan:
            unpacked = values.reshape(-1, nchan)
        else:
            packed = np.asarray(values, dtype=np.uint8).reshape(-1)
            if nbits == 8:
                unpacked = packed.reshape(-1, nchan)
            elif nbits in {1, 2, 4}:
                unpacked = self._unpack_bits(packed, nbits=nbits, nchan=nchan)
            else:
                raise ValueError(f"Unsupported NBITS for packed DATA: {nbits}")

        expected_samples = max(1, raw_data.shape[0]) * nsblk
        if unpacked.shape[0] < expected_samples:
            raise ValueError(
                "PSRFITS DATA smaller than expected: "
                f"{unpacked.shape[0]} < {expected_samples}"
            )
        unpacked = unpacked[:expected_samples, :nchan]
        return unpacked.astype(np.float32).T

    def _unpack_bits(self, packed: np.ndarray, *, nbits: int, nchan: int) -> np.ndarray:
        mask = (1 << nbits) - 1
        shifts = np.arange(8 - nbits, -1, -nbits, dtype=np.uint8)
        unpacked = ((packed[:, None] >> shifts[None, :]) & mask).reshape(-1)
        usable = (unpacked.size // nchan) * nchan
        if usable == 0:
            raise ValueError("Packed DATA is empty after unpacking.")
        return unpacked[:usable].reshape(-1, nchan)

    def _time_axis(
        self,
        header: fits.Header,
        nsblk: int,
        nsamp: int,
    ) -> np.ndarray:
        tsamp = float(header.get("TBIN", 1.0))
        duration = nsamp * tsamp
        return np.linspace(0.0, duration, nsamp, endpoint=False)

    def _frequency_axis(
        self,
        primary_header: fits.Header,
        subint_header: fits.Header,
        nchan: int,
    ) -> np.ndarray:
        obsfreq = float(primary_header.get("OBSFREQ", 0.0))
        obsbw = float(primary_header.get("OBSBW", 0.0))
        chan_bw = float(subint_header.get("CHAN_BW", 0.0))

        if obsfreq and obsbw:
            start = obsfreq - obsbw / 2.0 + chan_bw / 2.0
            stop = obsfreq + obsbw / 2.0 - chan_bw / 2.0
            return np.linspace(start, stop, nchan)

        data_freq = subint_header.get("DAT_FREQ")
        if data_freq is not None:
            freq = np.asarray(data_freq, dtype=float).reshape(-1)
            if freq.size >= nchan:
                return freq[:nchan]

        return np.arange(nchan, dtype=float)

    def _plot_spectrum(
        self,
        spectrum: DynamicSpectrum,
        image_id: str,
        image_path: Path,
    ) -> None:
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

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        return normalize_for_display(
            data,
            normalization=self.config.normalization,
            percentile_low=self.config.percentile_low,
            percentile_high=self.config.percentile_high,
        )

    def _scale_to_unit(self, data: np.ndarray, low: float, high: float) -> np.ndarray:
        return _scale_to_unit(data, low, high)

    def _plot_with_pfits(self, fits_path: Path, image_path: Path) -> None:
        if shutil.which("pfits_plot") is None:
            raise RuntimeError("Fallback pfits_plot indisponivel no PATH.")

        with tempfile.TemporaryDirectory(prefix="pfits_plot_") as tmp:
            work_dir = Path(tmp)
            anonymous_fits = work_dir / "input.fits"
            shutil.copy2(fits_path, anonymous_fits)
            env = {
                **os.environ,
                "PGPLOT_DEV": "/png",
            }
            subprocess.run(
                ["pfits_plot", "-f", str(anonymous_fits), "-s1", "0", "-s2", "0"],
                cwd=work_dir,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            png_candidates = sorted(work_dir.glob("*.png"))
            if not png_candidates:
                raise RuntimeError("pfits_plot ran but produced no PNG.")
            shutil.move(str(png_candidates[0]), image_path)

    def _write_manifest(self, records: list[dict[str, Any]]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

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
