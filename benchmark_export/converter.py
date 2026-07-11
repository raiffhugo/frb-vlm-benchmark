from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)

BENCHMARK_IMAGE_PREFIX = "sample"
DEFAULT_FITS_DIR = "fits"
DEFAULT_METADATA_DIR = "metadata"
DEFAULT_MANIFEST_JSONL = "benchmark_manifest.jsonl"
DEFAULT_MANIFEST_CSV = "benchmark_manifest.csv"


@dataclass(frozen=True)
class BenchmarkExportConfig:
    labels_path: Path
    output_dir: Path
    manifest_jsonl: Path | None = None
    manifest_csv: Path | None = None


def convert_to_benchmark_fits(src: Path, dst: Path) -> None:
    """Convert createSearchFile output into a generic benchmark FITS file."""
    warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    src = src.expanduser()
    dst = dst.expanduser()
    with fits.open(src, mode="readonly", lazy_load_hdus=False) as fp:
        primary = fp[0].copy()
        subint = _find_subint_hdu(fp, src)
        _copy_timing_and_pointing_keys(primary.header, subint.header)

        data = subint.data
        if data is None:
            raise RuntimeError(f"{src}: SUBINT HDU has no table data")
        col_data = data["DATA"]

        if col_data.ndim == 4:
            new_shape = col_data.shape + (1,)
            new_data = col_data.reshape(new_shape)
            new_hdu = _subint_with_5d_data(data, subint.header, new_shape, new_data)
        else:
            new_hdu = subint.copy()

        hdul = fits.HDUList([primary, new_hdu])
        dst.parent.mkdir(parents=True, exist_ok=True)
        hdul.writeto(dst, overwrite=True)


class BenchmarkExporter:
    def __init__(self, *, config: BenchmarkExportConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.labels_path = self._resolve(config.labels_path)
        self.output_dir = self._resolve(config.output_dir)
        self.fits_dir = self.output_dir / DEFAULT_FITS_DIR
        self.metadata_dir = self.output_dir / DEFAULT_METADATA_DIR
        self.manifest_jsonl = self._resolve(
            config.manifest_jsonl or self.metadata_dir / DEFAULT_MANIFEST_JSONL
        )
        self.manifest_csv = self._resolve(
            config.manifest_csv or self.metadata_dir / DEFAULT_MANIFEST_CSV
        )

    def export(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        records = self._load_labels()
        if overwrite:
            self._reset_outputs()
        self.fits_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_jsonl.exists() and not overwrite:
            raise FileExistsError(
                f"Manifest already exists: {self.manifest_jsonl}. Use --overwrite."
            )
        if self.manifest_csv.exists() and not overwrite:
            raise FileExistsError(
                f"Manifest already exists: {self.manifest_csv}. Use --overwrite."
            )

        exported: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for index, record in enumerate(
            tqdm(records, desc="Exporting benchmark FITS", unit="file")
        ):
            export_record = self._export_one(record, index=index, overwrite=overwrite)
            exported.append(export_record)
            if export_record["conversion_status"] != "ok":
                errors.append(export_record)

        self._write_manifests(exported)
        LOGGER.info("Benchmark export finished in %s", self.output_dir)
        if errors:
            raise RuntimeError(
                f"Failed to export {len(errors)} file(s). "
                f"See {self.manifest_jsonl}."
            )
        return exported

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
        required = {"sample_id", "label", "fits_path"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Line {line_number} is missing required keys: {', '.join(missing)}"
            )

    def _reset_outputs(self) -> None:
        if self.fits_dir.exists():
            for path in self.fits_dir.glob(f"{BENCHMARK_IMAGE_PREFIX}_*.fits"):
                if path.is_file():
                    path.unlink()
                    LOGGER.info("Old artifact removed: %s", path)
        for path in (self.manifest_jsonl, self.manifest_csv):
            if path.exists():
                path.unlink()
                LOGGER.info("Old artifact removed: %s", path)

    def _export_one(
        self,
        record: dict[str, Any],
        *,
        index: int,
        overwrite: bool,
    ) -> dict[str, Any]:
        sample_id = self._anonymous_sample_id(index)
        source_path = self._resolve(Path(record["fits_path"]))
        exported_path = self.fits_dir / f"{sample_id}.fits"

        if exported_path.exists() and not overwrite:
            raise FileExistsError(
                f"Exported file already exists: {exported_path}. Use --overwrite."
            )

        export_record = {
            "sample_id": sample_id,
            "source_sample_id": record["sample_id"],
            "true_label": record["label"],
            "source_fits_path": self._relative_path(source_path),
            "exported_fits_path": self._relative_path(exported_path),
            "conversion_status": "ok",
            "error": None,
        }

        try:
            convert_to_benchmark_fits(source_path, exported_path)
        except Exception as exc:
            export_record["conversion_status"] = "error"
            export_record["error"] = f"{type(exc).__name__}: {exc}"
            LOGGER.error(
                "Failed to export %s to %s: %s",
                source_path,
                exported_path,
                exc,
            )
        return export_record

    def _anonymous_sample_id(self, index: int) -> str:
        if index < 0:
            raise ValueError("index must be >= 0.")
        return f"{BENCHMARK_IMAGE_PREFIX}_{index:06d}"

    def _write_manifests(self, records: list[dict[str, Any]]) -> None:
        self.manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_jsonl.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        self.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id",
            "source_sample_id",
            "true_label",
            "source_fits_path",
            "exported_fits_path",
            "conversion_status",
            "error",
        ]
        with self.manifest_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

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


def _find_subint_hdu(hdul: fits.HDUList, src: Path) -> fits.BinTableHDU:
    for hdu in hdul[1:]:
        if hdu.name.upper() == "SUBINT":
            if not isinstance(hdu, fits.BinTableHDU):
                raise RuntimeError(f"{src}: SUBINT HDU is not a binary table")
            if hdu.columns is None or "DATA" not in hdu.columns.names:
                raise RuntimeError(f"{src}: SUBINT HDU has no DATA column")
            return hdu
    raise RuntimeError(f"{src}: SUBINT HDU not found")


def _copy_timing_and_pointing_keys(
    primary_header: fits.Header,
    subint_header: fits.Header,
) -> None:
    for key in ("STT_IMJD", "STT_SMJD", "STT_OFFS", "RA", "DEC", "RAJ", "DECJ"):
        if key not in primary_header and key in subint_header:
            primary_header[key] = subint_header[key]


def _subint_with_5d_data(
    data: fits.FITS_rec,
    header: fits.Header,
    new_shape: tuple[int, ...],
    new_data: Any,
) -> fits.BinTableHDU:
    _, nt, npol, nchan, nbin = new_shape
    cols = []
    for column in data.columns:
        if column.name == "DATA":
            cols.append(
                fits.Column(
                    name="DATA",
                    format=f"{nt * npol * nchan * nbin}B",
                    dim=f"({nbin},{nchan},{npol},{nt})",
                    array=new_data,
                )
            )
        else:
            cols.append(
                fits.Column(
                    name=column.name,
                    format=column.format,
                    dim=column.dim,
                    unit=column.unit,
                    array=data[column.name],
                )
            )
    return fits.BinTableHDU.from_columns(cols, header=header, name="SUBINT")
