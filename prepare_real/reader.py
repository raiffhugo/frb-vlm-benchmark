from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PsrfitsLayout:
    path: Path
    nchan: int
    npol: int
    nsblk: int
    nsub: int
    nbits: int
    tsamp_s: float
    nsamp_total: int
    frequencies_mhz: np.ndarray
    frequency_descending: bool
    zero_off: float

    @property
    def duration_s(self) -> float:
        return self.nsamp_total * self.tsamp_s


@dataclass(frozen=True)
class SpectrumBlock:
    data: np.ndarray
    frequencies_mhz: np.ndarray
    tsamp_s: float
    start_sample: int
    channel_weights: np.ndarray


def _find_subint_hdu(hdul: fits.HDUList) -> fits.BinTableHDU:
    if "SUBINT" in hdul and "DATA" in hdul["SUBINT"].columns.names:
        return hdul["SUBINT"]

    for hdu in hdul:
        if not isinstance(hdu, fits.BinTableHDU):
            continue
        if hdu.columns is not None and "DATA" in hdu.columns.names:
            return hdu
    raise ValueError("No SUBINT/DATA extension found.")


def _header_float(header: fits.Header, key: str, default: float) -> float:
    value = header.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _frequency_axis(
    primary_header: fits.Header,
    subint: fits.BinTableHDU,
    nchan: int,
) -> np.ndarray:
    if "DAT_FREQ" in subint.columns.names:
        freq = np.asarray(subint.data["DAT_FREQ"][0], dtype=float).reshape(-1)
        if freq.size >= nchan:
            return freq[:nchan]

    obsfreq = _header_float(primary_header, "OBSFREQ", 0.0)
    obsbw = _header_float(primary_header, "OBSBW", 0.0)
    chan_bw = _header_float(subint.header, "CHAN_BW", 0.0)
    if obsfreq and obsbw:
        start = obsfreq - obsbw / 2.0 + chan_bw / 2.0
        stop = obsfreq + obsbw / 2.0 - chan_bw / 2.0
        return np.linspace(start, stop, nchan)

    return np.arange(nchan, dtype=float)


def read_layout(path: Path) -> PsrfitsLayout:
    with fits.open(path, memmap=True) as hdul:
        primary_header = hdul[0].header
        subint = _find_subint_hdu(hdul)
        header = subint.header

        nchan = int(header.get("NCHAN", primary_header.get("OBSNCHAN", 0)))
        nsblk = int(header.get("NSBLK", 0))
        if nchan <= 0 or nsblk <= 0:
            raise ValueError(f"PSRFITS header without valid NCHAN/NSBLK: {path}")

        npol = max(1, int(header.get("NPOL", 1)))
        nbits = int(header.get("NBITS", 8))
        nsub = int(header.get("NAXIS2", len(subint.data)))
        tsamp = _header_float(header, "TBIN", 0.0)
        if tsamp <= 0.0:
            raise ValueError(f"PSRFITS header without a valid TBIN: {path}")

        frequencies = _frequency_axis(primary_header, subint, nchan)
        descending = bool(frequencies.size > 1 and frequencies[0] > frequencies[-1])
        if descending:
            frequencies = frequencies[::-1].copy()

        return PsrfitsLayout(
            path=Path(path),
            nchan=nchan,
            npol=npol,
            nsblk=nsblk,
            nsub=nsub,
            nbits=nbits,
            tsamp_s=tsamp,
            nsamp_total=nsub * nsblk,
            frequencies_mhz=frequencies,
            frequency_descending=descending,
            zero_off=_header_float(header, "ZERO_OFF", 0.0),
        )


def _unpack_bits(packed: np.ndarray, *, nbits: int) -> np.ndarray:
    mask = (1 << nbits) - 1
    shifts = np.arange(8 - nbits, -1, -nbits, dtype=np.uint8)
    return ((packed.reshape(-1)[:, None] >> shifts[None, :]) & mask).reshape(-1)


def _decode_rows(
    raw_rows: np.ndarray,
    *,
    layout: PsrfitsLayout,
    n_rows: int,
) -> np.ndarray:
    expected = n_rows * layout.nsblk * layout.npol * layout.nchan
    values = np.asarray(raw_rows)

    if layout.nbits in {1, 2, 4}:
        values = _unpack_bits(
            np.ascontiguousarray(values, dtype=np.uint8), nbits=layout.nbits
        )
    else:
        values = values.reshape(-1)

    if values.size < expected:
        raise ValueError(
            "PSRFITS DATA smaller than expected: "
            f"{values.size} < {expected} in {layout.path}"
        )
    if values.size > expected:
        if values.size % expected != 0:
            raise ValueError(
                f"PSRFITS DATA with inconsistent size in {layout.path}: "
                f"{values.size} is not a multiple of {expected}"
            )
        nbin = values.size // expected
        if nbin != 1:
            raise ValueError(
                f"PSRFITS DATA with NBIN={nbin} is not supported in {layout.path}"
            )

    return values[:expected].reshape(
        n_rows, layout.nsblk, layout.npol, layout.nchan
    ).astype(np.float32)


def _scale_column(
    subint: fits.BinTableHDU,
    name: str,
    *,
    row0: int,
    row1: int,
    layout: PsrfitsLayout,
    default: float,
) -> np.ndarray:
    n_rows = row1 - row0
    shape = (n_rows, layout.npol, layout.nchan)
    if name not in subint.columns.names:
        return np.full(shape, default, dtype=np.float32)

    values = np.asarray(subint.data[name][row0:row1], dtype=np.float32)
    expected = n_rows * layout.npol * layout.nchan
    if values.size == expected:
        return values.reshape(shape)
    if values.size == n_rows * layout.nchan and layout.npol > 1:
        return np.repeat(
            values.reshape(n_rows, 1, layout.nchan), layout.npol, axis=1
        )
    LOGGER.warning(
        "Column %s has an unexpected size (%d) in %s; falling back to %.1f.",
        name,
        values.size,
        layout.path,
        default,
    )
    return np.full(shape, default, dtype=np.float32)


def read_block(
    path: Path,
    *,
    start_sample: int,
    stop_sample: int,
    layout: PsrfitsLayout | None = None,
) -> SpectrumBlock:
    if layout is None:
        layout = read_layout(path)

    if not 0 <= start_sample < stop_sample <= layout.nsamp_total:
        raise ValueError(
            f"Invalid window [{start_sample}, {stop_sample}) for a file with "
            f"{layout.nsamp_total} samples: {path}"
        )

    row0 = start_sample // layout.nsblk
    row1 = (stop_sample - 1) // layout.nsblk + 1
    n_rows = row1 - row0

    with fits.open(path, memmap=True) as hdul:
        subint = _find_subint_hdu(hdul)
        raw = _decode_rows(
            np.asarray(subint.data["DATA"][row0:row1]),
            layout=layout,
            n_rows=n_rows,
        )
        scales = _scale_column(
            subint, "DAT_SCL", row0=row0, row1=row1, layout=layout, default=1.0
        )
        offsets = _scale_column(
            subint, "DAT_OFFS", row0=row0, row1=row1, layout=layout, default=0.0
        )
        if "DAT_WTS" in subint.columns.names:
            weights = np.asarray(subint.data["DAT_WTS"][row0:row1], dtype=np.float32)
            weights = weights.reshape(n_rows, -1)[:, : layout.nchan].mean(axis=0)
        else:
            weights = np.ones(layout.nchan, dtype=np.float32)

    physical = (raw - np.float32(layout.zero_off)) * scales[:, None, :, :] + offsets[
        :, None, :, :
    ]
    if layout.npol == 1:
        intensity = physical[:, :, 0, :]
    else:
        intensity = physical[:, :, : min(2, layout.npol), :].mean(axis=2)

    samples = intensity.reshape(n_rows * layout.nsblk, layout.nchan)
    offset_in_rows = start_sample - row0 * layout.nsblk
    samples = samples[offset_in_rows : offset_in_rows + (stop_sample - start_sample)]

    data = np.ascontiguousarray(samples.T, dtype=np.float32)
    frequencies = layout.frequencies_mhz
    if layout.frequency_descending:
        data = data[::-1, :].copy()
        weights = weights[::-1].copy()

    return SpectrumBlock(
        data=data,
        frequencies_mhz=frequencies,
        tsamp_s=layout.tsamp_s,
        start_sample=start_sample,
        channel_weights=weights,
    )
