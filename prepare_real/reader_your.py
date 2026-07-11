from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from prepare_real.reader import PsrfitsLayout, SpectrumBlock


LOGGER = logging.getLogger(__name__)


def _load_your():
    try:
        from your import Your
    except ImportError as exc:
        raise RuntimeError(
            "The 'your' library is not installed. Install it with "
            "uv pip install -e '.[real]' or use --reader astropy."
        ) from exc
    return Your


def read_layout(path: Path) -> PsrfitsLayout:
    Your = _load_your()
    reader = Your(str(path))
    header = reader.your_header

    frequencies = np.asarray(reader.chan_freqs, dtype=float).reshape(-1)
    descending = bool(frequencies.size > 1 and frequencies[0] > frequencies[-1])
    if descending:
        frequencies = frequencies[::-1].copy()

    # nsblk/nsub/zero_off are astropy-backend details; 'your' abstracts
    # per-sample access and applies DAT_SCL/DAT_OFFS/ZERO_OFF internally.
    return PsrfitsLayout(
        path=Path(path),
        nchan=int(header.nchans),
        npol=int(getattr(header, "npol", 1) or 1),
        nsblk=1,
        nsub=int(header.nspectra),
        nbits=int(header.nbits),
        tsamp_s=float(header.tsamp),
        nsamp_total=int(header.nspectra),
        frequencies_mhz=frequencies,
        frequency_descending=descending,
        zero_off=0.0,
    )


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

    Your = _load_your()
    reader = Your(str(path))
    if hasattr(reader, "read_subint") and hasattr(reader, "nsamp_per_subint"):
        # PSRFITS path: read_subint applies DAT_SCL/DAT_OFFS/DAT_WTS in float,
        # whereas the high-level get_data truncates back to the native dtype
        # (uint8) and would lose the fractional part of the scales.
        nsblk = int(reader.nsamp_per_subint)
        row0 = start_sample // nsblk
        row1 = (stop_sample - 1) // nsblk + 1
        parts = []
        for row in range(row0, row1):
            subint = np.asarray(reader.read_subint(row), dtype=np.float32)
            parts.append(subint.reshape(nsblk, -1, layout.nchan)[:, 0, :])
        samples = np.concatenate(parts, axis=0)
        offset = start_sample - row0 * nsblk
        data = samples[offset : offset + (stop_sample - start_sample)]
    else:
        data = np.asarray(
            reader.get_data(nstart=start_sample, nsamp=stop_sample - start_sample),
            dtype=np.float32,
        )
        if data.ndim == 3:
            data = data[:, 0, :]
    data = np.ascontiguousarray(data.T)

    if layout.frequency_descending:
        data = data[::-1, :].copy()

    return SpectrumBlock(
        data=data,
        frequencies_mhz=layout.frequencies_mhz,
        tsamp_s=layout.tsamp_s,
        start_sample=start_sample,
        channel_weights=np.ones(layout.nchan, dtype=np.float32),
    )
