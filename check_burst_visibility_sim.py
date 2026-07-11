#!/usr/bin/env python3
"""Measure burst visibility in the SIMULATED samples, on the SAME grid as the real ones.

This exists to make the comparison "recall X on the sharp real bursts against
recall Y on the simulated benchmark" a controlled one: without measuring both
sets with the same diagnostic, the comparison assumes -- without demonstrating
-- that the injected bursts all sit well above the noise.

The core of the measurement is imported from
`check_burst_visibility.dedispersed_snr`, with no reimplementation: truncated
dedispersion at the sample DM followed by a boxcar matched filter, with the MAD
re-estimated at each width.

COMMON GRID
-----------
The simulated PSRFITS has 10173 samples of 196.608 us x 2048 channels, whereas
the real image handed to the model has 1024 bins of 1.966 ms x 256 channels.
Measuring the two at their native resolution would give `sigma` different
meanings: boxcar widths are counted in bins, so the same trial would cover
6.3 ms on one side and 63 ms on the other. Here the simulated array is decimated
before the measurement with the same `block_average` as the real path, using
`time_factor=10` and `freq_bins=256`:

    196.608 us x 10 = 1.966 ms  <->  98.304 us x 20 = 1.966 ms   (real)
    2048 channels / 8 = 256     <->  4096 channels / 16 = 256    (real)

that is, both sets end up measured in bins of the same physical duration and
the same number of channels over the same L band. The final 3 samples (0.6 ms)
are discarded so that 10170 is divisible by 10.

ASSUMED DIFFERENCE
------------------
The variance zap and the edge-channel cut of the real path are not applied here:
the simulated band has neither narrowband RFI nor edge rolloff to mitigate, and
applying a 3-sigma cut to clean noise would only discard good channels. This
asymmetry is real and favours the simulated set; it is stated in the article.

SWEEPS NOT CONTAINED
--------------------
With DM drawn up to ~900 pc/cm3, the sweep from 1000 to 1500 MHz reaches 2.07 s
in a 2 s file: only 549 of the 1000 simulated FRBs have the whole sweep inside
the file. The `sweep_contained` column marks which ones, and the analysis should
be restricted to them -- sweeps clipped by the edge are not comparable to
complete sweeps, and patching the partial coverage was tested and discarded (see
the docstring of `dedispersed_snr`).

    uv run python check_burst_visibility_sim.py \
      --manifest dataset/metadata/image_manifest.jsonl \
      --out dataset/metadata/burst_visibility_sim.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from check_burst_visibility import DISPERSION_CONSTANT_S_MHZ2, dedispersed_snr
from plot_dataset.plotter import DatasetPlotter, PlotConfig
from prepare_real.preprocess import (
    averaged_frequency_axis,
    block_average,
    flatten_rows,
    normalize_per_channel,
    robust_channel_stats,
)

# Grid shared with the real path (see docstring).
TIME_FACTOR = 10
FREQ_BINS = 256

# Blind grid used on the negatives, identical to the real path.
BLIND_DM_GRID = range(100, 700, 25)

BAND_LOW_MHZ = 1000.0
BAND_HIGH_MHZ = 1500.0


def build_reader(project_root: Path) -> DatasetPlotter:
    """Reuse the PSRFITS reader of the benchmark plotting module.

    The PlotConfig output paths are required by the constructor but are not used
    here: only the PSRFITS decoder is called, never the rendering.
    """
    config = PlotConfig(
        labels_path=Path("dataset/metadata/labels.jsonl"),
        output_dir=Path("dataset/images"),
        manifest_path=Path("dataset/metadata/image_manifest.jsonl"),
    )
    return DatasetPlotter(config=config, project_root=project_root)


def sweep_contained(params: dict) -> bool:
    """Does the whole sweep fit inside the 2 s of the file?

    O burst chega em `arrival_time` na `reference_frequency`; o atraso relativo
    a essa referencia e negativo acima dela e positivo abaixo.
    """
    dm = float(params["dm"])
    arrival = float(params["arrival_time"])
    reference = float(params["reference_frequency"])
    top = arrival + DISPERSION_CONSTANT_S_MHZ2 * dm * (BAND_HIGH_MHZ**-2 - reference**-2)
    bottom = arrival + DISPERSION_CONSTANT_S_MHZ2 * dm * (BAND_LOW_MHZ**-2 - reference**-2)
    return top >= 0.0 and bottom <= 2.0


def load_spectrum(reader: DatasetPlotter, fits_path: Path):
    """Return (array decimated on the common grid, frequency axis, s per bin)."""
    spectrum = reader._read_with_astropy(fits_path)
    data = np.asarray(spectrum.data, dtype=np.float32)

    frequencies = np.asarray(spectrum.frequency_axis, dtype=float)
    if frequencies.size > 1 and frequencies[0] > frequencies[-1]:
        frequencies = frequencies[::-1].copy()
        data = data[::-1, :].copy()

    median, sigma = robust_channel_stats(data)
    keep = sigma > np.finfo(np.float32).tiny
    zscored = normalize_per_channel(data, median=median, sigma=sigma, keep=keep)

    # block_average requires nsamp to be a multiple of time_factor; 10173 -> 10170.
    usable = (zscored.shape[1] // TIME_FACTOR) * TIME_FACTOR
    zscored = zscored[:, :usable]

    averaged, freq_factor, trim_low = block_average(
        zscored, keep=keep, time_factor=TIME_FACTOR, freq_bins=FREQ_BINS
    )
    averaged = flatten_rows(averaged)
    freqs = averaged_frequency_axis(
        frequencies, freq_bins=FREQ_BINS, freq_factor=freq_factor, trim_low=trim_low
    )

    time_axis = np.asarray(spectrum.time_axis, dtype=float)
    tsamp = float(time_axis[1] - time_axis[0]) if time_axis.size > 1 else 0.0
    return averaged, freqs, tsamp * TIME_FACTOR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("dataset/metadata/image_manifest.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--only-frb",
        action="store_true",
        help="Measure only the FRB samples (the blind scan of the negatives is expensive).",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    reader = build_reader(project_root)

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.only_frb:
        records = [r for r in records if r["true_label"] == "FRB"]

    rows = []
    for index, record in enumerate(records, start=1):
        fits_path = project_root / record["fits_path"]
        image, freqs, bin_seconds = load_spectrum(reader, fits_path)
        params = record.get("simulation_parameters") or {}
        dm = params.get("dm")

        if record["true_label"] == "FRB" and dm is not None:
            peak, t_peak = dedispersed_snr(image, freqs, bin_seconds, float(dm))
            contained = sweep_contained(params)
        else:
            peak, t_peak = max(
                (dedispersed_snr(image, freqs, bin_seconds, float(trial)) for trial in BLIND_DM_GRID),
                key=lambda item: item[0],
            )
            contained = ""

        rows.append(
            {
                "sample_id": record["sample_id"],
                "source_sample_id": record["source_sample_id"],
                "true_label": record["true_label"],
                "dm": dm if dm is not None else "",
                "flux_density": params.get("flux_density", ""),
                "width": params.get("width", ""),
                "arrival_time": params.get("arrival_time", ""),
                "sweep_contained": int(contained) if contained != "" else "",
                "peak_sigma": round(peak, 2),
                "peak_time_s": round(t_peak, 5),
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"[{index}/{len(records)}] {rows[-1]['source_sample_id']}: {rows[-1]['peak_sigma']} sigma", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
