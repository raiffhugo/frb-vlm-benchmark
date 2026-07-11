#!/usr/bin/env python3
"""Diagnostic: is the burst actually visible in the image handed to the VLM?

Reproduces the `prepare-real` preprocessing exactly (the same functions from
`prepare_real`), dedisperses the decimated array at the catalogued DM and
measures the peak of the time profile in units of sigma. It neither modifies
nor runs the pipeline; it is a read-only diagnostic, used to separate "the
model got it wrong" from "the image carried no signal".

    uv run python check_burst_visibility.py \
      --manifest dataset_real_n100/metadata/image_manifest.jsonl \
      --out fast_frex/burst_visibility_n100.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from prepare_real import reader as astropy_reader
from prepare_real.preprocess import (
    DISPERSION_CONSTANT_S_MHZ2,
    averaged_frequency_axis,
    block_average,
    channel_keep_mask,
    flatten_rows,
    normalize_per_channel,
    robust_channel_stats,
)


def rebuild_image(record: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Rebuild the array that became a PNG, using the window recorded in the manifest."""
    pre = record["preprocessing"]
    path = Path(record["fits_path"])
    layout = astropy_reader.read_layout(path)

    start = int(round(pre["window_start_seconds"] / layout.tsamp_s))
    n_samples = pre["time_bins"] * pre["time_factor"]
    block = astropy_reader.read_block(
        path, start_sample=start, stop_sample=start + n_samples, layout=layout
    )

    median, sigma = robust_channel_stats(block.data)
    keep = channel_keep_mask(
        median=median,
        sigma=sigma,
        weights=block.channel_weights,
        zap_sigma=pre["zap_sigma"],
        edge_channels=pre["edge_channels"],
    )
    zscored = normalize_per_channel(block.data, median=median, sigma=sigma, keep=keep)
    averaged, freq_factor, trim_low = block_average(
        zscored, keep=keep, time_factor=pre["time_factor"], freq_bins=pre["freq_bins"]
    )
    averaged = flatten_rows(averaged)
    freqs = averaged_frequency_axis(
        block.frequencies_mhz,
        freq_bins=pre["freq_bins"],
        freq_factor=freq_factor,
        trim_low=trim_low,
    )
    return averaged, freqs, pre["time_factor"] * layout.tsamp_s


BOXCAR_WIDTHS = (1, 2, 4, 8, 16, 32)


def dedispersed_snr(
    image: np.ndarray,
    freqs: np.ndarray,
    bin_seconds: float,
    dm: float,
    *,
    min_coverage: float = 1.0,
) -> tuple[float, float]:
    """Peak of the dedispersed profile, in sigma, and the peak time (s in the window).

    The dedispersion is truncated, not circular: each channel contributes only
    where the shifted sample actually exists. An earlier version used `np.roll`
    with conditional edge masking, which produced two opposite artifacts
    depending on the DM -- at high DM the wrap entered the statistics, and at
    low DM the mask reduced the search to a sliver that did not contain the
    burst.

    `min_coverage` is the minimum fraction of channels that must contribute for
    a time bin to enter the search. The default of 1.0 restricts the search to
    the fully covered bins, where the noise is homogeneous and the robust
    statistic is well founded. Smaller values were tested to accommodate sweeps
    partially outside the file and discarded: the residual variance of the
    partial region is larger than that of the full region, so any single scale
    inflates the peaks there (positives and negatives both rose 40-60%,
    indiscriminately). Sweeps that are not contained should be excluded from
    the analysis, not patched.

    The peak is searched with a boxcar matched filter: FAST-FREX bursts span
    0.34 to 78 ms against bins of ~2 ms, and without it the narrow ones are
    diluted away.
    """
    reference = float(freqs.max())
    delays = DISPERSION_CONSTANT_S_MHZ2 * dm * (freqs.astype(float) ** -2 - reference**-2)
    shifts = np.round(delays / bin_seconds).astype(int)
    shifts = np.clip(shifts, 0, None)

    n_bins = image.shape[1]
    live = np.flatnonzero(np.any(image != 0.0, axis=1))
    if live.size == 0:
        return 0.0, 0.0

    # The search window is limited by the largest delay among the LIVE channels.
    # An earlier version used shifts.max() over all channels, including those
    # masked by the zap: since the zap nulls precisely the lowest frequencies,
    # which have the largest delay, channels contributing nothing were shortening
    # the search (362 of 470 available bins in a typical case) and degrading the
    # MAD estimate, which is what sets the sigma scale.
    total = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.float64)
    for row in live:
        start = int(shifts[row])
        if start >= n_bins:
            continue
        usable = n_bins - start
        total[:usable] += image[row, start:]
        count[:usable] += 1.0

    # Since the channels are in z-score units, summing n channels gives variance
    # n; dividing by sqrt(n) restores unit noise whatever the coverage, which
    # makes the profile comparable over time even with varying coverage.
    # count is non-increasing in t, so the valid region is the prefix [0, limit).
    limit = int(np.count_nonzero(count >= min_coverage * live.size))
    if limit < 8:
        return 0.0, 0.0
    profile = total[:limit] / np.sqrt(count[:limit])

    best_sigma = 0.0
    best_time = 0.0
    for width in BOXCAR_WIDTHS:
        if width > limit // 4:
            break
        if width == 1:
            smoothed = profile
        else:
            smoothed = np.convolve(profile, np.ones(width) / np.sqrt(width), mode="valid")
        # The noise scale is re-estimated at each width, rather than assuming it
        # falls as sqrt(w). The residual noise of these spectra is correlated in
        # time, so white scaling underestimates it and inflates the peak: in a
        # direct comparison, faint samples and negatives rose 40-60% under the
        # white version, which is artifact, not signal.
        med = float(np.median(smoothed))
        mad = float(np.median(np.abs(smoothed - med)))
        scale = 1.4826 * mad
        if scale <= 0:
            continue
        z = (smoothed - med) / scale
        peak = int(np.argmax(z))
        if float(z[peak]) > best_sigma:
            best_sigma = float(z[peak])
            best_time = (peak + width / 2.0) * bin_seconds
    return best_sigma, best_time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--negative-dm",
        choices=("blind", "matched", "both"),
        default="blind",
        help=(
            "How to measure the peak of the negatives. 'blind' scans 100..675 pc/cm3 "
            "and keeps the maximum, which answers 'would a blind search find "
            "anything?'. 'matched' assigns each negative a DM drawn from the "
            "distribution of the positives, so that both classes are measured with the "
            "SAME number of trials -- necessary to compare the distributions, since the "
            "maximum over 24 DMs is biased upwards relative to a single measurement. "
            "'both' computes the two in a single pass over the FITS and writes both "
            "columns."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the DM draw used by --negative-dm matched.")
    parser.add_argument(
        "--matched-dm-map",
        type=Path,
        help=(
            "JSON {sample_id: dm} holding a draw made in advance. The sequential draw "
            "depends on the whole record list, so running the manifest in parallel "
            "chunks would assign different DMs; precompute the map from the complete "
            "manifest (--dump-matched-dm) and pass the same file to every chunk."
        ),
    )
    parser.add_argument(
        "--dump-matched-dm",
        type=Path,
        help="Only build the matched-DM map from this manifest and exit, measuring nothing.",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]

    matched_dm: dict[str, float] = {}
    if args.matched_dm_map is not None:
        matched_dm = {k: float(v) for k, v in json.loads(args.matched_dm_map.read_text()).items()}
    elif args.negative_dm in ("matched", "both") or args.dump_matched_dm is not None:
        catalogued = [
            float(r["real_parameters"]["dm"])
            for r in records
            if r["real_parameters"].get("dm") is not None
        ]
        rng = np.random.default_rng(args.seed)
        for record in records:
            if record["real_parameters"].get("dm") is None:
                matched_dm[record["sample_id"]] = float(rng.choice(catalogued))

    if args.dump_matched_dm is not None:
        args.dump_matched_dm.parent.mkdir(parents=True, exist_ok=True)
        args.dump_matched_dm.write_text(json.dumps(matched_dm, indent=1))
        print(f"Mapa de DM casado ({len(matched_dm)} negativos) gravado em {args.dump_matched_dm}")
        return 0

    rows = []
    for i, record in enumerate(records, start=1):
        catalogued_dm = record["real_parameters"].get("dm")
        image, freqs, bin_seconds = rebuild_image(record)

        blind = matched = None
        if catalogued_dm is not None:
            # Positive: a single DM, the catalogued one. The two columns coincide,
            # since there is nothing to draw and nothing to scan.
            snr, t_peak = dedispersed_snr(image, freqs, bin_seconds, float(catalogued_dm))
            blind = matched = (snr, t_peak)
            dm_used = catalogued_dm
        else:
            dm_used = matched_dm.get(record["sample_id"], "")
            if args.negative_dm in ("blind", "both"):
                # Scan a DM grid and keep the best: "would a blind search find anything?".
                blind = max(
                    (
                        dedispersed_snr(image, freqs, bin_seconds, float(trial))
                        for trial in range(100, 700, 25)
                    ),
                    key=lambda item: item[0],
                )
            if args.negative_dm in ("matched", "both"):
                # Same number of trials as a positive: a single DM, 6 boxcars.
                matched = dedispersed_snr(image, freqs, bin_seconds, float(dm_used))
            snr, t_peak = blind if args.negative_dm != "matched" else matched

        row = {
            "sample_id": record["sample_id"],
            "source_sample_id": record["source_sample_id"],
            "true_label": record["true_label"],
            "source": record["real_parameters"].get("source") or "",
            "dm": dm_used if dm_used != "" else "",
            "peak_sigma": round(snr, 2),
            "peak_time_s": round(t_peak, 4),
        }
        if args.negative_dm == "both":
            # peak_sigma keeps the original (blind) semantics; the extra column is the
            # symmetric one.
            row["peak_sigma_matched"] = round(matched[0], 2)
            row["peak_time_matched_s"] = round(matched[1], 4)
        rows.append(row)
        print(f"[{i}/{len(records)}] {row['source_sample_id']}: {row['peak_sigma']} sigma", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
