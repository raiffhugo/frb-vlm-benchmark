from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Dispersion constant k_DM in s * MHz^2 * pc^-1 * cm^3.
DISPERSION_CONSTANT_S_MHZ2 = 4.148808e3

TOA_REFERENCES = ("top", "bottom", "infinite")


def dispersion_delay_s(
    dm: float,
    freq_mhz: float | np.ndarray,
    reference_freq_mhz: float,
) -> float | np.ndarray:
    freq = np.asarray(freq_mhz, dtype=float)
    delay = DISPERSION_CONSTANT_S_MHZ2 * dm * (freq**-2 - reference_freq_mhz**-2)
    if np.isscalar(freq_mhz):
        return float(delay)
    return delay


def burst_center_s(
    *,
    toa_s: float,
    dm: float,
    f_low_mhz: float,
    f_high_mhz: float,
    toa_reference: str = "top",
) -> float:
    if toa_reference not in TOA_REFERENCES:
        raise ValueError(f"invalid toa_reference: {toa_reference!r}")
    if f_low_mhz >= f_high_mhz:
        raise ValueError("f_low_mhz must be smaller than f_high_mhz.")

    span = float(dispersion_delay_s(dm, f_low_mhz, f_high_mhz))
    if toa_reference == "top":
        t_top = toa_s
    elif toa_reference == "bottom":
        t_top = toa_s - span
    else:
        t_top = toa_s + DISPERSION_CONSTANT_S_MHZ2 * dm * f_high_mhz**-2
    return t_top + span / 2.0


@dataclass(frozen=True)
class WindowPlan:
    start_sample: int
    stop_sample: int
    time_factor: int

    @property
    def n_samples(self) -> int:
        return self.stop_sample - self.start_sample


def plan_window(
    *,
    center_s: float,
    window_seconds: float,
    tsamp_s: float,
    nsamp_total: int,
    time_bins: int,
) -> WindowPlan:
    if time_bins <= 0:
        raise ValueError("time_bins must be positive.")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")

    factor = max(1, round(window_seconds / tsamp_s / time_bins))
    needed = factor * time_bins
    if needed > nsamp_total:
        factor = nsamp_total // time_bins
        if factor < 1:
            raise ValueError(
                f"A file with {nsamp_total} samples is too short for "
                f"{time_bins} time bins."
            )
        needed = factor * time_bins

    center_sample = int(round(center_s / tsamp_s))
    start = center_sample - needed // 2
    start = min(max(start, 0), nsamp_total - needed)
    return WindowPlan(start_sample=start, stop_sample=start + needed, time_factor=factor)


def robust_channel_stats(
    data: np.ndarray,
    *,
    clip_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float32)
    median = np.median(data, axis=1).astype(np.float32)
    deviations = np.abs(data - median[:, None])
    mad_sigma = 1.4826 * np.median(deviations, axis=1).astype(np.float32)

    # Final sigma from a standard deviation clipped at clip_sigma MADs: on
    # data quantized to few bits the raw MAD is locked to integer ADU steps and
    # under/overestimates the noise in bands of channels; the clipped standard
    # deviation stays robust to bursts/RFI and varies continuously.
    limit = np.where(mad_sigma > 0, clip_sigma * mad_sigma, np.inf)[:, None]
    within = deviations <= limit
    counts = within.sum(axis=1)
    sums = np.where(within, data, 0.0).sum(axis=1)
    means = sums / np.maximum(counts, 1)
    squares = np.where(within, (data - means[:, None].astype(np.float32)) ** 2, 0.0).sum(axis=1)
    variance = squares / np.maximum(counts - 1, 1)
    sigma = np.sqrt(variance, dtype=np.float32)
    sigma[counts < 2] = 0.0
    return median, sigma.astype(np.float32)


def _robust_zscores(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - med) / scale).astype(np.float32)


def channel_keep_mask(
    *,
    median: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray | None = None,
    zap_sigma: float = 5.0,
    edge_channels: int = 0,
) -> np.ndarray:
    nchan = sigma.size
    keep = np.ones(nchan, dtype=bool)

    if weights is not None:
        keep &= np.asarray(weights) > 0

    tiny = np.finfo(np.float32).tiny
    keep &= sigma > tiny

    if edge_channels > 0:
        keep[:edge_channels] = False
        keep[nchan - edge_channels :] = False

    valid = keep.copy()
    if zap_sigma > 0 and valid.any():
        sigma_z = _robust_zscores(sigma[valid])
        median_z = _robust_zscores(median[valid])
        keep[valid] &= (sigma_z <= zap_sigma) & (median_z <= zap_sigma)

    return keep


def normalize_per_channel(
    data: np.ndarray,
    *,
    median: np.ndarray,
    sigma: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    safe_sigma = np.where(sigma > 0, sigma, 1.0)
    zscored = (data - median[:, None]) / safe_sigma[:, None]
    zscored[~keep, :] = 0.0
    return zscored.astype(np.float32)


def block_average(
    zscored: np.ndarray,
    *,
    keep: np.ndarray,
    time_factor: int,
    freq_bins: int,
) -> tuple[np.ndarray, int, int]:
    nchan, nsamp = zscored.shape
    if nsamp % time_factor != 0:
        raise ValueError("nsamp must be a multiple of time_factor.")
    freq_factor = max(1, nchan // freq_bins)
    used = freq_factor * freq_bins
    trim_low = (nchan - used) // 2

    window = zscored[trim_low : trim_low + used]
    weights = keep[trim_low : trim_low + used].astype(np.float32)

    time_avg = window.reshape(used, nsamp // time_factor, time_factor).mean(axis=2)
    numerator = (time_avg * weights[:, None]).reshape(
        freq_bins, freq_factor, nsamp // time_factor
    ).sum(axis=1)
    denominator = weights.reshape(freq_bins, freq_factor).sum(axis=1)

    averaged = np.zeros_like(numerator, dtype=np.float32)
    populated = denominator > 0
    averaged[populated] = numerator[populated] / denominator[populated, None]

    # Equalize the per-row variance: the mean of time_factor samples times
    # n surviving channels has a deviation of 1/sqrt(time_factor * n); rescaling
    # by sqrt(time_factor * n) returns pixels in z-score units and prevents rows
    # with many masked channels from showing a brighter texture.
    averaged[populated] *= np.sqrt(
        time_factor * denominator[populated, None], dtype=np.float32
    )
    return averaged, freq_factor, trim_low


def flatten_rows(averaged: np.ndarray, *, clip_sigma: float = 5.0) -> np.ndarray:
    """Renormalize each row of the decimated image (median 0, deviation 1).

    Per-channel normalization on the raw data does not remove variance that is
    correlated between neighbouring channels (polyphase filterbank, broadband
    RFI); after averaging in frequency those bands end up with a brighter or
    dimmer texture. This second pass, applied at the final resolution, flattens
    any residual per-row structure and erases persistent RFI lines while
    preserving time-localized transients (the per-row statistics are robust).
    """
    median, sigma = robust_channel_stats(averaged, clip_sigma=clip_sigma)
    flattened = averaged - median[:, None]
    populated = sigma > 0
    flattened[populated] /= sigma[populated, None]
    flattened[~populated] = 0.0
    return flattened.astype(np.float32)


def averaged_frequency_axis(
    frequencies_mhz: np.ndarray,
    *,
    freq_bins: int,
    freq_factor: int,
    trim_low: int,
) -> np.ndarray:
    used = freq_factor * freq_bins
    window = np.asarray(frequencies_mhz, dtype=float)[trim_low : trim_low + used]
    return window.reshape(freq_bins, freq_factor).mean(axis=1)
