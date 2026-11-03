"""Injecao de pulsos dispersos em fundos reais do FAST-FREX (E2 e E5; plano, secoes 10.4 e 10.7).

O pulso entra no array em z-score por canal, depois de ``normalize_per_channel`` e antes de
``block_average``, com as estatisticas e a mascara de canais do proprio fundo. Dai em diante
o caminho e o P0 do artigo, com as mesmas funcoes de ``prepare_real``: sem pulso,
``process_window`` + ``render`` reproduzem os PNGs do artigo byte a byte (``e2_injection
check-control``).

A amplitude e fixada pelo S/N esperado, sem ruido, no diagnostico dos bursts reais
(``check_burst_visibility.dedispersed_snr``): o pulso de amplitude 1 passa pelo mesmo
``block_average`` e pela escala por linha do ``flatten_rows`` do fundo, e o sinal no perfil
dedispersado e dividido pela escala MAD do perfil do fundo, boxcar a boxcar. As tabelas usam
o sigma medido depois da injecao; o alvo so serve para cobrir a faixa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import exponnorm

from check_burst_visibility import BOXCAR_WIDTHS
from plot_dataset.plotter import DEFAULT_IMAGE_DPI, DynamicSpectrum, render_spectrum_png
from prepare_real.preprocess import (
    DISPERSION_CONSTANT_S_MHZ2,
    averaged_frequency_axis,
    block_average,
    channel_keep_mask,
    flatten_rows,
    normalize_per_channel,
    robust_channel_stats,
)


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
SCATTERING_REFERENCE_MHZ = 1250.0


@dataclass(frozen=True)
class P0:
    """O pre-processamento e a renderizacao do artigo (manifesto de dataset_real_full)."""

    time_bins: int = 1024
    time_factor: int = 20
    freq_bins: int = 256
    zap_sigma: float = 3.0
    edge_channels: int = 410
    cmap: str = "viridis"
    normalization: str = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    width: int = 1024
    height: int = 768
    dpi: int = DEFAULT_IMAGE_DPI

    @property
    def window_samples(self) -> int:
        return self.time_bins * self.time_factor


@dataclass(frozen=True)
class Pulse:
    t_top_s: float  # chegada no canal mais alto, em s desde o inicio da janela
    dm: float  # pc cm^-3
    fwhm_s: float  # largura intrinseca (FWHM da gaussiana)
    band_center_mhz: float | None = None  # None: banda larga
    band_fwhm_mhz: float | None = None
    tau_1250_s: float = 0.0  # espalhamento em 1,25 GHz, escala com nu^-4

    def sweep_s(self, f_low_mhz: float, f_high_mhz: float) -> tuple[float, float]:
        delay = DISPERSION_CONSTANT_S_MHZ2 * self.dm * (f_low_mhz**-2 - f_high_mhz**-2)
        return self.t_top_s, self.t_top_s + delay


def pulse_unit(
    pulse: Pulse, frequencies_mhz: np.ndarray, n_samples: int, tsamp_s: float
) -> np.ndarray:
    """Pulso de amplitude 1 (pico da gaussiana intrinseca) em z-score por canal.

    Em cada canal: gaussiana com a FWHM intrinseca somada em quadratura ao alargamento
    intracanal (2 k DM dnu / nu^3), convoluida com exp(-t/tau), tau = tau_1250 (nu/1250)^-4,
    com a fluencia conservada; atraso k DM (nu^-2 - nu_topo^-2); envelope gaussiano em
    frequencia quando a banda e limitada. Cada amostra e avaliada no seu centro.
    """
    freqs = np.asarray(frequencies_mhz, dtype=float)
    top = float(freqs.max())
    channel_width = float(np.median(np.abs(np.diff(freqs))))
    delays = DISPERSION_CONSTANT_S_MHZ2 * pulse.dm * (freqs**-2 - top**-2)
    smear = 2.0 * DISPERSION_CONSTANT_S_MHZ2 * pulse.dm * channel_width / freqs**3
    sigma_t = np.sqrt(pulse.fwhm_s**2 + smear**2) / FWHM_PER_SIGMA
    tau = pulse.tau_1250_s * (freqs / SCATTERING_REFERENCE_MHZ) ** -4
    fluence = pulse.fwhm_s / FWHM_PER_SIGMA * np.sqrt(2.0 * np.pi)
    if pulse.band_center_mhz is None:
        envelope = np.ones_like(freqs)
    else:
        envelope = np.exp(
            -4.0 * np.log(2.0) * ((freqs - pulse.band_center_mhz) / pulse.band_fwhm_mhz) ** 2
        )

    out = np.zeros((freqs.size, n_samples), dtype=np.float32)
    for row in range(freqs.size):
        if envelope[row] < 1e-6:
            continue
        center = pulse.t_top_s + delays[row]
        low = max(0, int(np.floor((center - 8.0 * sigma_t[row]) / tsamp_s)))
        high = min(n_samples, int(np.ceil((center + 8.0 * sigma_t[row] + 15.0 * tau[row]) / tsamp_s)) + 1)
        if high <= low:
            continue
        t = (np.arange(low, high) + 0.5) * tsamp_s
        if tau[row] > 1e-3 * sigma_t[row]:
            shape = exponnorm.pdf(t, tau[row] / sigma_t[row], loc=center, scale=sigma_t[row])
        else:
            shape = np.exp(-0.5 * ((t - center) / sigma_t[row]) ** 2) / (
                sigma_t[row] * np.sqrt(2.0 * np.pi)
            )
        out[row, low:high] = fluence * envelope[row] * shape
    return out


@dataclass(frozen=True)
class Window:
    """Uma janela pelo P0: o array que vira PNG e o que a calibracao precisa."""

    image: np.ndarray  # decimado e aplainado (o que o PNG mostra)
    averaged: np.ndarray  # decimado, antes do flatten_rows
    frequencies_mhz: np.ndarray  # eixo do array decimado
    bin_seconds: float
    keep: np.ndarray  # canais mantidos no array cru
    channels_zapped: int


def zscore(block, p0: P0) -> tuple[np.ndarray, np.ndarray]:
    median, sigma = robust_channel_stats(block.data)
    keep = channel_keep_mask(
        median=median,
        sigma=sigma,
        weights=block.channel_weights,
        zap_sigma=p0.zap_sigma,
        edge_channels=p0.edge_channels,
    )
    return normalize_per_channel(block.data, median=median, sigma=sigma, keep=keep), keep


def process_window(
    zscored: np.ndarray,
    keep: np.ndarray,
    frequencies_mhz: np.ndarray,
    tsamp_s: float,
    p0: P0,
    *,
    injection: np.ndarray | None = None,
) -> Window:
    """O P0 a partir do z-score. ``injection`` (ja escalado) e somado nos canais mantidos."""
    if injection is not None:
        zscored = zscored + injection * keep[:, None].astype(np.float32)
    averaged, freq_factor, trim_low = block_average(
        zscored, keep=keep, time_factor=p0.time_factor, freq_bins=p0.freq_bins
    )
    freqs = averaged_frequency_axis(
        frequencies_mhz, freq_bins=p0.freq_bins, freq_factor=freq_factor, trim_low=trim_low
    )
    return Window(
        image=flatten_rows(averaged),
        averaged=averaged,
        frequencies_mhz=freqs,
        bin_seconds=p0.time_factor * tsamp_s,
        keep=keep,
        channels_zapped=int(np.count_nonzero(~keep)),
    )


def render(window: Window, image_path: Path, p0: P0) -> None:
    spectrum = DynamicSpectrum(
        data=window.image,
        time_axis=np.arange(window.image.shape[1], dtype=float) * window.bin_seconds,
        frequency_axis=window.frequencies_mhz,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    render_spectrum_png(
        spectrum,
        image_path=image_path,
        cmap=p0.cmap,
        normalization=p0.normalization,
        percentile_low=p0.percentile_low,
        percentile_high=p0.percentile_high,
        width=p0.width,
        height=p0.height,
        dpi=p0.dpi,
        title=None,
    )


def dedispersed_profile(
    image: np.ndarray,
    freqs: np.ndarray,
    bin_seconds: float,
    dm: float,
    *,
    live: np.ndarray | None = None,
) -> np.ndarray:
    """O perfil de ``dedispersed_snr`` (dedispersao truncada, cobertura total, soma/sqrt(n)).

    ``live`` fixa os canais que contribuem; por padrao, as linhas nao nulas da imagem."""
    reference = float(freqs.max())
    delays = DISPERSION_CONSTANT_S_MHZ2 * dm * (freqs.astype(float) ** -2 - reference**-2)
    shifts = np.clip(np.round(delays / bin_seconds).astype(int), 0, None)
    n_bins = image.shape[1]
    if live is None:
        live = np.flatnonzero(np.any(image != 0.0, axis=1))
    total = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.float64)
    for row in live:
        start = int(shifts[row])
        if start >= n_bins:
            continue
        usable = n_bins - start
        total[:usable] += image[row, start:]
        count[:usable] += 1.0
    limit = int(np.count_nonzero(count >= live.size))
    return total[:limit] / np.sqrt(count[:limit])


def expected_sigma_per_unit(
    background: Window, pulse_averaged: np.ndarray, dm: float
) -> float:
    """S/N esperado, sem ruido, do pulso de amplitude 1 no diagnostico dos bursts reais."""
    _, row_sigma = robust_channel_stats(background.averaged, clip_sigma=5.0)
    populated = row_sigma > 0
    pulse_image = np.zeros_like(pulse_averaged, dtype=np.float32)
    pulse_image[populated] = pulse_averaged[populated] / row_sigma[populated, None]
    live = np.flatnonzero(np.any(background.image != 0.0, axis=1))
    noise = dedispersed_profile(
        background.image, background.frequencies_mhz, background.bin_seconds, dm, live=live
    )
    signal = dedispersed_profile(
        pulse_image, background.frequencies_mhz, background.bin_seconds, dm, live=live
    )
    best = 0.0
    for width in BOXCAR_WIDTHS:
        if width > noise.size // 4:
            break
        kernel = np.ones(width) / np.sqrt(width)
        smoothed_noise = np.convolve(noise, kernel, mode="valid")
        smoothed_signal = np.convolve(signal, kernel, mode="valid")
        scale = 1.4826 * float(np.median(np.abs(smoothed_noise - np.median(smoothed_noise))))
        if scale > 0:
            best = max(best, float(smoothed_signal.max()) / scale)
    return best
