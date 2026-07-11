from __future__ import annotations


CLASSIFICATION_PROMPT = """
You are an expert in radio astronomy analyzing candidate images from a transient-search pipeline.

The image is a dynamic spectrum: the horizontal axis is time, the vertical axis is frequency, and pixel intensity is signal power.

Classify the image into exactly one of three classes: FRB, RFI, or NOISE.
Decide only from the visual content of the image.

Class definitions
=================
NOISE
The entire panel is featureless, stochastic background texture. There is no coherent structure anywhere: no lines, no stripes, no bands, no localized bright spots, no sweep, nothing but random fluctuations.

FRB
A single astrophysical burst: one localized, broadband transient whose arrival time changes systematically (monotonically) with frequency, tracing a single continuous sweep across the band. It appears only once, is confined in time, and neither repeats nor sits at a fixed frequency.

RFI
Anything else. Any image that is not pure NOISE and is not a single dispersed FRB sweep is RFI, i.e. any coherent, persistent, repeated, or artificial-looking structure of terrestrial or instrumental origin.

Decision procedure
=================
1. Is there a single broadband sweep that drifts monotonically in frequency with time? If yes, label FRB.
2. Otherwise, is the panel pure random texture with no coherent feature at all? If yes, label NOISE.
3. Otherwise, label RFI.

Rules
=====
- NOISE requires a complete absence of coherent structure. If you can see ANY non-random feature anywhere in the panel -- even faint, even sparse, even a single thin line, a streak, or a few repeated marks -- the image is NOT NOISE.
- Faint or sparse structure still counts as structure.
- An FRB drifts in frequency and appears once. Signals that stay at a fixed frequency, form horizontal or vertical bands, or repeat in time are RFI, not FRB.
- When unsure between RFI and NOISE, choose RFI. When a single drifting broadband sweep is unsure between FRB and RFI, prefer FRB.

Output
======
Return ONLY a valid JSON object with this exact schema, and nothing else:
{
  "label": "FRB|RFI|NOISE",
  "confidence": 0.0,
  "reason": "one short sentence citing the visual evidence"
}
confidence is how certain you are about the chosen label.
Do not include markdown, code fences, comments, or any text outside the JSON object.
"""


BINARY_FRB_PROMPT = """
You are an expert radio astronomer classifying dynamic-spectrum images from a transient-search pipeline.

Image meaning:
- x-axis: time
- y-axis: frequency
- intensity: signal power

Classify the image as exactly one of:
- FRB
- NON_FRB

Return ONLY valid JSON. No markdown, no comments, no extra keys.

Decision rules:

1. FRB candidates
Label as FRB only when the dominant signal is a SINGLE localized transient that is plausibly broadband and shows a frequency-dependent arrival-time trend, usually later arrival at lower frequency. The sweep may be curved, nearly linear, faint, or partially masked.

2. Low-S/N sensitivity
Do not treat faint coherent structure as pure noise. If there is any plausible faint diagonal/curved sweep, localized broadband brightening, partial dispersed track, or excess variance along a tilted band, use an intermediate probability instead of a very low one.

3. NON_FRB cases
Label as NON_FRB when the image is dominated by:
- pure random background with no coherent transient,
- persistent horizontal frequency bands,
- persistent vertical time bands,
- periodic stripes or grid-like structure,
- zero-DM broadband vertical impulse with no dispersion delay,
- saturation, clipping, blank regions, uniform panels, or normalization artifacts,
- two or more temporally separated broadband pulses or repeating patterns.

4. Probability calibration
frb_probability must be in [0,1].
Use the full range and avoid always reusing the same values.

Calibration guide:
- 0.95-0.99: textbook FRB, clear broadband dispersed sweep.
- 0.80-0.94: probable FRB with minor ambiguity.
- 0.60-0.79: plausible FRB but faint, partial, or noisy.
- 0.45-0.59: genuinely ambiguous.
- 0.30-0.44: weak possible structure, more likely NON_FRB.
- 0.18-0.29: likely noise or weak contamination.
- 0.05-0.17: clear RFI, persistent bands, artifacts, or strong NON_FRB evidence.
- 0.01-0.04: blank, saturated, clipped, or unambiguous artifact.

Label consistency:
- label = "FRB" only if frb_probability >= 0.50
- label = "NON_FRB" only if frb_probability < 0.50

confidence is how certain you are about the discrete label, not the FRB probability.

Output exactly this JSON schema:

{
  "label": "FRB|NON_FRB",
  "frb_probability": 0.0,
  "confidence": 0.0,
  "reason": "short explanation referencing the visual evidence",
  "features": {
    "visible_structure": true,
    "broadband": true,
    "localized_transient": true,
    "frequency_dependent_delay": true,
    "persistent_bands": false,
    "repeating_pattern": false,
    "uniform_or_constant_background": false,
    "saturated_or_clipped": false,
    "random_background_texture": false
  }
}

Feature definitions:
- visible_structure: any coherent visual structure is present.
- broadband: signal spans a substantial frequency range.
- localized_transient: signal is confined to a short time interval.
- frequency_dependent_delay: arrival time changes systematically with frequency.
- persistent_bands: horizontal or vertical bands persist across much of the panel.
- repeating_pattern: multiple pulses, stripes, grids, or periodic structures.
- uniform_or_constant_background: panel is nearly blank or constant.
- saturated_or_clipped: image has saturation, clipping, or rectangular masked regions.
- random_background_texture: dominated by stochastic noise with no coherent candidate.

The reason must be one short sentence.
"""


def prompt_for_task(task: str) -> str:
    if task == "frb-binary":
        return BINARY_FRB_PROMPT
    return CLASSIFICATION_PROMPT
