from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence


VALID_LABELS = ("FRB", "RFI", "NOISE")
DEFAULT_GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
# The README/logs use google/gemma-4-E2B-it for "2B" and
# google/gemma-4-E4B-it for "4B"; both are variants of the Gemma 4 family.
# The short alias keeps pointing to E4B for compatibility.
GEMMA4_ALIASES = {"gemma4", "gemma-4", "gemma4-e4b", "gemma4-e4b-it"}

LOGGER = logging.getLogger(__name__)


class BaseVLM(ABC):
    @abstractmethod
    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        """Return the raw model response as text."""


class DryRunVLM(BaseVLM):
    def __init__(self, *, seed: int, labels: Sequence[str] = VALID_LABELS) -> None:
        self._rng = random.Random(seed)
        if not labels:
            raise ValueError("labels must not be empty.")
        self.labels = tuple(labels)

    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        label = self._rng.choice(self.labels)
        confidence = round(self._rng.uniform(0.35, 0.95), 4)
        response = {
            "label": label,
            "confidence": confidence,
            "reason": "Dry-run synthetic prediction.",
        }
        if "NON_FRB" in self.labels:
            if label == "FRB":
                frb_probability = round(self._rng.uniform(0.51, 0.95), 4)
            else:
                frb_probability = round(self._rng.uniform(0.05, 0.49), 4)
            response["frb_probability"] = frb_probability
            response["confidence"] = round(max(frb_probability, 1.0 - frb_probability), 4)
        return json.dumps(response, sort_keys=True)


def resolve_gemma4_model_id(model_name: str | None) -> str:
    if model_name is None:
        return DEFAULT_GEMMA4_MODEL_ID

    candidate = model_name.strip()
    if candidate.lower() in GEMMA4_ALIASES:
        return DEFAULT_GEMMA4_MODEL_ID
    return candidate


def _none_if_disabled(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null", "off"}:
        return None
    return normalized


def build_generation_kwargs(
    *,
    max_new_tokens: int,
    cache_implementation: str | None,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if cache_implementation is not None:
        kwargs["cache_implementation"] = cache_implementation
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    return kwargs


class Gemma4VLM(BaseVLM):
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_GEMMA4_MODEL_ID,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        attn_implementation: str | None = "sdpa",
        cache_implementation: str | None = "static",
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        hf_token: str | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be > 0 when provided.")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0 when provided.")

        self.model_id = resolve_gemma4_model_id(model_name)
        self.device_map = device_map
        self.dtype = _none_if_disabled(dtype)
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = _none_if_disabled(attn_implementation)
        self.cache_implementation = _none_if_disabled(cache_implementation)
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.hf_token = hf_token
        self.processor: Any | None = None
        self.model: Any | None = None

        self.load()

    def load(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        try:
            from huggingface_hub import login
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Gemma 4 dependencies not found. "
                "Install the project with: uv pip install -e ."
            ) from exc

        token = self.hf_token or os.getenv("HF_TOKEN")
        if token:
            login(token=token)

        model_kwargs: dict[str, Any] = {"device_map": self.device_map}
        if self.dtype is not None:
            model_kwargs["dtype"] = self.dtype
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        LOGGER.info("Loading Gemma 4: %s", self.model_id)
        LOGGER.info(
            "Gemma 4 generation configuration: do_sample=%s, temperature=%s, "
            "top_p=%s, top_k=%s, seed=%s",
            self.do_sample,
            self.temperature if self.do_sample else None,
            self.top_p if self.do_sample else None,
            self.top_k if self.do_sample else None,
            self.seed,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self.model.eval()

    def classify(self, *, image_path: Path, prompt: str, sample_id: str) -> str:
        self.load()
        if self.processor is None or self.model is None:
            raise RuntimeError("Gemma 4 was not loaded correctly.")

        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Image inference dependencies not found. "
                "Install the project with: uv pip install -e ."
            ) from exc

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        target_device = getattr(self.model, "device", None)
        if target_device is not None:
            inputs = inputs.to(target_device)

        input_len = inputs["input_ids"].shape[-1]
        generate_kwargs = build_generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            cache_implementation=self.cache_implementation,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        with torch.inference_mode():
            if self.seed is not None:
                from transformers import set_seed

                set_seed(self.seed)
            outputs = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        LOGGER.debug("Raw Gemma 4 response for %s: %s", sample_id, response)
        return str(response).strip()
