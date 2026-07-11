from __future__ import annotations

import json
import re
from typing import Any


TASK_MULTICLASS = "multiclass"
TASK_FRB_BINARY = "frb-binary"
VALID_TASKS = (TASK_MULTICLASS, TASK_FRB_BINARY)
MULTICLASS_LABELS = ("FRB", "RFI", "NOISE")
BINARY_LABELS = ("FRB", "NON_FRB")

LABEL_ALIASES = {
    "FRB": "FRB",
    "FAST RADIO BURST": "FRB",
    "FAST-RADIO-BURST": "FRB",
    "RFI": "RFI",
    "RADIO FREQUENCY INTERFERENCE": "RFI",
    "RADIO-FREQUENCY INTERFERENCE": "RFI",
    "NOISE": "NOISE",
    "SYSTEM NOISE": "NOISE",
    "BACKGROUND NOISE": "NOISE",
}

BINARY_LABEL_ALIASES = {
    "FRB": "FRB",
    "FAST RADIO BURST": "FRB",
    "FAST-RADIO-BURST": "FRB",
    "NON FRB": "NON_FRB",
    "NON-FRB": "NON_FRB",
    "NON_FRB": "NON_FRB",
    "NOT FRB": "NON_FRB",
    "NOT-FRB": "NON_FRB",
    "NOT_FRB": "NON_FRB",
    "NOT A FRB": "NON_FRB",
    "NO FRB": "NON_FRB",
    "RFI": "NON_FRB",
    "RADIO FREQUENCY INTERFERENCE": "NON_FRB",
    "RADIO-FREQUENCY INTERFERENCE": "NON_FRB",
    "NOISE": "NON_FRB",
    "SYSTEM NOISE": "NON_FRB",
    "BACKGROUND NOISE": "NON_FRB",
    "ARTIFACT": "NON_FRB",
    "ARTEFACT": "NON_FRB",
}


def classes_for_task(task: str = TASK_MULTICLASS) -> tuple[str, ...]:
    if task == TASK_MULTICLASS:
        return MULTICLASS_LABELS
    if task == TASK_FRB_BINARY:
        return BINARY_LABELS
    raise ValueError(f"Invalid task: {task!r}")


def normalize_task(task: str | None) -> str:
    normalized = (task or TASK_MULTICLASS).strip().lower()
    if normalized in {"binary", "frb_binary", "frb-binary", "frb-vs-non-frb"}:
        return TASK_FRB_BINARY
    if normalized in {"multi", "multiclass", "three-class", "3-class"}:
        return TASK_MULTICLASS
    raise ValueError(f"Invalid task: {task!r}")


def normalize_label(value: Any, *, task: str = TASK_MULTICLASS) -> str:
    task = normalize_task(task)
    if value is None:
        raise ValueError("Response has no label.")

    label = str(value).strip().upper()
    label = re.sub(r"[_\s]+", " ", label)
    label = label.strip('"\'` .,:;')

    aliases = LABEL_ALIASES if task == TASK_MULTICLASS else BINARY_LABEL_ALIASES
    if label in aliases:
        return aliases[label]

    if task == TASK_FRB_BINARY:
        if "NON" in label and ("FRB" in label or "FAST RADIO BURST" in label):
            return "NON_FRB"
        if "NOT" in label and ("FRB" in label or "FAST RADIO BURST" in label):
            return "NON_FRB"
        if "RFI" in label or "INTERFERENCE" in label:
            return "NON_FRB"
        if "NOISE" in label or "ARTIFACT" in label or "ARTEFACT" in label:
            return "NON_FRB"
        if "FRB" in label or "FAST RADIO BURST" in label:
            return "FRB"
        raise ValueError(f"Invalid label: {value!r}")

    if "FRB" in label or "FAST RADIO BURST" in label:
        return "FRB"
    if "RFI" in label or "INTERFERENCE" in label:
        return "RFI"
    if "NOISE" in label:
        return "NOISE"
    raise ValueError(f"Invalid label: {value!r}")


def map_label_for_task(value: Any, *, task: str = TASK_MULTICLASS) -> str:
    task = normalize_task(task)
    if task == TASK_MULTICLASS:
        return normalize_label(value, task=task)

    label = normalize_label(value, task=TASK_MULTICLASS)
    return "FRB" if label == "FRB" else "NON_FRB"


def parse_model_response(
    raw_response: str,
    *,
    task: str = TASK_MULTICLASS,
    binary_threshold: float = 0.5,
) -> dict[str, Any]:
    task = normalize_task(task)
    json_text = _extract_json_object(raw_response)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object.")

    label = normalize_label(parsed.get("label"), task=task)
    confidence = _parse_confidence(parsed.get("confidence"))
    reason = str(parsed.get("reason", "")).strip()

    response = {
        "label": label,
        "confidence": confidence,
        "reason": reason,
    }

    if task == TASK_FRB_BINARY:
        frb_probability = _parse_required_probability(
            parsed.get("frb_probability"),
            field_name="frb_probability",
        )
        derived_label = derive_binary_label_from_probability(
            frb_probability,
            threshold=binary_threshold,
        )
        response["model_self_label"] = label
        response["label"] = derived_label
        response["frb_probability"] = frb_probability
        response["decision_threshold"] = binary_threshold
        response["label_probability_consistent"] = label == derived_label
        if label != derived_label:
            response["content_warning"] = (
                "Self-reported label diverges from frb_probability at threshold "
                f"{binary_threshold:g}: model_self_label={label}, "
                f"derived_label={derived_label}."
            )
        features = parsed.get("features")
        if isinstance(features, dict):
            response["features"] = features

    return response


def derive_binary_label_from_probability(
    frb_probability: Any,
    *,
    threshold: float = 0.5,
) -> str:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    probability = _parse_required_probability(
        frb_probability,
        field_name="frb_probability",
    )
    return "FRB" if probability >= threshold else "NON_FRB"


def _parse_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        value = value.strip().rstrip("%")
        confidence = float(value)
        if confidence > 1.0:
            confidence /= 100.0
    else:
        confidence = float(value)
    return min(1.0, max(0.0, confidence))


def _parse_required_probability(value: Any, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"Response has no {field_name}.")
    probability = _parse_confidence(value)
    return probability


def _extract_json_object(text: str) -> str:
    text = str(text).strip()
    if not text:
        raise ValueError("Empty model response.")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    if start == -1:
        raise ValueError("Response has no JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Incomplete JSON object in the response.")
