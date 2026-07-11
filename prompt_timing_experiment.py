from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vlm_classifier.prompt import BINARY_FRB_PROMPT, CLASSIFICATION_PROMPT

LEAN_BINARY_PROMPT = """You are an expert radio astronomer classifying dynamic-spectrum images from a transient-search pipeline.

Image meaning:
- x-axis: time
- y-axis: frequency
- intensity: signal power

Classify the image as exactly one of:
- FRB
- NON_FRB

FRB: the dominant signal is a SINGLE localized transient, plausibly broadband, with a frequency-dependent arrival-time trend, usually later arrival at lower frequency. The sweep may be curved, nearly linear, faint, or partially masked. Do not treat faint coherent structure as pure noise: if there is any plausible faint dispersed sweep or localized broadband brightening, use an intermediate probability instead of a very low one.

NON_FRB: pure random background; persistent horizontal or vertical bands; periodic stripes or grid-like structure; zero-DM broadband vertical impulses with no dispersion delay; saturation, clipping, blank regions, or normalization artifacts; two or more separated broadband pulses or repeating patterns.

frb_probability must be in [0,1]:
- 0.95-0.99 textbook broadband dispersed sweep
- 0.60-0.94 probable or plausible FRB
- 0.45-0.59 genuinely ambiguous
- 0.18-0.44 weak structure, more likely NON_FRB
- 0.01-0.17 clear RFI, noise, or artifact

label = "FRB" only if frb_probability >= 0.50, otherwise "NON_FRB".

Return ONLY valid JSON, no markdown, no comments, no extra keys, exactly this schema:
{"label": "FRB|NON_FRB", "frb_probability": 0.0}"""

LEAN_REASON_SCHEMA = """Describe the visual evidence first, then decide. Return ONLY valid JSON, no markdown, no comments, no extra keys, exactly this schema:
{"reason": "one short sentence citing the visual evidence", "frb_probability": 0.0, "label": "FRB|NON_FRB"}"""

LEAN_REASON_BINARY_PROMPT = (
    LEAN_BINARY_PROMPT.rsplit("Return ONLY valid JSON", 1)[0].rstrip()
    + "\n\n"
    + LEAN_REASON_SCHEMA
)

ARMS: dict[str, str] = {
    "binary_full": BINARY_FRB_PROMPT,
    "binary_lean": LEAN_BINARY_PROMPT,
    "binary_lean_reason": LEAN_REASON_BINARY_PROMPT,
    "multiclass": CLASSIFICATION_PROMPT,
}
BINARY_ARMS = ("binary_full", "binary_lean", "binary_lean_reason")


def parse_json_response(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def rfi_subtype(record: dict[str, Any]) -> str | None:
    return (record.get("simulation_parameters") or {}).get("rfi_category")


def choose_subsample(
    records: list[dict[str, Any]],
    *,
    n_frb: int,
    per_rfi_subtype: int,
    n_noise: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {r["sample_id"]: r for r in records}
    frb_pool = sorted(r["sample_id"] for r in records if r["true_label"] == "FRB")
    noise_pool = sorted(r["sample_id"] for r in records if r["true_label"] == "NOISE")
    rfi_pools: dict[str, list[str]] = {}
    for r in records:
        if r["true_label"] == "RFI":
            rfi_pools.setdefault(rfi_subtype(r) or "unknown", []).append(r["sample_id"])

    rng = random.Random(seed)
    chosen: list[str] = []
    chosen += rng.sample(frb_pool, n_frb)
    for subtype in sorted(rfi_pools):
        chosen += rng.sample(sorted(rfi_pools[subtype]), per_rfi_subtype)
    chosen += rng.sample(noise_pool, n_noise)

    chosen_set = set(chosen)
    warmup_id = next(sid for sid in frb_pool if sid not in chosen_set)
    ordered = [by_id[sid] for sid in sorted(chosen)]
    return ordered, by_id[warmup_id]


def build_model(args: argparse.Namespace):
    if args.dry_run:
        from vlm_classifier.models import DryRunVLM

        binary_stub = DryRunVLM(seed=args.seed, labels=("FRB", "NON_FRB"))
        multi_stub = DryRunVLM(seed=args.seed + 1, labels=("FRB", "RFI", "NOISE"))

        class _Router:
            def classify(self, *, image_path, prompt, sample_id):
                stub = multi_stub if prompt is CLASSIFICATION_PROMPT else binary_stub
                return stub.classify(
                    image_path=image_path, prompt=prompt, sample_id=sample_id
                )

        return _Router()

    from vlm_classifier.models import Gemma4VLM

    return Gemma4VLM(
        model_name=args.model,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        cache_implementation=args.cache_implementation,
    )


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_samples": len(rows),
        "prompt_chars": {arm: len(prompt) for arm, prompt in ARMS.items()},
        "arms": {},
    }
    for arm in ARMS:
        seconds = sorted(r[arm]["seconds"] for r in rows)
        chars = [len(r[arm]["raw_response"]) for r in rows]
        parse_failures = sum(1 for r in rows if r[arm]["parsed"] is None)
        arm_summary: dict[str, Any] = {
            "seconds_mean": sum(seconds) / len(seconds),
            "seconds_median": percentile(seconds, 0.5),
            "seconds_p10": percentile(seconds, 0.1),
            "seconds_p90": percentile(seconds, 0.9),
            "seconds_min": seconds[0],
            "seconds_max": seconds[-1],
            "response_chars_mean": sum(chars) / len(chars),
            "parse_failures": parse_failures,
        }
        if arm in BINARY_ARMS:
            correct = 0
            probs: Counter[float] = Counter()
            for r in rows:
                parsed = r[arm]["parsed"] or {}
                prob = parsed.get("frb_probability")
                pred = None
                if isinstance(prob, (int, float)):
                    probs[round(float(prob), 4)] += 1
                    pred = "FRB" if float(prob) >= 0.5 else "NON_FRB"
                truth = "FRB" if r["source_true_label"] == "FRB" else "NON_FRB"
                if pred == truth:
                    correct += 1
            arm_summary["binary_accuracy"] = correct / len(rows)
            arm_summary["distinct_probabilities"] = {
                str(k): v for k, v in sorted(probs.items())
            }
        else:
            correct3 = sum(
                1
                for r in rows
                if (r[arm]["parsed"] or {}).get("label") == r["source_true_label"]
            )
            correct_bin = 0
            for r in rows:
                pred = (r[arm]["parsed"] or {}).get("label")
                pred_bin = "FRB" if pred == "FRB" else "NON_FRB"
                truth = "FRB" if r["source_true_label"] == "FRB" else "NON_FRB"
                if pred_bin == truth:
                    correct_bin += 1
            arm_summary["multiclass_accuracy"] = correct3 / len(rows)
            arm_summary["binarized_accuracy"] = correct_bin / len(rows)
        summary["arms"][arm] = arm_summary

    agreements: dict[str, dict[str, Any]] = {}
    for arm in BINARY_ARMS:
        if arm == "binary_full":
            continue
        agree = 0
        both = 0
        for r in rows:
            labels = []
            for candidate in ("binary_full", arm):
                prob = (r[candidate]["parsed"] or {}).get("frb_probability")
                if isinstance(prob, (int, float)):
                    labels.append("FRB" if float(prob) >= 0.5 else "NON_FRB")
            if len(labels) == 2:
                both += 1
                if labels[0] == labels[1]:
                    agree += 1
        agreements[arm] = {
            "agreement_with_full": agree / both if both else None,
            "compared_samples": both,
        }
    summary["label_agreement_vs_full"] = agreements
    return summary


def write_report(summary: dict[str, Any], path: Path) -> None:
    lines = ["Prompt timing experiment report", ""]
    lines.append(f"samples: {summary['n_samples']}")
    for arm, chars in summary["prompt_chars"].items():
        lines.append(f"prompt chars [{arm}]: {chars}")
    lines.append("")
    for arm, s in summary["arms"].items():
        lines.append(
            f"{arm}: mean={s['seconds_mean']:.2f}s median={s['seconds_median']:.2f}s "
            f"p10={s['seconds_p10']:.2f}s p90={s['seconds_p90']:.2f}s "
            f"chars={s['response_chars_mean']:.0f} parse_failures={s['parse_failures']}"
        )
        if "binary_accuracy" in s:
            lines.append(f"  binary accuracy @0.5: {s['binary_accuracy']:.4f}")
        if "multiclass_accuracy" in s:
            lines.append(
                f"  multiclass accuracy: {s['multiclass_accuracy']:.4f} "
                f"(binarized: {s['binarized_accuracy']:.4f})"
            )
    lines.append("")
    for arm, info in summary["label_agreement_vs_full"].items():
        lines.append(
            f"{arm} vs binary_full label agreement: {info['agreement_with_full']:.4f} "
            f"over {info['compared_samples']} samples"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paired prompt-latency audit on the binary benchmark: four prompt "
            "variants (full binary, lean, lean+reason, multiclass) applied to "
            "the same stratified subsample, each inference timed individually."
        )
    )
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument(
        "--device-map",
        default="cuda",
        help="device_map for from_pretrained; single-device placement is recommended.",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--cache-implementation",
        default="none",
        help='cache_implementation for generate; "none" matches the controlled run in the paper.',
    )
    parser.add_argument(
        "--manifest", default="dataset_binary/metadata/image_manifest.jsonl"
    )
    parser.add_argument("--output-dir", default="results_prompt_timing_2b")
    parser.add_argument("--n-frb", type=int, default=100)
    parser.add_argument("--per-rfi-subtype", type=int, default=10)
    parser.add_argument("--n-noise", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = root / args.output_dir
    predictions_path = output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {predictions_path}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(root / args.manifest)
    samples, warmup = choose_subsample(
        records,
        n_frb=args.n_frb,
        per_rfi_subtype=args.per_rfi_subtype,
        n_noise=args.n_noise,
        seed=args.seed,
    )
    print(
        f"subsample: {len(samples)} images "
        f"({args.n_frb} FRB, 5x{args.per_rfi_subtype} RFI, {args.n_noise} NOISE); "
        f"warmup: {warmup['sample_id']}",
        flush=True,
    )

    model = build_model(args)

    for arm, prompt in ARMS.items():
        started = time.perf_counter()
        model.classify(
            image_path=root / warmup["image_path"],
            prompt=prompt,
            sample_id=f"warmup-{arm}",
        )
        print(f"warmup {arm}: {time.perf_counter() - started:.2f}s", flush=True)

    rows: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(samples, start=1):
            row: dict[str, Any] = {
                "sample_id": record["sample_id"],
                "source_true_label": record["true_label"],
                "rfi_subtype": rfi_subtype(record),
            }
            for arm, prompt in ARMS.items():
                started = time.perf_counter()
                raw = model.classify(
                    image_path=root / record["image_path"],
                    prompt=prompt,
                    sample_id=record["sample_id"],
                )
                elapsed = time.perf_counter() - started
                row[arm] = {
                    "seconds": elapsed,
                    "raw_response": raw,
                    "parsed": parse_json_response(raw),
                }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(samples):
                means = {
                    arm: sum(r[arm]["seconds"] for r in rows) / len(rows)
                    for arm in ARMS
                }
                pretty = " ".join(f"{arm}={m:.2f}s" for arm, m in means.items())
                print(f"[{index}/{len(samples)}] running means: {pretty}", flush=True)

    summary = summarize(rows)
    summary["model"] = args.model if not args.dry_run else "dry-run"
    summary["seed"] = args.seed
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(summary, output_dir / "report.txt")
    print(f"results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
