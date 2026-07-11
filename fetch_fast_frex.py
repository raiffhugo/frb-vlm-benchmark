#!/usr/bin/env python3
"""Select and download a subset of FAST-FREX from the Science Data Bank.

This script is NOT part of the pipeline: it only (a) builds the public file
index of the dataset, (b) draws a stratified sample of positives and negatives,
(c) downloads the FITS with MD5 verification and (d) writes the input CSV
manifest consumed by `run_pipeline.py prepare-real`.

The index comes from the JSON-LD (Croissant) block embedded in the public page
of the dataset, which lists name, size, MD5 and download URL for the 1600 files.
The downloads themselves are anonymous and support Range, so they are resumable.

Typical use:

    uv run python fetch_fast_frex.py \
      --n-frb 50 --n-negative 50 --seed 42 \
      --out-manifest fast_frex/real_manifest_n100.csv \
      --workers 6
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

DATASET_ID = "3b3cf2f75a74419b89a56cc9626af2a0"
DETAIL_URL = f"https://www.scidb.cn/en/detail?dataSetId={DATASET_ID}"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

SOURCES = ("FRB20121102", "FRB20180301", "FRB20201124")

# Files already used during development (representation/prompt tuning) and in
# the holdout check; they are excluded from any new sample.
ALREADY_USED = frozenset(
    {
        "FRB20121102_0001.fits",
        "FRB20121102_0038.fits",
        "FRB20180301_0001.fits",
        "FRB20180301_0004.fits",
        "FRB20201124_0001.fits",
        "FRB20201124_0016.fits",
        "00001_neg_sample.fits",
        "00500_neg_sample.fits",
        "01000_neg_sample.fits",
    }
)


def build_index(cache_path: Path, *, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Return {file_name: {url, md5, size}} for the 1603 files."""
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    print(f"Downloading the file index from {DETAIL_URL} ...", flush=True)
    result = subprocess.run(
        ["curl", "-sS", "-L", "-H", f"User-Agent: {USER_AGENT}", DETAIL_URL],
        capture_output=True,
        check=True,
    )
    html = result.stdout.decode("utf-8", errors="replace")

    marker = '"distribution":['
    position = html.find(marker)
    if position < 0:
        raise RuntimeError("The 'distribution' block was not found on the dataset page.")

    start = html.index("[", position)
    depth = 0
    end = start
    for offset in range(start, len(html)):
        if html[offset] == "[":
            depth += 1
        elif html[offset] == "]":
            depth -= 1
            if depth == 0:
                end = offset
                break
    payload = html[start : end + 1]
    payload = payload.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&")

    index = {
        entry["name"]: {
            "url": entry["contentUrl"],
            "md5": entry["md5"],
            "size": int(str(entry["contentSize"]).split()[0]),
        }
        for entry in json.loads(payload)
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    print(f"Index with {len(index)} files written to {cache_path}", flush=True)
    return index


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("file")]


def select_sample(
    *,
    summary_dir: Path,
    index: dict[str, dict[str, Any]],
    n_frb: int,
    n_negative: int,
    seed: int,
    exclude_used: bool,
    min_per_source: int,
) -> tuple[list[dict[str, str]], list[str]]:
    """Draw positives proportionally to the catalogue and negatives uniformly.

    Strict proportionality would zero out FRB20180301 (3 bursts available
    against 468 and 123 from the other sources); `min_per_source` guarantees a
    minimum presence of each repeater, taking the excess from the most numerous
    source.
    """
    excluded = ALREADY_USED if exclude_used else frozenset()

    catalog: dict[str, list[dict[str, str]]] = {}
    for source in SOURCES:
        rows = read_summary(summary_dir / f"{source}_summary.csv")
        catalog[source] = [row for row in rows if row["file"] not in excluded]

    population = {source: len(rows) for source in SOURCES for rows in [catalog[source]]}
    total = sum(population.values())

    # Proportional allocation with largest remainders, respecting each source's stock.
    exact = {source: n_frb * population[source] / total for source in SOURCES}
    quota = {source: int(np.floor(value)) for source, value in exact.items()}
    remainder = n_frb - sum(quota.values())
    for source, _ in sorted(exact.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True):
        if remainder <= 0:
            break
        if quota[source] < population[source]:
            quota[source] += 1
            remainder -= 1
    for source in SOURCES:
        quota[source] = min(quota[source], population[source])

    # Per-source floor: the deficit is taken from the source with the largest
    # quota, so as not to change the order of magnitude of the proportional split.
    for source in SOURCES:
        floor = min(min_per_source, population[source])
        while quota[source] < floor:
            donor = max(SOURCES, key=lambda candidate: quota[candidate])
            if quota[donor] <= floor:
                break
            quota[donor] -= 1
            quota[source] += 1

    rng = np.random.default_rng(seed)
    positives: list[dict[str, str]] = []
    for source in SOURCES:
        rows = sorted(catalog[source], key=lambda row: row["file"])
        picks = rng.choice(len(rows), size=quota[source], replace=False)
        positives.extend(rows[int(i)] for i in sorted(picks))

    negative_pool = sorted(
        name for name in index if "neg_sample" in name and name not in excluded
    )
    picks = rng.choice(len(negative_pool), size=n_negative, replace=False)
    negatives = [negative_pool[int(i)] for i in sorted(picks)]

    print("Sample drawn (seed=%d):" % seed, flush=True)
    for source in SOURCES:
        print(f"  {source}: {quota[source]} of {population[source]} available")
    print(f"  negatives: {len(negatives)} of {len(negative_pool)} available")
    return positives, negatives


def write_manifest(
    path: Path, positives: list[dict[str, str]], negatives: list[str], *, data_dir: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fits_path", "label", "toa", "dm", "source"])
        for row in positives:
            writer.writerow(
                [f"{data_dir}/{row['file']}", "FRB", row["toa"], row["dms"], row["source"]]
            )
        for name in negatives:
            writer.writerow([f"{data_dir}/{name}", "RFI", "", "", ""])
    print(f"Input manifest written to {path}", flush=True)


# A server error body (e.g. the 429 JSON) is ~100 bytes and gets written in
# place of the FITS; resuming from it with `curl -C -` would corrupt the file.
MIN_PLAUSIBLE_SIZE = 1_000_000


class Throttle:
    """Global backoff: a 429 in any thread pauses all the others.

    ScienceDB rate-limits requests per client. Without this, the remaining
    threads keep hitting the server during the block, burn each file's retries
    and turn a temporary limit into a permanent failure.
    """

    def __init__(self, *, base_penalty: float = 120.0, max_penalty: float = 1800.0) -> None:
        self._lock = threading.Lock()
        self._resume_at = 0.0
        self._penalty = base_penalty
        self._base = base_penalty
        self._max = max_penalty
        self.trips = 0

    def wait(self) -> None:
        while True:
            with self._lock:
                remaining = self._resume_at - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 5.0))

    def trip(self) -> float:
        with self._lock:
            self.trips += 1
            penalty = self._penalty
            self._resume_at = max(self._resume_at, time.monotonic() + penalty)
            self._penalty = min(self._penalty * 2, self._max)
            return penalty

    def success(self) -> None:
        with self._lock:
            self._penalty = self._base


def file_md5(path: Path, *, chunk: int = 1 << 22) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _discard_error_body(target: Path) -> None:
    """Remove leftovers too small to be FITS (a server error body)."""
    if target.exists() and target.stat().st_size < MIN_PLAUSIBLE_SIZE:
        target.unlink()


def download_one(
    name: str, entry: dict[str, Any], data_dir: Path, throttle: Throttle
) -> tuple[str, str]:
    target = data_dir / name
    if target.exists():
        size = target.stat().st_size
        if size == entry["size"]:
            if file_md5(target) == entry["md5"]:
                return name, "ok-cache"
            target.unlink()
        else:
            _discard_error_body(target)

    for attempt in range(1, 6):
        throttle.wait()
        result = subprocess.run(
            [
                "curl", "-sS", "-L", "-C", "-",
                "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                "--connect-timeout", "30",
                "-H", f"User-Agent: {USER_AGENT}",
                "-w", "%{http_code}",
                entry["url"], "-o", str(target),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        status = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if status[0] == "429":
            penalty = throttle.trip()
            _discard_error_body(target)
            print(f"  429 on {name}: pausing {penalty:.0f}s", flush=True)
            continue

        if target.exists() and target.stat().st_size == entry["size"]:
            if file_md5(target) == entry["md5"]:
                throttle.success()
                return name, "ok"
            target.unlink()
        else:
            _discard_error_body(target)
    return name, "FALHOU"


def download_all(
    names: list[str], index: dict[str, dict[str, Any]], data_dir: Path, *, workers: int
) -> list[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    pending = [name for name in names if name in index]
    missing_from_index = [name for name in names if name not in index]
    if missing_from_index:
        raise RuntimeError(f"Files missing from the index: {missing_from_index}")

    total_bytes = sum(index[name]["size"] for name in pending)
    print(
        f"Downloading {len(pending)} files ({total_bytes / 1e9:.1f} GB) "
        f"with {workers} connections into {data_dir} ...",
        flush=True,
    )

    throttle = Throttle()
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, name, index[name], data_dir, throttle): name
            for name in pending
        }
        for future in as_completed(futures):
            name, status = future.result()
            done += 1
            print(f"[{done}/{len(pending)}] {name}: {status}", flush=True)
            if status == "FALHOU":
                failures.append(name)
    if throttle.trips:
        print(f"\n{throttle.trips} 429 responses from the server; backoff applied.", flush=True)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-frb", type=int, default=50)
    parser.add_argument("--n-negative", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("fast_frex"))
    parser.add_argument("--out-manifest", type=Path, default=Path("fast_frex/real_manifest_n100.csv"))
    parser.add_argument("--index-cache", type=Path, default=Path("fast_frex/scidb_index.json"))
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent connections. Above ~16 ScienceDB starts answering 429.",
    )
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--keep-used", action="store_true", help="Do not delete the files already used.")
    parser.add_argument(
        "--min-per-source",
        type=int,
        default=1,
        help="Minimum number of bursts per repeater (0 = strict proportionality).",
    )
    parser.add_argument("--select-only", action="store_true", help="Only draw the sample and write the manifest.")
    args = parser.parse_args()

    index = build_index(args.index_cache, refresh=args.refresh_index)

    for source in SOURCES:
        summary = args.data_dir / f"{source}_summary.csv"
        if not summary.exists():
            download_all([f"{source}_summary.csv"], index, args.data_dir, workers=1)

    positives, negatives = select_sample(
        summary_dir=args.data_dir,
        index=index,
        n_frb=args.n_frb,
        n_negative=args.n_negative,
        seed=args.seed,
        exclude_used=not args.keep_used,
        min_per_source=args.min_per_source,
    )
    write_manifest(
        args.out_manifest, positives, negatives, data_dir=args.data_dir.as_posix()
    )
    if args.select_only:
        return 0

    names = [row["file"] for row in positives] + negatives
    failures = download_all(names, index, args.data_dir, workers=args.workers)
    if failures:
        print(f"\n{len(failures)} files failed: {failures}", file=sys.stderr)
        return 1
    print("\nAll files downloaded and verified by MD5.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
