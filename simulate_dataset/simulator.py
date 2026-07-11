from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from simulate_dataset.config import FRB_MIN_WIDTH, TASK_FRB_BINARY, SimulationConfig


LOGGER = logging.getLogger(__name__)

SIMULATESEARCH_EXECUTABLES = (
    "simulateSystemNoise",
    "simulateBurst",
    "simulateRFI",
    "simulateRFI_far_sidelobes",
    "createSearchFile",
)

CLASSES = ("frb", "rfi", "noise")
RFI_CATEGORIES = (
    "persistent_narrowband",
    "impulsive_broadband",
    "wifi_multiband",
    "point_to_point_microwave",
    "satellite_like_inband",
)
NOISE_TSYS_SCALE_RANGE = (0.65, 1.45)


@dataclass(frozen=True)
class SampleContext:
    sample_id: str
    label: str
    sample_seed: int
    rng: random.Random
    system_tsys: float
    system_tsys_scale: float
    params_dir: Path
    work_dir: Path
    output_path: Path


class SimulateSearchDatasetGenerator:
    def __init__(self, *, config: SimulationConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.output_dir = config.output_dir.resolve()
        self.fits_dir = self.output_dir / "fits"
        self.metadata_dir = self.output_dir / "metadata"
        self.params_root = self.metadata_dir / "params"

    def generate(self, *, overwrite: bool = False) -> list[dict[str, Any]]:
        self._validate_executables()
        if overwrite:
            self._reset_outputs()
        self._create_directories()

        class_counts = self._class_counts()
        LOGGER.info(
            "Starting generation task=%s: FRB=%d, RFI=%d, NOISE=%d, seed=%d",
            self.config.task,
            class_counts["frb"],
            class_counts["rfi"],
            class_counts["noise"],
            self.config.seed,
        )

        rng = random.Random(self.config.seed)
        records: list[dict[str, Any]] = []
        for label in CLASSES:
            count = class_counts[label]
            rfi_categories = (
                self._balanced_rfi_categories(count)
                if label == "rfi"
                else []
            )
            iterator = tqdm(
                range(count),
                desc=f"Generating {label.upper()}",
                unit="sample",
            )
            for index in iterator:
                sample_seed = rng.randrange(1, 2**31 - 1)
                sample_rng = random.Random(sample_seed)
                records.append(
                    self._generate_one(
                        label=label,
                        index=index,
                        sample_seed=sample_seed,
                        sample_rng=sample_rng,
                        rfi_category=rfi_categories[index] if label == "rfi" else None,
                        overwrite=overwrite,
                    )
                )

        self._write_labels(records)
        LOGGER.info("Synthetic dataset generated in %s", self.output_dir)
        return records

    def _class_counts(self) -> dict[str, int]:
        if self.config.task == TASK_FRB_BINARY:
            non_frb_per_source = self.config.n_per_class // 2
            return {
                "frb": self.config.n_per_class,
                "rfi": non_frb_per_source,
                "noise": non_frb_per_source,
            }
        return {label: self.config.n_per_class for label in CLASSES}

    def _validate_executables(self) -> None:
        missing = [
            executable
            for executable in SIMULATESEARCH_EXECUTABLES
            if shutil.which(executable) is None
        ]
        if missing:
            raise RuntimeError(
                "SimulateSearch executables missing from PATH: "
                + ", ".join(missing)
            )
        for executable in SIMULATESEARCH_EXECUTABLES:
            LOGGER.debug("Executable found: %s -> %s", executable, shutil.which(executable))

    def _create_directories(self) -> None:
        for label in CLASSES:
            (self.fits_dir / label).mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.params_root.mkdir(parents=True, exist_ok=True)

    def _reset_outputs(self) -> None:
        targets: list[Path] = [self.params_root]
        targets.extend(self.fits_dir / label for label in CLASSES)
        targets.extend(
            [
                self.metadata_dir / "labels.jsonl",
                self.metadata_dir / "labels.csv",
            ]
        )

        for target in targets:
            if target.is_dir():
                shutil.rmtree(target)
                LOGGER.info("Old artifact removed: %s", target)
            elif target.exists():
                target.unlink()
                LOGGER.info("Old artifact removed: %s", target)

    def _generate_one(
        self,
        *,
        label: str,
        index: int,
        sample_seed: int,
        sample_rng: random.Random,
        rfi_category: str | None,
        overwrite: bool,
    ) -> dict[str, Any]:
        sample_id = f"{label}_{index:05d}"
        output_path = self.fits_dir / label / f"{sample_id}.fits"
        if output_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"File already exists: {output_path}. Use --overwrite to replace it."
                )
            output_path.unlink()

        params_dir = self.params_root / sample_id
        params_dir.mkdir(parents=True, exist_ok=True)
        system_tsys_scale = self._sample_tsys_scale(sample_seed)
        system_tsys = self.config.tsys * system_tsys_scale

        with tempfile.TemporaryDirectory(prefix=f"{sample_id}_", dir=self.metadata_dir) as tmp:
            context = SampleContext(
                sample_id=sample_id,
                label=label,
                sample_seed=sample_seed,
                rng=sample_rng,
                system_tsys=system_tsys,
                system_tsys_scale=system_tsys_scale,
                params_dir=params_dir,
                work_dir=Path(tmp),
                output_path=output_path,
            )
            system_params = self._write_system_params(context)
            self._run(
                ["simulateSystemNoise", "-p", system_params.name, "-o", "noise.dat"],
                context,
            )

            if label == "frb":
                event_params = self._generate_frb(context)
                create_inputs = ["-f", "burst.dat", "-f", "noise.dat"]
                metadata = event_params
            elif label == "rfi":
                event_params, rfi_inputs = self._generate_rfi(
                    context,
                    category=rfi_category,
                )
                create_inputs = [*rfi_inputs, "-f", "noise.dat"]
                metadata = event_params
            elif label == "noise":
                create_inputs = ["-f", "noise.dat"]
                metadata = {"noise": "system_noise"}
            else:
                raise ValueError(f"Unknown class: {label}")

            self._run(
                [
                    "createSearchFile",
                    *create_inputs,
                    "-p",
                    system_params.name,
                    "-o",
                    str(output_path),
                ],
                context,
            )
            self._normalize_fits_date(output_path)

        record = {
            "sample_id": sample_id,
            "label": label.upper(),
            "fits_path": self._relative_path(output_path),
            "seed": sample_seed,
            "parameters": metadata,
            "system": {
                "name": self._format_config_text(self.config.name, context),
                "telescope": self.config.telescope,
                "observer": self.config.observer,
                "f1": self.config.f1,
                "f2": self.config.f2,
                "nchan": self.config.nchan,
                "t0": self.config.t0,
                "t1": self.config.t1,
                "tsamp": self.config.tsamp,
                "raj": self.config.raj,
                "decj": self.config.decj,
                "useAngle": self.config.use_angle,
                "nbits": self.config.nbits,
                "nsblk": self.config.samples_per_file,
                "gain": self.config.gain,
                "tsys": round(context.system_tsys, 6),
                "base_tsys": self.config.tsys,
                "tsys_scale": round(context.system_tsys_scale, 6),
                "imjd": self.config.imjd,
                "smjd": self._sample_smjd(context),
                "levelset": self.config.levelset,
            },
        }
        LOGGER.info("Sample generated: %s -> %s", sample_id, output_path)
        return record

    def _balanced_rfi_categories(self, count: int | None = None) -> list[str]:
        count = self.config.n_per_class if count is None else count
        categories: list[str] = []
        while len(categories) < count:
            categories.extend(RFI_CATEGORIES)
        categories = categories[:count]
        random.Random(self.config.seed + 104729).shuffle(categories)
        return categories

    def _sample_tsys_scale(self, sample_seed: int) -> float:
        low, high = NOISE_TSYS_SCALE_RANGE
        return random.Random(sample_seed ^ 0x5EED5EED).uniform(low, high)

    def _write_system_params(self, context: SampleContext) -> Path:
        name = self._format_config_text(self.config.name, context)
        smjd = self._sample_smjd(context)
        text = "\n".join(
            [
                f"name: {name}",
                f"telescope: {self.config.telescope}",
                f"observer: {self.config.observer}",
                f"f1: {self.config.f1:g}",
                f"f2: {self.config.f2:g}",
                f"nchan: {self.config.nchan}",
                f"t0: {self.config.t0:g}",
                f"t1: {self.config.t1:g}",
                f"tsamp: {self.config.tsamp:g}",
                f"raj: {self.config.raj}",
                f"decj: {self.config.decj}",
                f"useAngle: {self.config.use_angle}",
                f"gain: {self.config.gain:g}",
                f"tsys: {context.system_tsys:g}",
                f"nbits: {self.config.nbits}",
                f"nsblk: {self.config.samples_per_file}",
                f"imjd: {self.config.imjd}",
                f"smjd: {smjd}",
                f"levelset: {self.config.levelset}",
                f"seed: {-context.sample_seed}",
                "",
            ]
        )
        return self._write_param_file(context, "system.params", text)

    def _format_config_text(self, value: str, context: SampleContext) -> str:
        return value.format(
            sample_id=context.sample_id,
            label=context.label,
            seed=context.sample_seed,
        )

    def _sample_smjd(self, context: SampleContext) -> int:
        if self.config.smjd is not None:
            return self.config.smjd
        return context.sample_seed % 86400

    def _generate_frb(self, context: SampleContext) -> dict[str, Any]:
        f_low = min(self.config.f1, self.config.f2)
        f_high = max(self.config.f1, self.config.f2)
        bandwidth = f_high - f_low

        arrival_time = context.rng.uniform(0.35, 1.65)
        reference_frequency = context.rng.uniform(
            f_low + 0.2 * bandwidth,
            f_high - 0.2 * bandwidth,
        )
        flux_density = context.rng.uniform(4.0, 12.0)
        dm_index = context.rng.choice([-2.0, -2.1, -1.9])
        width = context.rng.uniform(FRB_MIN_WIDTH, self.config.frb_max_width)
        dm = context.rng.uniform(120.0, 900.0)
        profile_type = context.rng.choice([1, 2])

        burst_text = (
            "dmburst: "
            f"{arrival_time:.6f} "
            f"{reference_frequency:.6f} "
            f"{flux_density:.6f} "
            f"{dm_index:.6f} "
            f"{width:.6f} "
            f"{dm:.6f} "
            f"{profile_type}\n"
        )
        burst_params = self._write_param_file(context, "burst.params", burst_text)
        self._run(
            [
                "simulateBurst",
                "-p",
                burst_params.name,
                "-p",
                "system.params",
                "-o",
                "burst.dat",
            ],
            context,
        )

        return {
            "arrival_time": round(arrival_time, 6),
            "reference_frequency": round(reference_frequency, 6),
            "flux_density": round(flux_density, 6),
            "dm_index": round(dm_index, 6),
            "width": round(width, 6),
            "max_width": round(self.config.frb_max_width, 6),
            "dm": round(dm, 6),
            "profile_type": profile_type,
        }

    def _generate_rfi(
        self,
        context: SampleContext,
        *,
        category: str | None,
    ) -> tuple[dict[str, Any], list[str]]:
        selected = category or context.rng.choice(RFI_CATEGORIES)
        writers = {
            "persistent_narrowband": (self._write_narrowband_rfi, "narrowband.dat"),
            "impulsive_broadband": (self._write_impulsive_rfi, "impulsive.dat"),
            "wifi_multiband": (self._write_wifi_rfi, "wifi.dat"),
            "point_to_point_microwave": (
                self._write_point_to_point_rfi,
                "point_to_point.dat",
            ),
            "satellite_like_inband": (
                self._write_satellite_like_rfi,
                "satellite_like.dat",
            ),
        }
        if selected not in writers:
            raise ValueError(f"Unknown RFI category: {selected}")

        writer, output_name = writers[selected]
        params = writer(context)
        if selected in {"point_to_point_microwave", "satellite_like_inband"}:
            self._run(
                [
                    "simulateRFI_far_sidelobes",
                    "-p",
                    params["params_file"],
                    "-p",
                    "system.params",
                    "-o",
                    output_name,
                ],
                context,
            )
        else:
            self._run(
                [
                    "simulateRFI",
                    "-seed",
                    str(context.sample_seed),
                    "-p",
                    params["params_file"],
                    "-p",
                    "system.params",
                    "-o",
                    output_name,
                ],
                context,
            )

        return (
            {
                "rfi_types": [selected],
                "rfi_category": selected,
                selected: params,
            },
            ["-f", output_name],
        )

    def _write_narrowband_rfi(self, context: SampleContext) -> dict[str, Any]:
        f_low = min(self.config.f1, self.config.f2)
        f_high = max(self.config.f1, self.config.f2)
        bandwidth = f_high - f_low
        width_mhz = context.rng.uniform(4.0, max(5.0, 0.05 * bandwidth))
        lower_freq = context.rng.uniform(f_low + 0.1 * bandwidth, f_high - width_mhz)
        upper_freq = lower_freq + width_mhz
        flux_density = context.rng.uniform(40.0, 140.0)
        spike_width = context.rng.uniform(0.02, 0.12)
        gap_time = context.rng.uniform(0.01, 0.06)

        text = "\n".join(
            [
                "name: persistent narrowband RFI",
                "rfi_narrowBand: "
                f"{lower_freq:.6f} "
                f"{upper_freq:.6f} "
                f"{flux_density:.6f} "
                f"{self.config.t0:.6f} "
                f"{self.config.duration:.6f} "
                f"{spike_width:.6f} "
                f"{gap_time:.6f}",
                "",
            ]
        )
        self._write_param_file(context, "narrowband.params", text)
        return {
            "params_file": "narrowband.params",
            "lower_freq": round(lower_freq, 6),
            "upper_freq": round(upper_freq, 6),
            "flux_density": round(flux_density, 6),
            "start_time": self.config.t0,
            "duration": self.config.duration,
            "spike_width": round(spike_width, 6),
            "gap_time": round(gap_time, 6),
        }

    def _write_wifi_rfi(self, context: SampleContext) -> dict[str, Any]:
        f_low = min(self.config.f1, self.config.f2)
        f_high = max(self.config.f1, self.config.f2)
        bandwidth = f_high - f_low
        band_count = context.rng.choice([2, 3])
        bands: list[dict[str, float]] = []
        lines = ["name: wifi-like multi-band RFI"]

        for index in range(band_count):
            center_fraction = (index + 1) / (band_count + 1)
            center_fraction += context.rng.uniform(-0.04, 0.04)
            center_fraction = min(0.9, max(0.1, center_fraction))
            center_frequency = f_low + center_fraction * bandwidth
            width_mhz = context.rng.uniform(
                max(3.0, 0.025 * bandwidth),
                max(4.0, 0.075 * bandwidth),
            )
            lower_freq = max(f_low, center_frequency - width_mhz / 2.0)
            upper_freq = min(f_high, center_frequency + width_mhz / 2.0)
            duration = context.rng.uniform(0.45 * self.config.duration, self.config.duration)
            start_time = context.rng.uniform(self.config.t0, self.config.t1 - duration)
            flux_density = context.rng.uniform(45.0, 160.0)
            spike_width = context.rng.uniform(0.003, 0.02)
            gap_time = context.rng.uniform(0.03, 0.15)

            lines.append(
                "rfi_narrowBand: "
                f"{lower_freq:.6f} "
                f"{upper_freq:.6f} "
                f"{flux_density:.6f} "
                f"{start_time:.6f} "
                f"{duration:.6f} "
                f"{spike_width:.6f} "
                f"{gap_time:.6f}"
            )
            bands.append(
                {
                    "lower_freq": round(lower_freq, 6),
                    "upper_freq": round(upper_freq, 6),
                    "flux_density": round(flux_density, 6),
                    "start_time": round(start_time, 6),
                    "duration": round(duration, 6),
                    "spike_width": round(spike_width, 6),
                    "gap_time": round(gap_time, 6),
                }
            )

        lines.append("")
        self._write_param_file(context, "wifi.params", "\n".join(lines))
        return {
            "params_file": "wifi.params",
            "band_count": band_count,
            "bands": bands,
        }

    def _write_point_to_point_rfi(self, context: SampleContext) -> dict[str, Any]:
        f_low = min(self.config.f1, self.config.f2)
        f_high = max(self.config.f1, self.config.f2)
        preferred_frequency = 1495.0
        if f_low <= preferred_frequency <= f_high:
            frequency = preferred_frequency + context.rng.uniform(-2.0, 2.0)
            frequency = min(f_high, max(f_low, frequency))
        else:
            frequency = context.rng.uniform(f_low + 0.2 * (f_high - f_low), f_high - 0.2 * (f_high - f_low))
        source_amplitude = context.rng.uniform(0.2, 1.0)
        source_duration = context.rng.uniform(0.4, self.config.duration)
        sidelobe_amplitude = context.rng.uniform(2.0, 7.0)
        time_scale = context.rng.uniform(0.2, 0.8)

        freq_text = f"{frequency:.6f} {source_amplitude:.6f} {source_duration:.6f}\n"
        self._write_param_file(context, "point_to_point.freq", freq_text)
        params_text = "\n".join(
            [
                "name: point to point microwave link",
                f"farsidelobe: point_to_point.freq {sidelobe_amplitude:.6f} {time_scale:.6f}",
                "",
            ]
        )
        self._write_param_file(context, "point_to_point.params", params_text)
        return {
            "params_file": "point_to_point.params",
            "frequency_file": "point_to_point.freq",
            "frequency": round(frequency, 6),
            "source_amplitude": round(source_amplitude, 6),
            "source_duration": round(source_duration, 6),
            "sidelobe_amplitude": round(sidelobe_amplitude, 6),
            "time_scale": round(time_scale, 6),
        }

    def _write_satellite_like_rfi(self, context: SampleContext) -> dict[str, Any]:
        f_low = min(self.config.f1, self.config.f2)
        f_high = max(self.config.f1, self.config.f2)
        bandwidth = f_high - f_low
        source_count = context.rng.randint(3, 5)
        sources: list[dict[str, float]] = []
        freq_lines: list[str] = []
        params_lines = ["name: satellite-like in-band RFI"]

        for index in range(source_count):
            fraction = (index + 1) / (source_count + 1)
            frequency = f_low + fraction * bandwidth + context.rng.uniform(-0.03, 0.03) * bandwidth
            frequency = min(f_high, max(f_low, frequency))
            source_amplitude = context.rng.uniform(0.3, 2.5)
            source_duration = context.rng.uniform(0.25, self.config.duration)
            freq_lines.append(f"{frequency:.6f} {source_amplitude:.6f} {source_duration:.6f}")
            sources.append(
                {
                    "frequency": round(frequency, 6),
                    "source_amplitude": round(source_amplitude, 6),
                    "source_duration": round(source_duration, 6),
                }
            )

        farsidelobe_count = context.rng.randint(3, 6)
        for _ in range(farsidelobe_count):
            amplitude = context.rng.uniform(0.05, 0.4)
            time_scale = context.rng.uniform(0.8, 2.4)
            params_lines.append(
                f"farsidelobe: satellite_like.freq {amplitude:.6f} {time_scale:.6f}"
            )

        freq_text = "\n".join(freq_lines) + "\n"
        params_text = "\n".join(params_lines) + "\n"
        self._write_param_file(context, "satellite_like.freq", freq_text)
        self._write_param_file(context, "satellite_like.params", params_text)
        return {
            "params_file": "satellite_like.params",
            "frequency_file": "satellite_like.freq",
            "source_count": source_count,
            "farsidelobe_count": farsidelobe_count,
            "sources": sources,
        }

    def _write_impulsive_rfi(self, context: SampleContext) -> dict[str, Any]:
        event_count = context.rng.randint(6, 16)
        events: list[dict[str, float]] = []
        lines: list[str] = []
        for _ in range(event_count):
            flux_density = context.rng.uniform(4.0, 30.0)
            width = context.rng.uniform(
                max(self.config.tsamp, 0.0005),
                min(0.05, max(0.002, 20 * self.config.tsamp)),
            )
            events.append(
                {
                    "flux_density": round(flux_density, 6),
                    "width": round(width, 6),
                }
            )
            lines.append(f"{flux_density:.6f} {width:.6f}")

        events_text = "\n".join(lines) + "\n"
        self._write_param_file(context, "rfi_events.rfi", events_text)

        params_text = "\n".join(
            [
                "name: impulsive broadband RFI",
                "rfi_file: rfi_events.rfi",
                f"nrfi: {event_count}",
                "",
            ]
        )
        self._write_param_file(context, "impulsive.params", params_text)
        return {
            "params_file": "impulsive.params",
            "rfi_file": "rfi_events.rfi",
            "nrfi": event_count,
            "events": events,
        }

    def _write_tone_rfi(self, context: SampleContext) -> dict[str, Any]:
        tones = []
        lines = ["name: birdies tone RFI"]
        for base_frequency in [50.0, 100.0]:
            amplitude = context.rng.uniform(0.3, 2.0)
            tone_type = context.rng.choice([1, 2])
            phase = context.rng.uniform(0.0, 1.0)
            changing_frequency = context.rng.choice([0.0, 0.01, -0.01])
            tones.append(
                {
                    "frequency_hz": base_frequency,
                    "amplitude": round(amplitude, 6),
                    "type": tone_type,
                    "phase": round(phase, 6),
                    "changing_frequency": changing_frequency,
                }
            )
            lines.append(
                "rfi_tone: "
                f"{base_frequency:.6f} "
                f"{amplitude:.6f} "
                f"{tone_type} "
                f"{phase:.6f} "
                f"{changing_frequency:.6f}"
            )
        lines.append("")
        self._write_param_file(context, "tones.params", "\n".join(lines))
        return {
            "params_file": "tones.params",
            "tones": tones,
        }

    def _write_param_file(
        self,
        context: SampleContext,
        filename: str,
        text: str,
    ) -> Path:
        archive_path = context.params_dir / filename
        work_path = context.work_dir / filename
        archive_path.write_text(text, encoding="utf-8")
        work_path.write_text(text, encoding="utf-8")
        return work_path

    def _run(self, command: list[str], context: SampleContext) -> None:
        env = os.environ.copy()
        env.setdefault("SEED", str(context.sample_seed))
        env.setdefault("GSL_RNG_SEED", str(context.sample_seed))
        env.setdefault("SIMULATESEARCH_SEED", str(context.sample_seed))

        LOGGER.debug("Running in %s: %s", context.work_dir, " ".join(command))
        try:
            completed = subprocess.run(
                command,
                cwd=context.work_dir,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            LOGGER.error("Command failed: %s", " ".join(command))
            if exc.stdout:
                LOGGER.error("stdout:\n%s", exc.stdout)
            if exc.stderr:
                LOGGER.error("stderr:\n%s", exc.stderr)
            raise

        if completed.stdout:
            LOGGER.debug("stdout:\n%s", completed.stdout)
        if completed.stderr:
            LOGGER.debug("stderr:\n%s", completed.stderr)

    def _normalize_fits_date(self, path: Path) -> None:
        replacement = (
            "DATE    = '2000-01-01T00:00:00' / file creation date "
            "(YYYY-MM-DDThh:mm:ss UT)"
        )
        replacement_bytes = replacement.ljust(80)[:80].encode("ascii")

        data = bytearray(path.read_bytes())
        replaced = False
        for offset in range(0, len(data), 80):
            card = data[offset : offset + 80]
            if len(card) < 80:
                break
            if card.startswith(b"DATE    ="):
                data[offset : offset + 80] = replacement_bytes
                replaced = True
                break
            if card.startswith(b"END"):
                break

        if replaced:
            path.write_bytes(data)
            LOGGER.debug("FITS DATE field normalized: %s", path)
        else:
            LOGGER.warning("DATE field not found in FITS: %s", path)

    def _write_labels(self, records: list[dict[str, Any]]) -> None:
        jsonl_path = self.metadata_dir / "labels.jsonl"
        csv_path = self.metadata_dir / "labels.csv"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        fieldnames = [
            "sample_id",
            "label",
            "fits_path",
            "seed",
            "arrival_time",
            "reference_frequency",
            "flux_density",
            "dm_index",
            "width",
            "dm",
            "profile_type",
            "rfi_types",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                params = record["parameters"]
                row = {
                    "sample_id": record["sample_id"],
                    "label": record["label"],
                    "fits_path": record["fits_path"],
                    "seed": record["seed"],
                    "arrival_time": params.get("arrival_time", ""),
                    "reference_frequency": params.get("reference_frequency", ""),
                    "flux_density": params.get("flux_density", ""),
                    "dm_index": params.get("dm_index", ""),
                    "width": params.get("width", ""),
                    "dm": params.get("dm", ""),
                    "profile_type": params.get("profile_type", ""),
                    "rfi_types": "|".join(params.get("rfi_types", [])),
                }
                writer.writerow(row)

        LOGGER.info("Labels written to %s and %s", jsonl_path, csv_path)

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
