from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


FRB_MIN_WIDTH = 0.003
FRB_DEFAULT_MAX_WIDTH = 0.005
SUPPORTED_NBITS = {1, 2, 4, 8}
TASK_MULTICLASS = "multiclass"
TASK_FRB_BINARY = "frb-binary"
VALID_SIMULATION_TASKS = (TASK_MULTICLASS, TASK_FRB_BINARY)
DEFAULT_SIMULATION_NAME = "FRB SVLM synthetic {sample_id}"
DEFAULT_TELESCOPE = "SimulateSearch mock telescope"
DEFAULT_OBSERVER = "frb_svlm"
DEFAULT_IMJD = 58456
DEFAULT_LEVELSET = 1

REQUIRED_KEYS = {
    "f1",
    "f2",
    "nchan",
    "tsamp",
    "nbits",
    "gain",
    "tsys",
    "output_dir",
    "seed",
    "n_per_class",
}


@dataclass(frozen=True)
class SimulationConfig:
    f1: float
    f2: float
    nchan: int
    tsamp: float
    nbits: int
    gain: float
    tsys: float
    output_dir: Path
    seed: int
    n_per_class: int
    t0: float = 0.0
    t1: float = 2.0
    frb_max_width: float = FRB_DEFAULT_MAX_WIDTH
    name: str = DEFAULT_SIMULATION_NAME
    telescope: str = DEFAULT_TELESCOPE
    observer: str = DEFAULT_OBSERVER
    raj: str = "0"
    decj: str = "0"
    use_angle: int = 0
    imjd: int = DEFAULT_IMJD
    smjd: int | None = None
    levelset: int = DEFAULT_LEVELSET
    task: str = TASK_MULTICLASS

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", normalize_simulation_task(self.task))
        self.validate()

    @classmethod
    def from_yaml(cls, path: Path) -> "SimulationConfig":
        path = path.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config YAML must contain a mapping: {path}")

        missing = sorted(REQUIRED_KEYS - data.keys())
        if missing:
            raise ValueError(
                f"Config YAML missing required keys: {', '.join(missing)}"
            )

        base_dir = path.resolve().parent
        output_dir = Path(str(data["output_dir"])).expanduser()
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir

        config = cls(
            f1=float(data["f1"]),
            f2=float(data["f2"]),
            nchan=int(data["nchan"]),
            tsamp=float(data["tsamp"]),
            nbits=int(data["nbits"]),
            gain=float(data["gain"]),
            tsys=float(data["tsys"]),
            output_dir=output_dir,
            seed=int(data["seed"]),
            n_per_class=int(data["n_per_class"]),
            t0=float(data.get("t0", 0.0)),
            t1=float(data.get("t1", 2.0)),
            frb_max_width=float(data.get("frb_max_width", FRB_DEFAULT_MAX_WIDTH)),
            name=str(data.get("name", DEFAULT_SIMULATION_NAME)),
            telescope=str(data.get("telescope", DEFAULT_TELESCOPE)),
            observer=str(data.get("observer", DEFAULT_OBSERVER)),
            raj=str(data.get("raj", "0")),
            decj=str(data.get("decj", "0")),
            use_angle=int(data.get("useAngle", data.get("use_angle", 0))),
            imjd=int(data.get("imjd", DEFAULT_IMJD)),
            smjd=(
                int(data["smjd"])
                if data.get("smjd") is not None
                else None
            ),
            levelset=int(data.get("levelset", DEFAULT_LEVELSET)),
            task=normalize_simulation_task(data.get("task", TASK_MULTICLASS)),
        )
        return config

    def with_overrides(
        self,
        *,
        n_per_class: int | None = None,
        seed: int | None = None,
        output_dir: Path | None = None,
        task: str | None = None,
    ) -> "SimulationConfig":
        updates: dict[str, Any] = {}
        if n_per_class is not None:
            updates["n_per_class"] = n_per_class
        if seed is not None:
            updates["seed"] = seed
        if output_dir is not None:
            updates["output_dir"] = output_dir.expanduser()
        if task is not None:
            updates["task"] = normalize_simulation_task(task)
        config = replace(self, **updates)
        return config

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    @property
    def samples_per_file(self) -> int:
        return int(round(self.duration / self.tsamp))

    def validate(self) -> None:
        if self.n_per_class <= 0:
            raise ValueError("n_per_class must be greater than zero.")
        if self.task not in VALID_SIMULATION_TASKS:
            supported = ", ".join(VALID_SIMULATION_TASKS)
            raise ValueError(f"task must be one of: {supported}.")
        if self.task == TASK_FRB_BINARY and self.n_per_class % 2 != 0:
            raise ValueError(
                "--task frb-binary requires an even n_per_class to split "
                "NON_FRB equally between RFI and NOISE."
            )
        if self.nchan <= 0:
            raise ValueError("nchan must be greater than zero.")
        if self.tsamp <= 0:
            raise ValueError("tsamp must be greater than zero.")
        if self.nbits not in SUPPORTED_NBITS:
            supported = ", ".join(str(value) for value in sorted(SUPPORTED_NBITS))
            raise ValueError(f"nbits must be one of the supported values: {supported}.")
        if self.f1 == self.f2:
            raise ValueError("f1 and f2 must be different.")
        if self.t0 != 0.0 or self.t1 != 2.0:
            raise ValueError("This module generates 2-second PSRFITS files: use t0=0 and t1=2.")
        if self.duration <= 0:
            raise ValueError("The duration must be positive.")
        if self.samples_per_file <= 0:
            raise ValueError("The t0/t1/tsamp combination must yield at least 1 sample.")
        if self.frb_max_width < FRB_MIN_WIDTH:
            raise ValueError(
                f"frb_max_width must be >= {FRB_MIN_WIDTH:g} seconds."
            )
        if self.frb_max_width >= self.duration:
            raise ValueError("frb_max_width must be shorter than the file duration.")


def normalize_simulation_task(task: str | None) -> str:
    normalized = (task or TASK_MULTICLASS).strip().lower()
    if normalized in {"multiclass", "multi-class", "three-class", "3-class"}:
        return TASK_MULTICLASS
    if normalized in {"binary", "frb_binary", "frb-binary", "frb-vs-non-frb"}:
        return TASK_FRB_BINARY
    raise ValueError(f"Invalid simulation task: {task!r}")
