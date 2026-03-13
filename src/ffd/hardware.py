"""Hardware introspection helpers."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict

import psutil


@dataclass(slots=True)
class HardwareSnapshot:
    cpu: str
    cores: int
    memory_gb: float
    os: str
    gpu: str | None = None
    gpu_memory_gb: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _probe_nvidia() -> tuple[str | None, float | None]:
    """Return name/memory by querying ``nvidia-smi`` when available."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    first_line = result.stdout.strip().splitlines()[0]
    if not first_line:
        return None, None
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 2:
        return None, None
    name, memory_raw = parts
    memory_value = float(memory_raw.split()[0]) if memory_raw else None
    return name, memory_value


def gather_hardware_snapshot() -> HardwareSnapshot:
    """Collect basic host metrics for run reports."""

    virtual_mem = psutil.virtual_memory()
    gpu_name, gpu_mem = _probe_nvidia()
    return HardwareSnapshot(
        cpu=platform.processor() or platform.machine(),
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 0,
        memory_gb=round(virtual_mem.total / (1024**3), 2),
        os=f"{platform.system()} {platform.release()}",
        gpu=gpu_name,
        gpu_memory_gb=round(gpu_mem / 1024, 2) if gpu_mem else None,
    )
