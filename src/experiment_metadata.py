"""Save resolved experiment settings and hardware without importing neural models."""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _command(arguments: list[str]) -> str:
    try:
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_environment() -> dict:
    """Collect the execution host, not the machine later used to generate reports."""
    environment = {
        "source": "Automatically collected on the experiment machine",
        "architecture": platform.machine(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "cpu": {"logical_processors": os.cpu_count()},
    }
    try:
        fields = {
            item["field"].rstrip(":"): item["data"]
            for item in json.loads(_command(["lscpu", "--json"]))["lscpu"]
        }
    except (ValueError, KeyError, TypeError):
        fields = {}
    cpu = environment["cpu"]
    cpu["model"] = fields.get("Model name", platform.processor() or None)
    for source, target in (
        ("CPU(s)", "logical_processors"),
        ("Socket(s)", "sockets"),
        ("Core(s) per socket", "cores_per_socket"),
        ("Thread(s) per core", "threads_per_core"),
    ):
        if str(fields.get(source, "")).isdigit():
            cpu[target] = int(fields[source])
    environment["virtualization"] = fields.get("Hypervisor vendor")
    try:
        environment["system_memory_mib"] = round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**2
        )
    except (OSError, ValueError, AttributeError):
        pass
    query = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for row in csv.reader(query.splitlines()):
        if len(row) == 3:
            model, memory, driver = (value.strip() for value in row)
            gpus.append(
                {
                    "model": model,
                    "memory_mib": int(memory) if memory.isdigit() else None,
                    "driver_version": driver,
                }
            )
    if gpus:
        supported = re.search(r"CUDA Version:\s*([\d.]+)", _command(["nvidia-smi"]))
        for gpu in gpus:
            if supported:
                gpu["driver_supported_cuda_version"] = supported[1]
        environment["gpu"] = gpus[0]
        environment["gpus"] = gpus
    # Neural runners already import torch. Classical HHC should not load it for metadata.
    torch = sys.modules.get("torch")
    if torch is not None:
        environment["pytorch_version"] = str(torch.__version__)
        environment["pytorch_cuda_version"] = torch.version.cuda
        environment["pytorch_cpu_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            environment["active_cuda_device"] = torch.cuda.current_device()
            environment["active_cuda_device_name"] = torch.cuda.get_device_name()
    return environment


def metadata_path(metrics_path: str | Path) -> Path:
    path = Path(metrics_path)
    stem = path.stem.removesuffix("_metrics")
    return path.with_name(f"{stem}_experiment.json")


def write_experiment_metadata(args, method: str, *, resolved_device: str) -> Path:
    """Persist all parsed arguments, including defaults, after the run timer stops."""
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "schema_version": 1,
        "method": method,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source": "Captured at experiment completion",
        "parameters": parameters,
        "resolved": {"device": resolved_device},
        "environment": collect_environment(),
    }
    path = metadata_path(args.metrics_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
