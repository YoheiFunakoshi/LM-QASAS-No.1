"""Record local requirements without loading models or reading repertoire files.

Exit codes: 0 = report written; 1 = required imports unavailable; 2 = output error.
CUDA is optional: a CPU-only installation can pass --require-packages.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import importlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def failure(exc: Exception) -> dict[str, str]:
    """Avoid paths, account names and environment variables in error reports."""
    return {"status": "error", "error_type": type(exc).__name__}


def memory_info() -> dict[str, Any]:
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64),
                    ("available_physical", ctypes.c_uint64),
                    ("total_pagefile", ctypes.c_uint64),
                    ("available_pagefile", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64),
                ]

            state = MemoryStatus()
            state.length = ctypes.sizeof(state)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            query = kernel32.GlobalMemoryStatusEx
            query.argtypes = [ctypes.POINTER(MemoryStatus)]
            query.restype = ctypes.c_int
            if not query(ctypes.byref(state)):
                raise ctypes.WinError(ctypes.get_last_error())
            return {
                "status": "ok",
                "total_bytes": state.total_physical,
                "available_bytes": state.available_physical,
            }
        if hasattr(os, "sysconf"):
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            return {"status": "ok", "total_bytes": total}
        return {"status": "unavailable", "reason": "unsupported_platform"}
    except (OSError, ValueError, AttributeError) as exc:
        return failure(exc)


def disk_info(workspace: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(workspace)
        return {"status": "ok", "total_bytes": usage.total, "free_bytes": usage.free}
    except OSError as exc:
        return failure(exc)


def import_package(name: str) -> tuple[dict[str, Any], Any]:
    result: dict[str, Any] = {}
    try:
        result["version"] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result["version"] = None
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        result.update(failure(exc))
        result["reason"] = "package_import_failed"
        return result, None
    result["status"] = "ok"
    if result["version"] is None:
        result["version"] = getattr(module, "__version__", None)
    return result, module


def cuda_info(torch: Any) -> dict[str, Any]:
    if torch is None:
        return {"status": "unavailable", "reason": "torch_import_failed"}
    result: dict[str, Any] = {"built_cuda": getattr(torch.version, "cuda", None)}
    try:
        available = bool(torch.cuda.is_available())
        result.update({"status": "ok", "available": available, "devices": []})
        if available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                result["devices"].append({
                    "index": index,
                    "name": props.name,
                    "vram_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                })
    except Exception as exc:
        result.update(failure(exc))
        result["reason"] = "cuda_query_failed"
    return result


def nvidia_smi_info() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "reason": "command_not_found"}
    command = [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": "command_timeout", "timeout_seconds": 15}
    except OSError as exc:
        return {**failure(exc), "reason": "command_start_failed"}
    if completed.returncode != 0:
        return {"status": "error", "reason": "command_failed", "exit_code": completed.returncode}
    try:
        devices = []
        for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError("Unexpected number of GPU fields")
            devices.append({
                "name": row[0].strip(),
                "vram_mib": int(row[1].strip()),
                "driver_version": row[2].strip(),
            })
        return {"status": "ok", "devices": devices}
    except ValueError as exc:
        return {**failure(exc), "reason": "unexpected_command_output"}


def collect_report(workspace: Path) -> dict[str, Any]:
    torch_info, torch = import_package("torch")
    ablang_info, _ = import_package("ablang2")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "logical_cpu_count": os.cpu_count(),
        "memory": memory_info(),
        "workspace_disk": disk_info(workspace),
        "packages": {"torch": torch_info, "ablang2": ablang_info},
        "cuda": cuda_info(torch),
        "nvidia_smi": nvidia_smi_info(),
        "scope": "Package and hardware inspection only; no model loading or repertoire access.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Existing directory whose free disk space should be checked.")
    parser.add_argument("--output", type=Path,
                        help="New JSON file (existing files are never overwritten). Defaults to workspace/local_records/.")
    parser.add_argument("--require-packages", action="store_true",
                        help="Exit 1 if torch or ablang2 cannot be imported. GPU is not required.")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or args.workspace / "local_records" / f"environment_{stamp}.json"
    report = collect_report(args.workspace)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except OSError as exc:
        print(f"Cannot save environment report: {type(exc).__name__}", file=sys.stderr)
        return 2
    missing = [name for name, result in report["packages"].items() if result["status"] != "ok"]
    print("Environment report saved. Package imports: " + (", ".join(missing) + " failed." if missing else "OK."))
    return 1 if args.require_packages and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
