"""Read-only machine metadata for a numerical experiment; no network calls."""
import contextlib
import datetime
import importlib.metadata
import io
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture(threads=1):
    import numpy as np
    import scipy
    import scipy.linalg
    import scipy.sparse.linalg
    from threadpoolctl import threadpool_info, threadpool_limits

    system = platform.system()
    cpu = None
    memory_bytes = None
    lscpu = None
    if system == "Linux":
        lscpu = command(["lscpu", "--json"])
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    memory_bytes = int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    elif system == "Darwin":
        cpu = command(["sysctl", "-n", "machdep.cpu.brand_string"])
        memory = command(["sysctl", "-n", "hw.memsize"])
        if memory and memory.isdigit():
            memory_bytes = int(memory)
    dependencies = {}
    for package in ("lsqn-mdp", "numpy", "scipy", "threadpoolctl", "setuptools", "pip"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    build = io.StringIO()
    with contextlib.redirect_stdout(build):
        np.show_config()
        scipy.show_config()
    with threadpool_limits(limits=threads):
        pools = [{key: value for key, value in item.items() if key != "filepath"}
                 for item in threadpool_info()]
    return dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        system=system, release=platform.release(), architecture=platform.machine(),
        python=sys.version, cpu_model=cpu, logical_cpus=os.cpu_count(),
        memory_bytes=memory_bytes, lscpu_json=lscpu,
        gpu_inventory=command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]),
        gpu_used=False, dependencies=dependencies, requested_blas_threads=threads,
        threadpools_under_requested_limit=pools, numpy_scipy_build=build.getvalue(),
        cpu_affinity="Not pinned; one BLAS thread does not select a physical core.",
    )
