"""Explicit preparation of the original native kernels on Linux and macOS.

The kernels call the BLAS actually linked to NumPy. LP64 and ILP64 are compiled
with different C function-pointer signatures; the two ABIs are never cast into
one another. Runtime loading never invokes a compiler or falls back to Python.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import tempfile

import numpy as np

_ROOT = Path(__file__).resolve().parent
_STATE = None
_P = ctypes.c_void_p
_I = ctypes.c_int64
_N = ctypes.c_size_t
_D = ctypes.c_double
_MAX_DIMENSION = 2**31 - 1


class NativeBackendError(RuntimeError):
    """The requested native implementation cannot run in this environment."""


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_dimensions(*dimensions):
    """All supported kernels use dimensions within both BLAS integer ranges."""
    if any(int(n) != n or n < 0 or n > _MAX_DIMENSION for n in dimensions):
        raise NativeBackendError("Native dimensions must be integers in [0, 2**31 - 1].")


def _symbol_file(function):
    class DlInfo(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char_p), ("base", _P),
                    ("symbol", ctypes.c_char_p), ("address", _P)]

    handle = ctypes.CDLL(None)
    if not hasattr(handle, "dladdr"):
        library = ctypes.util.find_library("dl")
        if not library:
            raise NativeBackendError("Cannot verify the BLAS symbol's shared library (dladdr unavailable).")
        handle = ctypes.CDLL(library)
    address = handle.dladdr
    address.argtypes = [_P, ctypes.POINTER(DlInfo)]
    address.restype = ctypes.c_int
    info = DlInfo()
    if not address(ctypes.cast(function, _P), ctypes.byref(info)) or not info.name:
        raise NativeBackendError("Cannot identify the actual shared library for a BLAS symbol.")
    return str(Path(os.fsdecode(info.name)).resolve())


def _discover_blas():
    if platform.system() not in ("Linux", "Darwin"):
        raise NativeBackendError("The native backend supports Linux and macOS only.")
    if ctypes.sizeof(_P) != 8:
        raise NativeBackendError("The native backend requires a 64-bit process.")
    # dlsym on NumPy's own extension searches its actual loaded dependencies.
    # This avoids selecting a second, unrelated BLAS loaded by SciPy or conda.
    extension = np.core._multiarray_umath
    try:
        numpy_library = ctypes.CDLL(extension.__file__)
    except OSError as exc:
        raise NativeBackendError("Cannot inspect NumPy's linked BLAS: " + str(exc)) from exc
    return _detect_openblas(numpy_library)


def _detect_openblas(numpy_library):
    """Verify symbol ownership and OpenBLAS's own integer-width declaration."""
    choices = [
        (64, "cblas_ddot64_", "cblas_dgemv64_", "openblas_get_config64_"),
        (32, "cblas_ddot", "cblas_dgemv", "openblas_get_config"),
    ]
    detected = []
    for bits, dot_name, gemv_name, config_name in choices:
        if not all(hasattr(numpy_library, name) for name in (dot_name, gemv_name, config_name)):
            continue
        dot = getattr(numpy_library, dot_name)
        gemv = getattr(numpy_library, gemv_name)
        config = getattr(numpy_library, config_name)
        files = {_symbol_file(symbol) for symbol in (dot, gemv, config)}
        if len(files) != 1:
            raise NativeBackendError("NumPy BLAS symbols resolve to different libraries; refusing an ambiguous ABI.")
        config.argtypes = []
        config.restype = ctypes.c_char_p
        value = config()
        description = value.decode("ascii", errors="strict") if value else ""
        if not description.startswith("OpenBLAS "):
            raise NativeBackendError("Unrecognized OpenBLAS configuration string.")
        _validate_abi_config(bits, description)
        detected.append(dict(bits=bits, dot=dot, gemv=gemv, library=numpy_library,
                             path=files.pop(), config=description,
                             symbols=dict(dot=dot_name, gemv=gemv_name, config=config_name)))
    if len(detected) != 1:
        raise NativeBackendError(
            "Native kernels require one identifiable NumPy OpenBLAS LP64/ILP64 backend. "
            "Install this package's pinned NumPy wheel and rerun scripts/prepare_backend.py. "
            "MKL, Accelerate, and ambiguous BLAS configurations are not silently substituted."
        )
    return detected[0]


def _validate_abi_config(bits, description):
    declared_bits = 64 if "USE64BITINT" in description.split() else 32
    if bits != declared_bits:
        raise NativeBackendError("OpenBLAS symbol names and integer configuration disagree.")


def _identity(blas):
    sources = sorted((_ROOT / "sources").glob("*"))
    hashes = {path.name: _digest(path) for path in sources if path.suffix in (".c", ".h")}
    if not {"copy_rows.c", "selected_rows.c", "query.c", "abi.c", "blas_abi.h"} <= hashes.keys():
        raise NativeBackendError("Native source files are missing from the installed package.")
    identity = dict(schema="lsqn-native-v1", platform=platform.system(), machine=platform.machine(),
                    numpy=np.__version__, blas_integer_bits=blas["bits"],
                    blas_library=blas["path"], blas_library_sha256=_digest(blas["path"]),
                    blas_config=blas["config"], blas_symbols=blas["symbols"], source_sha256=hashes)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, key


def _cache_root(cache_dir=None):
    return Path(cache_dir or os.environ.get("LSQN_NATIVE_CACHE") or
                Path.home() / ".cache" / "lsqn-mdp" / "native").expanduser().resolve()


def _paths(key, cache_dir=None):
    directory = _cache_root(cache_dir) / key
    extension = ".dylib" if platform.system() == "Darwin" else ".so"
    return directory, directory / ("lsqn_kernels" + extension), directory / "build.json"


def _configure_library(library):
    signatures = {
        "scb_copy_policy_rows": ([_P] * 4 + [_N], None),
        "scb_dense_policy": ([_P] * 4 + [_N, _P], None),
        "scb_dense_rows": ([_P] * 4 + [_N, _N, _P], None),
        "scb_sparse_policy": ([_P] * 6 + [_N], None),
        "scb_sparse_j": ([_P] * 6 + [_N, _D], None),
        "scb_sparse_certificate": ([_P] * 9 + [_N, _N, _D], None),
        "scb_project": ([_P] * 3 + [_I, _I, _P], None),
        "scb_protect": ([_P] * 5 + [_I, _I, _P], None),
        "scb_lowrank": ([_P] * 5 + [_I, _I, _D, _P], None),
        "scb_query": ([_P, _I, _I, _I, _I, _D, _D, _P, _P, _P], ctypes.c_int),
        "lsqn_blas_abi_bits": ([], ctypes.c_int),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = result


def _load(blas, identity, key, cache_dir=None):
    global _STATE
    directory, binary, manifest = _paths(key, cache_dir)
    if not binary.is_file() or not manifest.is_file():
        raise NativeBackendError("Native kernels are not prepared for this environment. Run scripts/prepare_backend.py first.")
    try:
        metadata = json.loads(manifest.read_text())
        if metadata["identity"] != identity or metadata["binary_sha256"] != _digest(binary):
            raise NativeBackendError("Prepared native build does not match its source/BLAS/hash metadata; rerun preparation.")
        library = ctypes.CDLL(str(binary))
        _configure_library(library)
        if library.lsqn_blas_abi_bits() != blas["bits"]:
            raise NativeBackendError("Compiled kernels use a different BLAS integer ABI.")
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        raise NativeBackendError("Cannot load prepared native kernels: " + str(exc)) from exc
    _STATE = dict(library=library, blas=blas, metadata=metadata, binary=str(binary))
    return library


def _compiler():
    requested = os.environ.get("CC")
    command = shlex.split(requested) if requested else []
    if not command:
        command = [next((name for name in ("cc", "clang", "gcc") if shutil.which(name)), "")]
    if not command[0] or not shutil.which(command[0]):
        raise NativeBackendError("No C compiler is available. Install a C compiler in your environment, then rerun preparation; no sudo or automatic installation is performed.")
    try:
        result = subprocess.run(command + ["--version"], capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeBackendError("Cannot query the selected C compiler: " + str(exc)) from exc
    return command, result.stdout.strip() or result.stderr.strip()


def prepare(force=False, cache_dir=None):
    """Explicitly build/load kernels; all compilation occurs before solve timing."""
    blas = _discover_blas()
    identity, key = _identity(blas)
    directory, binary, manifest = _paths(key, cache_dir)
    if binary.is_file() and manifest.is_file() and not force:
        _load(blas, identity, key, cache_dir)
        return status()
    compiler, compiler_version = _compiler()
    flags = ["-std=c99", "-O3", "-ffp-contract=off", "-fPIC", "-DLSQN_BLAS_BITS=" + str(blas["bits"])]
    flags += ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared"]
    sources = [str(_ROOT / "sources" / name) for name in ("abi.c", "copy_rows.c", "selected_rows.c", "query.c")]
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as temporary:
        built = Path(temporary) / binary.name
        command = compiler + flags + sources + ["-o", str(built), "-lm"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise NativeBackendError("Native compilation failed: " + str(exc)) from exc
        if result.returncode:
            raise NativeBackendError("Native compilation failed:\n" + result.stdout + result.stderr)
        metadata = dict(identity=identity, compiler=compiler, compiler_version=compiler_version,
                        compiler_flags=flags, binary_sha256=_digest(built),
                        fast_math=False, compilation_in_solve_timing=False)
        staged = Path(temporary) / "build.json"
        staged.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(built, binary)
        os.replace(staged, manifest)
    _load(blas, identity, key, cache_dir)
    return status()


def get_library():
    """Load only a previously prepared matching build, never compile."""
    if _STATE is None:
        blas = _discover_blas()
        identity, key = _identity(blas)
        _load(blas, identity, key)
    return _STATE["library"]


def blas_pointers():
    """Return (dot, gemv) pointers with the compiled kernel's exact BLAS ABI."""
    get_library()
    return tuple(ctypes.cast(_STATE["blas"][name], _P) for name in ("dot", "gemv"))


def status():
    """JSON-serializable metadata; does not load or compile anything."""
    if _STATE is None:
        return dict(prepared=False)
    return dict(prepared=True, **_STATE["metadata"], binary=_STATE["binary"])


def selftest():
    """Exercise the prepared kernels against independent NumPy/SciPy formulas."""
    from ._selftest import run
    get_library()
    return run()
