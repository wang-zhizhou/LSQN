"""Prepare the numerical library and check it before running experiments."""
import json
import os


def main():
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits
    from lsqn import _native
    with threadpool_limits(limits=1):
        prepared = _native.prepare()
        checked = _native.selftest()
    print(json.dumps(dict(preparation=prepared, checks=checked), indent=2))
    print("Numerical library ready. You can now run the smoke checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
