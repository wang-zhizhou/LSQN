"""Run explicit LSQN implementation profiles on the three paper experiment suites.

Examples: python -m experiments.run --suite smoke --output results/smoke
          python -m experiments.run --suite all --output results/full
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import importlib.util
import json
import platform
from pathlib import Path
import signal
import sys
import time
import uuid

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

import lsqn
from lsqn import SolverOptions, solve
from .models import (DATA, build_model, case_id, file_hash, model_info, plans, save_model)
from .implementations import NATIVE_BACKENDS, backend_for

METHODS = ["LSQN", "VI", "PI", "MPI-4", "MPI-20", "QPI-U", "QPI-BT", "QPI-S"]
RESULTS_ONLY_METHODS = ("iGMRES-PI", "AA-I-S-5")


def plain(value):
    """Strict JSON: nonfinite numbers are explicit nulls, never NaN tokens."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(plain(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


@contextmanager
def time_budget(seconds):
    """POSIX soft wall limit, including safe restoration of the caller's timer."""
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("This experiment runner requires Linux or macOS (SIGALRM).")
    def expired(signum, frame):
        raise TimeoutError("wall-time budget exceeded")
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def residual_check(mdp, value, tolerance):
    v = np.asarray(value, dtype=float)
    if v.shape != (mdp.n_states,) or not np.all(np.isfinite(v)):
        return dict(passed=False, residual_inf=None, residual2=None, finite=False)
    tv, policy = mdp.bellman(v)
    residual = v - tv
    ri = float(np.linalg.norm(residual, np.inf))
    r2 = float(np.linalg.norm(residual))
    return dict(passed=bool(np.isfinite(ri) and ri <= tolerance),
                residual_inf=ri, residual2=r2, finite=True,
                tolerance=tolerance, value_error_bound_inf=ri / (1 - mdp.gamma))


def audit_steps(history, options, n):
    """Verify recorded forcing, descent and Armijo certificates independently."""
    failures = []
    for index, row in enumerate(history):
        required = ("residual2_before", "residual2_after", "slope", "forcing", "target", "alpha", "enrichments")
        if any(key not in row for key in required):
            failures.append(dict(step=index, check="missing_certificate"))
            continue
        before = row["residual2_before"] ** 2
        after = row["residual2_after"] ** 2
        scale = max(before, 1e-28)
        checks = {
            "finite": all(np.isfinite(row[key]) for key in required),
            "query_bound": 0 <= row["enrichments"] <= n,
            "forcing": row["forcing"] <= row["target"] * (1 + 1e-8),
            "descent": row["slope"] <= -options.descent_fraction * before + 1e-8 * scale,
            "armijo": .5 * after <= .5 * before + options.armijo_sigma * row["alpha"] * row["slope"] + 1e-8 * scale,
        }
        failures.extend(dict(step=index, check=name) for name, passed in checks.items() if not passed)
    return dict(passed=not failures, audited_steps=len(history), failures=failures)


def pi_reference(mdp, initial, seconds, tolerance):
    """Independent exact policy iteration for optional value verification.

    This diagnostic is outside LSQN timing and is not a baseline timing result.
    Every policy evaluation is an ordinary dense/sparse direct linear solve.
    """
    start = time.perf_counter()
    value = initial.copy()
    try:
        with time_budget(seconds):
            for iteration in range(1000):
                tv, policy = mdp.bellman(value)
                if np.linalg.norm(value - tv, np.inf) <= tolerance:
                    break
                value = mdp.evaluate_policy(policy)
            verification = residual_check(mdp, value, tolerance)
        return dict(status="complete" if verification["passed"] else "failed",
                    iterations=iteration, verification=verification,
                    diagnostic_wall_seconds=time.perf_counter() - start, value=value)
    except (TimeoutError, ArithmeticError, np.linalg.LinAlgError) as exc:
        return dict(status="failed", message=str(exc),
                    diagnostic_wall_seconds=time.perf_counter() - start)


def run_one(mdp, spec, options, limit, model, reference=None, method="LSQN",
            backend="reference", implementation="reference"):
    initial = (np.zeros(mdp.n_states) if spec["initialization"] == "zero"
               else mdp.common_lower_bound())
    trace = []
    accepted_history = []
    last_value = initial.copy()

    def callback(record):
        nonlocal last_value
        last_value = np.asarray(record["value"]).copy()
        row = {key: record[key] for key in ("k", "residual2", "residual_inf", "elapsed_s", "event")
               if key in record}
        if trace and trace[-1].get("k") == row.get("k"):
            trace[-1] = row
        else:
            trace.append(row)
        if record.get("event") == "accepted" and record.get("accepted_step") is not None:
            accepted_history.append(dict(record["accepted_step"]))

    gc.collect()
    start = time.perf_counter()
    timed_out = False
    try:
        with time_budget(limit):
            if method == "LSQN":
                result = solve(mdp, options=options, v0=initial, callback=callback, backend=backend)
                if result["backend"] != backend:
                    raise ValueError("Solver returned an unexpected implementation: " + str(result["backend"]))
            else:
                from lsqn_baselines import solve as baseline_solve
                result = baseline_solve(method, mdp, v0=initial, tol=options.tol, callback=callback)
    except (TimeoutError, ArithmeticError, np.linalg.LinAlgError) as exc:
        timed_out = isinstance(exc, TimeoutError)
        result = dict(converged=False, message=str(exc), value=last_value,
                      history=accepted_history, iterations=len(accepted_history), stats={})
    wall = time.perf_counter() - start
    verification = residual_check(mdp, result.get("value", last_value), options.tol)
    step_audit = audit_steps(result.get("history", []), options, mdp.n_states) if method == "LSQN" else None
    success = bool(result.get("converged", False) and verification["passed"]
                   and (step_audit is None or step_audit["passed"]))
    actual_backend = backend if method == "LSQN" else "lsqn_baselines:" + method
    result.update(method=method, implementation=implementation, backend=actual_backend, spec=spec,
                  model=model, wall_seconds=wall, wall_budget_seconds=limit,
                  timeout=timed_out, success=success, trace=trace,
                  independent_verification=verification)
    if method == "LSQN":
        queries = [int(row["enrichments"]) for row in result.get("history", []) if "enrichments" in row]
        result.update(step_audit=step_audit, queries=queries, options=asdict(options),
                      query_summary=dict(completed_outer_iterations=len(queries),
                                         mean=float(np.mean(queries)) if queries else None,
                                         maximum=max(queries) if queries else None,
                                         total=sum(queries), zero=queries.count(0),
                                         full_dimension=sum(q == mdp.n_states for q in queries)))
    else:
        result["baseline_settings"] = dict(tol=options.tol, implementation="lsqn_baselines",
                                            note="Method-specific fixed parameters and iteration limits are documented in baselines-port-notes.txt")
    if reference is not None:
        comparison = {k: v for k, v in reference.items() if k != "value"}
        if reference.get("status") == "failed":
            result["success"] = False
        if reference.get("status") == "complete" and verification["finite"]:
            difference = float(np.linalg.norm(np.asarray(result["value"]) - reference["value"], np.inf))
            bound = (verification["residual_inf"] + reference["verification"]["residual_inf"]) / (1 - mdp.gamma)
            comparison.update(value_difference_inf=difference, residual_error_bound=bound,
                              agrees_with_error_bound=bool(difference <= bound + 1e-8))
            if not comparison["agrees_with_error_bound"]:
                result["success"] = False
        result["reference_pi"] = comparison
    return result


def tokens(values, kind=str):
    return {kind(word) for value in values for word in value.split(",")} if values else None


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=("smoke", "main", "public", "scaling", "all"), default="smoke")
    ap.add_argument("--methods", nargs="+", default=["LSQN"],
                    help="LSQN (default), all included methods, or method names. all runs LSQN and seven baselines; comparison is available for main and smoke only.")
    ap.add_argument("--implementation", choices=("paper", "curves", "reference"), default="paper",
                    help="LSQN main profile: final runtime optimizations (paper), four-figure block32_cache (curves), or original reference. Public/scaling always retain reference.")
    ap.add_argument("--output", type=Path, required=True, help="New or empty output directory; existing results are never overwritten.")
    ap.add_argument("--repetitions", type=int, default=3, help="Main/public repetitions; smoke always runs each case once.")
    ap.add_argument("--scaling-repetitions", type=int, default=1, help="Default is one run per seeded scaling instance.")
    ap.add_argument("--time-limit", type=float, help="Override per-solve seconds. Default main:30/60; public/scaling/smoke:30.")
    ap.add_argument("--families", nargs="+", help="Space- or comma-separated family names.")
    ap.add_argument("--sizes", nargs="+", help="Filter state counts (including public absorbing states).")
    ap.add_argument("--gammas", nargs="+", help="Filter discounts.")
    ap.add_argument("--seeds", nargs="+", help="Filter random seeds; public models use 0 as an identifier.")
    ap.add_argument("--threads", type=int, default=1, help="BLAS threads; paper protocol uses 1.")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--max-outer", type=int, default=500, help="LSQN outer limit; baseline limits remain method-specific.")
    ap.add_argument("--save-models", action="store_true", help="Save exact input arrays to OUTPUT/inputs (dense models can be large).")
    ap.add_argument("--prepare-only", action="store_true", help="Save inputs and exit without any solve.")
    ap.add_argument("--input-models", type=Path, help="Load every requested input from an earlier OUTPUT/inputs directory.")
    ap.add_argument("--reference-pi", action="store_true", help="Optional independent value check, outside timing; smoke enables it automatically.")
    ap.add_argument("--reference-max-states", type=int, default=1024, help="Skip reference PI above this size.")
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    requested_methods = [name for value in args.methods for name in value.split(",")]
    results_only = {name.lower(): name for name in RESULTS_ONLY_METHODS}
    omitted = [results_only[name.lower()] for name in requested_methods if name.lower() in results_only]
    if omitted:
        ap.error(", ".join(omitted) + " is available as archived results only; "
                 "its implementation is not distributed in this public package.")
    if [name.lower() for name in requested_methods] == ["all"]:
        methods = METHODS.copy()
    else:
        lookup = {name.lower(): name for name in METHODS}
        if any(name.lower() not in lookup for name in requested_methods):
            ap.error("Unknown method; choose all or " + ", ".join(METHODS))
        methods = list(dict.fromkeys(lookup[name.lower()] for name in requested_methods))
    if args.suite not in ("main", "smoke") and methods != ["LSQN"]:
        ap.error("public, scaling and all suites use LSQN only. Use --suite main --methods all for the 768-run public subset comparison.")
    if methods != ["LSQN"]:
        from lsqn_baselines import METHODS as available
        if set(available) != set(METHODS[1:]):
            ap.error("Installed baseline package does not match the expected seven runnable baselines.")
    if min(args.repetitions, args.scaling_repetitions, args.threads, args.max_outer, args.reference_max_states) < 1:
        ap.error("Repetitions, threads, max-outer and reference-max-states must be positive.")
    if args.tol <= 0 or (args.time_limit is not None and args.time_limit <= 0):
        ap.error("Tolerance and time limit must be positive.")
    cases = plans(args.suite)
    try:
        for key, chosen in (("family", tokens(args.families)), ("n", tokens(args.sizes, int)),
                            ("gamma", tokens(args.gammas, float)), ("seed", tokens(args.seeds, int))):
            if chosen is not None:
                cases = [spec for spec in cases if spec[key] in chosen]
    except ValueError as exc:
        ap.error(str(exc))
    if not cases:
        ap.error("The filters selected no cases.")
    out = args.output.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        ap.error("Output must be new or empty: " + str(out))
    if args.input_models is not None:
        args.input_models = args.input_models.expanduser().resolve()
        if not (args.input_models / "manifest.json").is_file():
            ap.error("--input-models must contain manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    options = SolverOptions(tol=args.tol, max_outer=args.max_outer)
    experiment_package = Path(__file__).resolve().parent
    solver_package = Path(lsqn.__file__).resolve().parent
    source_hashes = {"experiments/" + p.name: file_hash(p) for p in sorted(experiment_package.glob("*.py"))}
    source_hashes.update({"src/lsqn/" + p.relative_to(solver_package).as_posix(): file_hash(p)
                          for p in sorted(solver_package.rglob("*")) if p.suffix in (".py", ".c", ".h")})
    baseline_spec = importlib.util.find_spec("lsqn_baselines")
    if baseline_spec is not None and baseline_spec.origin:
        baseline_package = Path(baseline_spec.origin).resolve().parent
        source_hashes.update({"src/lsqn_baselines/" + p.name: file_hash(p)
                              for p in sorted(baseline_package.glob("*.py"))})
    backend_plan = {case_id(spec): backend_for(spec, args.implementation) for spec in cases}
    required_backends = sorted(set(backend_plan.values())) if "LSQN" in methods and not args.prepare_only else []
    config = dict(schema="lsqn-run-protocol-v2", run_id=str(uuid.uuid4()), suite=args.suite, cases=cases,
                  methods=methods, method_order="cyclic rotation by repetition, serial runs",
                  requested_options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  solver_options=asdict(options), seed_policy="main130; scaling130/131/132; public frozen",
                  implementation=args.implementation, lsqn_backend_by_case=backend_plan,
                  implementation_sources_sha256=file_hash(DATA / "implementation_sources.json"),
                  comparison_scope=("Smoke health check; not a formal performance comparison" if args.suite == "smoke" else
                                    "Public subset comparison on identical inputs in this run; no archived timing cells are mixed"
                                    if methods == METHODS else "Selected methods: " + ", ".join(methods)),
                  reference_pi_scope="Optional reference PI is a correctness diagnostic outside all method timings",
                  timing="whole solver including data-dependent caches plus passive trace recording; excludes native compilation/library loading, model generation/loading, hashing, PI reference, independent verification and writing",
                  timeout="POSIX soft wall timer; a noninterruptible BLAS call can delay delivery until it returns",
                  python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                  source_sha256=source_hashes,
                  frozen_manifest_sha256=file_hash(DATA / "manifest.json"))
    entries = []
    input_manifest = dict(schema="lsqn-input-manifest-v1", models={})
    input_dir = out / "inputs"
    with threadpool_limits(limits=args.threads):
        config["backend_preparation"] = []
        for backend in required_backends:
            lsqn.prepare_backend(backend)
            config["backend_preparation"].append(dict(backend=backend, loaded=True))
        if NATIVE_BACKENDS.intersection(required_backends):
            from lsqn import _native
            _native.get_library()
            config["native_backend"] = _native.status()
        else:
            config["native_backend"] = dict(required=False)
        config["threadpools"] = threadpool_info()
        write_new(out / "protocol.json", config)
        for spec in cases:
            identity = case_id(spec)
            print("Preparing " + identity, flush=True)
            mdp, provenance = build_model(spec, args.input_models)
            model = {**model_info(mdp), **provenance}
            if mdp.n_states != spec["n"] or mdp.gamma != spec["gamma"]:
                raise ValueError("Input metadata does not match requested case: " + identity)
            if args.save_models or args.prepare_only:
                input_dir.mkdir(exist_ok=True)
                filename = identity + ".npz"
                digest = save_model(mdp, input_dir / filename)
                input_manifest["models"][identity] = dict(file=filename, file_sha256=digest,
                                                         model_sha256=model["model_sha256"],
                                                         source_id=model["source_id"], spec=spec)
            if args.prepare_only:
                entries.append(dict(case_id=identity, prepared=True, model=model))
                continue
            limit = args.time_limit or (60.0 if spec["suite"] == "main" and spec["n"] >= 2048 else 30.0)
            reference = None
            if args.reference_pi or args.suite == "smoke":
                if mdp.n_states <= args.reference_max_states:
                    initial = np.zeros(mdp.n_states) if spec["initialization"] == "zero" else mdp.common_lower_bound()
                    reference = pi_reference(mdp, initial, limit, args.tol)
                else:
                    reference = dict(status="skipped", message="above reference-max-states")
            repetitions = (1 if args.suite == "smoke" else
                           args.scaling_repetitions if spec["suite"] == "scaling" else args.repetitions)
            for repeat in range(repetitions):
                offset = repeat % len(methods)
                order = methods[offset:] + methods[:offset]
                for method in order:
                    result = run_one(mdp, spec, options, limit, model, reference, method=method,
                                     backend=backend_plan[identity], implementation=args.implementation)
                    result["repeat"] = repeat
                    result["run_id"] = config["run_id"]
                    filename = identity + "_" + method + "_r%d.json" % repeat
                    write_new(out / filename, result)
                    entries.append(dict(case_id=identity, method=method, backend=result["backend"], repeat=repeat, file=filename, success=result["success"],
                                        timeout=result["timeout"], wall_seconds=result["wall_seconds"],
                                        residual_inf=result["independent_verification"]["residual_inf"],
                                        query_summary=result.get("query_summary"), model_sha256=model["model_sha256"]))
                    print("%s method=%s repeat=%d success=%s residual=%s queries=%s time=%.3fs" %
                          (identity, method, repeat, result["success"], result["independent_verification"]["residual_inf"],
                           result.get("query_summary", {}).get("total", "n/a"), result["wall_seconds"]), flush=True)
            del mdp
    if args.save_models or args.prepare_only:
        write_new(input_dir / "manifest.json", input_manifest)
    report = dict(schema="lsqn-run-summary-v1", prepared_only=args.prepare_only,
                  implementation=args.implementation,
                  configurations=len(cases), methods=methods, executions=0 if args.prepare_only else len(entries),
                  all_success=None if args.prepare_only else all(row["success"] for row in entries), runs=entries)
    write_new(out / "summary.json", report)
    if not args.prepare_only:
        from .summarize import summarize
        write_new(out / "analysis.json", summarize(out))
    print("Saved " + str(out), flush=True)
    return 0 if args.prepare_only or report["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
