"""Seven runnable baselines from the LSQN experimental comparison.

AA-I-S-5 and iGMRES-PI are represented by archived results and settings only;
their implementations are not distributed in this public package.
"""

import numpy as np

from . import _basic, _comparators
from ._metrics import RunResult
from ._trace import Recorder


METHODS = (
    "VI", "PI", "MPI-4", "MPI-20", "QPI-U", "QPI-BT", "QPI-S",
)

RESULTS_ONLY_METHODS = ("iGMRES-PI", "AA-I-S-5")


def solve(method, mdp, v0=None, tol=1e-8, callback=None):
    """Run a baseline with final-paper parameters and unchanged numerical guards.

    The supplied starting vector is copied without shifting it. If omitted,
    ``mdp.common_lower_bound()`` is used. The absolute infinity-residual target
    defaults to 1e-8. Outer limits are 1000 for PI and 30000 otherwise.

    Returns a dictionary with value, policy, iterations, converged, message,
    history, stats, residual, runtime_s, method and trace. Numerical failures
    remain failures. Baseline-native history/statistic conventions are retained;
    the history length need not equal iterations (VI/PI include a terminal row).

    callback(record) follows the LSQN callback convention: initial, accepted,
    and finished events with k, a value copy, residual2, residual_inf, elapsed_s,
    accepted_step and queries=0 (LSQN learning queries do not apply here).
    TimeoutError propagates so the experiment runner can enforce its own timer.
    There is no operating-system timer or dynamic source rewriting in this API.
    """
    if method in RESULTS_ONLY_METHODS:
        raise ValueError(str(method) + " is available as archived results only; "
                         "its implementation is not distributed in this public package.")
    if method not in METHODS:
        raise ValueError("unknown baseline: " + str(method))
    initial = np.asarray(mdp.common_lower_bound() if v0 is None else v0,
                         dtype=float)
    if initial.shape != (mdp.n_states,) or not np.all(np.isfinite(initial)):
        raise ValueError("v0 must be a finite vector with one entry per state")
    recorder = Recorder(callback)
    try:
        if method == "VI":
            result = _basic.solve_vi(mdp, tol=tol, max_outer=30000, v0=initial,
                                     observe=recorder.observe)
        elif method == "PI":
            result = _basic.solve_pi(mdp, tol=tol, max_outer=1000, v0=initial,
                                     observe=recorder.observe)
        elif method == "QPI-U":
            result = _basic.solve_qpi(mdp, tol=tol, max_outer=30000, v0=initial,
                                      observe=recorder.observe)
        elif method.startswith("MPI-"):
            result = _comparators.solve_mpi_cached(
                mdp, v0=initial, tol=tol, max_outer=30000,
                depth=int(method.split("-")[1]), observe=recorder.observe,
            )
        elif method in ("QPI-BT", "QPI-S"):
            result = _comparators.solve_qpi_bt(
                mdp, v0=initial, tol=tol, max_outer=30000,
                secant=method == "QPI-S", observe=recorder.observe,
            )
    except (ArithmeticError, np.linalg.LinAlgError) as exc:
        if recorder.value is None:
            raise
        _, policy = mdp.bellman(recorder.value)
        result = dict(
            converged=False, message=str(exc), iterations=recorder.k,
            value=recorder.value.tolist(), policy=policy.tolist(),
            history=[], stats={},
        )
    if isinstance(result, RunResult):
        result = dict(
            converged=bool(result.converged), message=result.message or "converged",
            iterations=result.outer_iterations, value=result.value.tolist(),
            policy=result.policy.tolist(), history=result.history,
            stats=result.work.as_dict(),
        )
    result["method"] = method
    return recorder.finish(mdp, result)


__all__ = ["METHODS", "RESULTS_ONLY_METHODS", "solve"]
