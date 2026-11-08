"""VI, exact PI and uniform-prior QPI from the local equation-based implementations.

The initial-value handling matches final run_study.py: copy the supplied vector,
without the earlier per-baseline subsolution shift. See baselines-port-notes.txt.
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from lsqn.mdp import FiniteMDP, Matrix
from ._metrics import RunResult, WorkCounter
from ._trace import noop


def _bellman(mdp: FiniteMDP, v: np.ndarray, work: WorkCounter) -> Tuple[np.ndarray, np.ndarray]:
    work.bellman(mdp.n_actions)
    return mdp.bellman(v)



def _initial_value(
    mdp: FiniteMDP,
    v0: Optional[np.ndarray],
    enforce_subsolution: bool = True,
) -> np.ndarray:
    if v0 is None:
        return mdp.common_lower_bound()
    value = np.asarray(v0, dtype=float)
    return value.copy()



def _finish(
    method: str,
    mdp: FiniteMDP,
    v: np.ndarray,
    policy: np.ndarray,
    k: int,
    residual: float,
    start: float,
    work: WorkCounter,
    history: List[Dict[str, float]],
    converged: bool,
    message: str = "",
) -> RunResult:
    return RunResult(
        method=method,
        converged=converged,
        value=v,
        policy=policy,
        outer_iterations=k,
        residual=float(residual),
        runtime_s=time.perf_counter() - start,
        work=work,
        history=history,
        message=message,
    )



def solve_vi(
    mdp: FiniteMDP,
    tol: float = 1e-10,
    max_outer: int = 100000,
    v0: Optional[np.ndarray] = None,
    observe=noop,
) -> RunResult:
    method = "VI"
    start = time.perf_counter()
    work = WorkCounter()
    history: List[Dict[str, float]] = []
    v = _initial_value(mdp, v0)
    policy = np.zeros(mdp.n_states, dtype=int)
    previous_policy: Optional[np.ndarray] = None
    for k in range(max_outer + 1):
        tv, policy = _bellman(mdp, v, work)
        residual = float(np.linalg.norm(tv - v, np.inf))
        observe(v, v-tv, k, history[-1] if k and history else None)
        switched = 0 if previous_policy is None else int(np.count_nonzero(policy != previous_policy))
        history.append({"iteration": k, "residual": residual, "policy_switches": float(switched)})
        if residual <= tol:
            return _finish(method, mdp, v, policy, k, residual, start, work, history, True)
        v = tv
        previous_policy = policy.copy()
    return _finish(method, mdp, v, policy, max_outer, residual, start, work, history, False, "max_outer")



def solve_pi(
    mdp: FiniteMDP,
    tol: float = 1e-10,
    max_outer: int = 1000,
    v0: Optional[np.ndarray] = None,
    observe=noop,
) -> RunResult:
    method = "PI"
    start = time.perf_counter()
    work = WorkCounter()
    history: List[Dict[str, float]] = []
    v = _initial_value(mdp, v0)
    policy = np.zeros(mdp.n_states, dtype=int)
    previous_policy: Optional[np.ndarray] = None
    for k in range(max_outer + 1):
        tv, policy = _bellman(mdp, v, work)
        residual = float(np.linalg.norm(tv - v, np.inf))
        observe(v, v-tv, k, history[-1] if k and history else None)
        switched = 0 if previous_policy is None else int(np.count_nonzero(policy != previous_policy))
        history.append({"iteration": k, "residual": residual, "policy_switches": float(switched)})
        if residual <= tol:
            return _finish(method, mdp, v, policy, k, residual, start, work, history, True)
        j = mdp.policy_jacobian(policy)
        b = mdp.policy_reward(policy)
        if sp.issparse(j):
            v = np.asarray(spla.spsolve(j, b)).ravel()
            work.direct_solve_flops += float(max(1, j.nnz) ** 1.5)
        else:
            v = np.linalg.solve(j, b)
            work.direct_solve_flops += float(mdp.n_states ** 3) / 3.0
        previous_policy = policy.copy()
    return _finish(method, mdp, v, policy, max_outer, residual, start, work, history, False, "max_outer")



class _PriorInverse:
    def __init__(self, mdp: FiniteMDP, prior: str):
        self.mdp = mdp
        self.prior = prior
        self.uniform = prior == "uniform"
        if self.uniform:
            self.p = None
            self.g = None
        elif prior == "action_average":
            p = sum(mdp.transitions) * (1.0 / mdp.n_actions)
            self.p = p
            j = sp.eye(mdp.n_states, format="csc") - mdp.gamma * p if sp.issparse(p) else np.eye(mdp.n_states) - mdp.gamma * p
            self.g = spla.factorized(j) if sp.issparse(j) else np.linalg.inv(j)
        else:
            raise ValueError("unknown QPI prior")

    def p_dot(self, x: np.ndarray) -> np.ndarray:
        if self.uniform:
            return np.full_like(x, float(np.mean(x)))
        return np.asarray(self.p.dot(x)).ravel()

    def g_dot(self, x: np.ndarray) -> np.ndarray:
        if self.uniform:
            return x + self.mdp.gamma / (1.0 - self.mdp.gamma) * float(np.mean(x)) * np.ones_like(x)
        if callable(self.g):
            return np.asarray(self.g(x)).ravel()
        return self.g @ x

    def gt_dot(self, x: np.ndarray) -> np.ndarray:
        if self.uniform:
            return x + self.mdp.gamma / (1.0 - self.mdp.gamma) * float(np.sum(x)) / self.mdp.n_states * np.ones_like(x)
        if callable(self.g):
            # Sparse action-average prior is uncommon in the small QPI benchmark.  Use an
            # explicit transpose solve here to keep the formula exact.
            j = sp.eye(self.mdp.n_states, format="csc") - self.mdp.gamma * self.p
            return np.asarray(spla.spsolve(j.T, x)).ravel()
        return self.g.T @ x



def solve_qpi(
    mdp: FiniteMDP,
    prior: str = "uniform",
    safeguard: str = "paper",
    tol: float = 1e-10,
    max_outer: int = 100000,
    v0: Optional[np.ndarray] = None,
    observe=noop,
) -> RunResult:
    """Safeguarded QPI following Theorem 3.1 of Kolarijani--Esfahani.

    The implementation uses the exact rank-one inverse formula with either the uniform prior
    or the action-average transition prior.  `paper` uses theta_{k+1} <= gamma^{k+1} theta_0.
    """
    method = "QPI-U" if prior == "uniform" else "QPI-Pmu"
    start = time.perf_counter()
    work = WorkCounter()
    history: List[Dict[str, float]] = []
    v = _initial_value(mdp, v0)
    prior_inv = _PriorInverse(mdp, prior)
    if prior == "action_average":
        work.direct_solve_flops += float(mdp.n_states ** 3) / 3.0
    tv, policy = _bellman(mdp, v, work)
    theta0 = float(np.linalg.norm(tv - v, np.inf))
    previous_policy: Optional[np.ndarray] = None
    for k in range(max_outer + 1):
        d = tv - v
        residual = float(np.linalg.norm(d, np.inf))
        observe(v, v-tv, k, history[-1] if k and history else None)
        switched = 0 if previous_policy is None else int(np.count_nonzero(policy != previous_policy))
        if residual <= tol:
            history.append({"iteration": k, "residual": residual, "safeguarded": 0.0, "policy_switches": float(switched)})
            return _finish(method, mdp, v, policy, k, residual, start, work, history, True)
        if prior == "action_average":
            work.prior_matvecs += 4.0
        reward = mdp.policy_reward(policy)
        w = tv - reward - mdp.gamma * prior_inv.p_dot(v)
        wcheck = prior_inv.g_dot(w)
        u = v - float(np.mean(v))
        ucheck = prior_inv.gt_dot(u)
        denom = float(u @ (v - wcheck))
        q = prior_inv.g_dot(d)
        if abs(float(u @ v)) > 1e-28 and abs(denom) > 1e-28:
            q = q + (float(ucheck @ d) / denom) * wcheck
        candidate = v + q
        tc, pc = _bellman(mdp, candidate, work)
        theta_c = float(np.linalg.norm(tc - candidate, np.inf))
        if safeguard == "paper":
            accepted = theta_c <= (mdp.gamma ** (k + 1)) * theta0 * (1.0 + 1e-12)
        elif safeguard == "local":
            accepted = theta_c <= mdp.gamma * residual * (1.0 + 1e-12)
        else:
            raise ValueError("unknown safeguard")
        history.append({
            "iteration": k,
            "residual": residual,
            "candidate_residual": theta_c,
            "safeguarded": 0.0 if accepted else 1.0,
            "policy_switches": float(switched),
        })
        previous_policy = policy.copy()
        if accepted:
            v, tv, policy = candidate, tc, pc
        else:
            v = tv
            tv, policy = _bellman(mdp, v, work)
    return _finish(method, mdp, v, policy, max_outer, residual, start, work, history, False, "max_outer")

