"""Internal compatibility helpers extracted without changing their formulas.

Only the grouped sign envelope is used by the public solver. These helpers keep
legacy branches of the source solver intact; no old standalone solver is copied.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from .mdp import FiniteMDP


@dataclass
class WorkCounter:
    bellman_backups: float = 0.0
    action_matvecs: float = 0.0
    policy_matvecs: float = 0.0
    policy_rmatvecs: float = 0.0
    gmres_matvecs: float = 0.0
    memory_image_matvecs: float = 0.0
    prior_matvecs: float = 0.0
    direct_solve_flops: float = 0.0
    block_solve_flops: float = 0.0
    lp_calls: int = 0

    def bellman(self, n_actions: int) -> None:
        self.bellman_backups += 1.0
        self.action_matvecs += float(n_actions)

    @property
    def operator_products(self) -> float:
        return (
            self.action_matvecs
            + self.policy_matvecs
            + self.policy_rmatvecs
            + self.gmres_matvecs
            + self.memory_image_matvecs
            + self.prior_matvecs
        )

    def as_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["operator_products"] = self.operator_products
        return d



def _active_actions(q: np.ndarray, atol: float, rtol: float) -> List[np.ndarray]:
    best = np.max(q, axis=0)
    scale = np.maximum(1.0, np.abs(best))
    active: List[np.ndarray] = []
    for i in range(q.shape[1]):
        threshold = atol + rtol * scale[i]
        active.append(np.flatnonzero(best[i] - q[:, i] <= threshold))
    return active



def _minimum_norm_generalized_gradient(
    mdp: FiniteMDP,
    f: np.ndarray,
    policy: np.ndarray,
    active: Sequence[np.ndarray],
    work: WorkCounter,
) -> Tuple[np.ndarray, bool]:
    """Project zero onto a computable Clarke-gradient polytope.

    At a smooth point this is simply (I-gamma P_pi)^T f.  At an action tie,
    the convex weights choose a stochastic mixture of the tied transition
    rows and minimize the Euclidean norm of J_lambda^T f.
    """

    if all(actions.size == 1 for actions in active):
        work.policy_rmatvecs += 1.0
        return f - mdp.gamma * mdp.policy_rmatvec(policy, f), True

    pairs = [(i, int(a)) for i, actions in enumerate(active) for a in actions]
    m = len(pairs)
    columns = np.empty((mdp.n_states, m), dtype=float)
    for j, (i, action) in enumerate(pairs):
        row = mdp.transitions[action][i, :]
        columns[:, j] = -mdp.gamma * f[i] * np.asarray(
            row.toarray() if sp.issparse(row) else row
        ).ravel()

    x0 = np.zeros(m, dtype=float)
    constraints = []
    offset = 0
    for actions in active:
        size = actions.size
        x0[offset : offset + size] = 1.0 / size
        selector = np.zeros(m, dtype=float)
        selector[offset : offset + size] = 1.0
        constraints.append({"type": "eq", "fun": lambda x, s=selector: float(s @ x - 1.0),
                            "jac": lambda x, s=selector: s})
        offset += size

    def objective(weights: np.ndarray) -> float:
        g = f + columns @ weights
        return 0.5 * float(g @ g)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return columns.T @ (f + columns @ weights)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=[(0.0, 1.0)] * m,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-13, "maxiter": 500, "disp": False},
    )
    weights = result.x.copy() if result.success and np.all(np.isfinite(result.x)) else x0
    # SLSQP enforces the simplex constraints only up to its feasibility
    # tolerance.  Repair every block before forming the returned envelope
    # element, so the subsequent support-function certificate is rigorous for
    # the represented floating-point weights.
    offset = 0
    for actions in active:
        size = actions.size
        block = np.maximum(weights[offset : offset + size], 0.0)
        total = float(np.sum(block))
        if not np.isfinite(total) or total <= np.finfo(float).tiny:
            block = np.full(size, 1.0 / size)
        else:
            block = block / total
        weights[offset : offset + size] = block
        offset += size
    return f + columns @ weights, bool(result.success)



class Envelope:
    def __init__(self, mdp, active, policy, stats):
        self.mdp, self.stats = mdp, stats
        self.active, self.policy = active, policy
        self.states = np.repeat(np.arange(mdp.n_states), [len(a) for a in active])
        self.offsets = np.r_[0, np.cumsum([len(a) for a in active])[:-1]]
        self.p = mdp.policy_matrix(policy)
        if len(self.states) == mdp.n_states:
            self.rows = self.p
        else:
            rows = [mdp.transitions[int(a)][i:i+1, :]
                    for i, actions in enumerate(active) for a in actions]
            self.rows = sp.vstack(rows).tocsr() if mdp.sparse else np.vstack(rows)

    def j(self, x):
        self.stats['policy_products'] += 1
        return x - self.mdp.gamma * self.p.dot(x)

    def pairing(self, f, x, upper=True):
        self.stats['support_row_products'] += len(self.states)
        values = f[self.states] * (x[self.states] - self.mdp.gamma * self.rows.dot(x))
        reduce = np.maximum.reduceat if upper else np.minimum.reduceat
        return float(reduce(values, self.offsets).sum())

    def gradient(self, f):
        self.stats['gradient_requests'] += 1
        if len(self.states) == self.mdp.n_states:
            self.stats['policy_transpose_products'] += 1
            return f - self.mdp.gamma * self.p.T.dot(f)
        self.stats['qp_calls'] += 1
        # Keep the original projection implementation, including feasibility
        # repair. Its outcome must pass the alignment test below.
        g, _ = _minimum_norm_generalized_gradient(
            self.mdp, f, self.policy, self.active, WorkCounter())
        return g



