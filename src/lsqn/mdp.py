from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


Matrix = Union[np.ndarray, sp.csr_matrix]


def _is_sparse_matrix(x: Matrix) -> bool:
    return sp.issparse(x)


@dataclass
class FiniteMDP:
    """Finite discounted reward-maximization MDP.

    `transitions[a]` is an n-by-n row-stochastic matrix and `rewards[a, i]` is
    the expected one-step reward for action `a` in state `i`.
    """

    transitions: Sequence[Matrix]
    rewards: np.ndarray
    gamma: float
    name: str = "mdp"

    def __post_init__(self) -> None:
        self.transitions = [
            p.tocsr().astype(float) if sp.issparse(p) else np.asarray(p, dtype=float)
            for p in self.transitions
        ]
        self.rewards = np.asarray(self.rewards, dtype=float)
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie in (0, 1)")
        if not self.transitions:
            raise ValueError("at least one action is required")
        n = self.transitions[0].shape[0]
        if any(p.shape != (n, n) for p in self.transitions):
            raise ValueError("all transition matrices must be square with the same shape")
        if self.rewards.shape != (len(self.transitions), n):
            raise ValueError("rewards must have shape (n_actions, n_states)")
        for p in self.transitions:
            rows = np.asarray(p.sum(axis=1)).ravel()
            if np.min(p.data if sp.issparse(p) else p) < -1e-13:
                raise ValueError("transition entries must be nonnegative")
            if not np.allclose(rows, 1.0, atol=1e-10, rtol=1e-10):
                raise ValueError("transition rows must sum to one")

    @property
    def n_states(self) -> int:
        return self.transitions[0].shape[0]

    @property
    def n_actions(self) -> int:
        return len(self.transitions)

    @property
    def sparse(self) -> bool:
        return any(sp.issparse(p) for p in self.transitions)

    def action_values(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        return np.vstack(
            [self.rewards[a] + self.gamma * self.transitions[a].dot(v) for a in range(self.n_actions)]
        )

    def bellman(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = self.action_values(v)
        policy = np.argmax(q, axis=0).astype(np.int64)
        return q[policy, np.arange(self.n_states)], policy

    def policy_reward(self, policy: np.ndarray) -> np.ndarray:
        policy = np.asarray(policy, dtype=np.int64)
        return self.rewards[policy, np.arange(self.n_states)]

    def policy_matvec(self, policy: np.ndarray, x: np.ndarray) -> np.ndarray:
        policy = np.asarray(policy, dtype=np.int64)
        x = np.asarray(x, dtype=float)
        out = np.empty(self.n_states, dtype=float)
        for a in range(self.n_actions):
            idx = np.flatnonzero(policy == a)
            if idx.size:
                out[idx] = np.asarray(self.transitions[a][idx, :].dot(x)).ravel()
        return out

    def policy_rmatvec(self, policy: np.ndarray, x: np.ndarray) -> np.ndarray:
        policy = np.asarray(policy, dtype=np.int64)
        x = np.asarray(x, dtype=float)
        out = np.zeros(self.n_states, dtype=float)
        for a in range(self.n_actions):
            idx = np.flatnonzero(policy == a)
            if idx.size:
                if sp.issparse(self.transitions[a]):
                    out += self.transitions[a][idx, :].T.dot(x[idx])
                else:
                    out += self.transitions[a][idx, :].T @ x[idx]
        return np.asarray(out).ravel()

    def policy_matmat(self, policy: np.ndarray, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            return self.policy_matvec(policy, x)
        out = np.empty((self.n_states, x.shape[1]), dtype=float)
        for a in range(self.n_actions):
            idx = np.flatnonzero(policy == a)
            if idx.size:
                out[idx, :] = np.asarray(self.transitions[a][idx, :].dot(x))
        return out

    def policy_rows_matmat(
        self,
        policy: np.ndarray,
        rows: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """Apply selected rows of a policy transition matrix to one or more columns."""
        policy = np.asarray(policy, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        x = np.asarray(x, dtype=float)
        was_vector = x.ndim == 1
        if was_vector:
            x = x[:, None]
        out = np.empty((rows.size, x.shape[1]), dtype=float)
        for a in range(self.n_actions):
            local = np.flatnonzero(policy[rows] == a)
            if local.size:
                global_rows = rows[local]
                out[local, :] = np.asarray(self.transitions[a][global_rows, :].dot(x))
        return out[:, 0] if was_vector else out

    def policy_matrix(self, policy: np.ndarray) -> Matrix:
        policy = np.asarray(policy, dtype=np.int64)
        if self.sparse:
            out = sp.csr_matrix((self.n_states, self.n_states), dtype=float)
            for a in range(self.n_actions):
                mask = (policy == a).astype(float)
                if np.any(mask):
                    out = out + sp.diags(mask, format="csr").dot(self.transitions[a])
            return out.tocsr()
        out = np.empty((self.n_states, self.n_states), dtype=float)
        for a in range(self.n_actions):
            idx = np.flatnonzero(policy == a)
            if idx.size:
                out[idx] = self.transitions[a][idx]
        return out

    def policy_submatrix(
        self,
        policy: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> Matrix:
        """Extract P_pi[rows, cols] without assembling the full policy matrix."""
        policy = np.asarray(policy, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self.sparse:
            pieces = []
            for i in rows:
                pieces.append(self.transitions[int(policy[i])][int(i), cols])
            return sp.vstack(pieces, format="csr")
        out = np.empty((rows.size, cols.size), dtype=float)
        for a in range(self.n_actions):
            local = np.flatnonzero(policy[rows] == a)
            if local.size:
                out[local, :] = self.transitions[a][np.ix_(rows[local], cols)]
        return out

    def policy_jacobian_block(self, policy: np.ndarray, active: np.ndarray) -> Matrix:
        active = np.asarray(active, dtype=np.int64)
        paa = self.policy_submatrix(policy, active, active)
        if sp.issparse(paa):
            return sp.eye(active.size, format="csr") - self.gamma * paa
        return np.eye(active.size) - self.gamma * paa

    def policy_jacobian(self, policy: np.ndarray) -> Matrix:
        p = self.policy_matrix(policy)
        if sp.issparse(p):
            return sp.eye(self.n_states, format="csr") - self.gamma * p
        return np.eye(self.n_states) - self.gamma * p

    def common_lower_bound(self) -> np.ndarray:
        lower = float(np.min(self.rewards)) / (1.0 - self.gamma)
        return np.full(self.n_states, lower, dtype=float)

    def shift_to_subsolution(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).copy()
        tv, _ = self.bellman(v)
        shift = max(0.0, float(np.max(v - tv)) / (1.0 - self.gamma))
        return v - shift

    def evaluate_policy(self, policy: np.ndarray) -> np.ndarray:
        j = self.policy_jacobian(policy)
        b = self.policy_reward(policy)
        if sp.issparse(j):
            return np.asarray(spla.spsolve(j, b)).ravel()
        return np.linalg.solve(j, b)

    def reference_solution(self, tol: float = 1e-13, max_iter: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a high-accuracy reference via exact policy iteration."""
        v = self.common_lower_bound()
        old_policy = None
        for _ in range(max_iter):
            _, policy = self.bellman(v)
            v_new = self.evaluate_policy(policy)
            tv_new, policy_new = self.bellman(v_new)
            if np.linalg.norm(tv_new - v_new, np.inf) <= tol:
                return v_new, policy_new
            if old_policy is not None and np.array_equal(policy, old_policy):
                return v_new, policy_new
            old_policy = policy.copy()
            v = v_new
        raise RuntimeError("reference policy iteration did not converge")
