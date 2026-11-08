"""Original four paper generators; only the MDP import is changed.

Preserves sparse Queue/Garnet and dense Cluster/Latent storage and formulas.
Original source: scb_siopt_candidate_2026-09-20/baseline/code/src/ptbe/benchmarks.py
SHA256: eca2b4275d6d902f3e5a794c87d706fa88faac84d0fd9dff80472dfe37025a2e
"""
from __future__ import annotations
from typing import List
import numpy as np
import scipy.sparse as sp
from lsqn import MDP as FiniteMDP

def _row_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    z = np.exp(x)
    return z / np.sum(z, axis=1, keepdims=True)


def sparse_garnet(
    n: int = 2000,
    actions: int = 4,
    branching: int = 8,
    gamma: float = 0.99,
    seed: int = 0,
) -> FiniteMDP:
    """Sparse random MDP with fixed row branching for scaling studies."""
    rng = np.random.default_rng(seed)
    branching = min(branching, n)
    transitions: List[sp.csr_matrix] = []
    row_indices = np.repeat(np.arange(n, dtype=np.int64), branching)
    for _ in range(actions):
        columns = np.empty(n * branching, dtype=np.int64)
        data = np.empty(n * branching, dtype=float)
        for state in range(n):
            start = state * branching
            stop = start + branching
            columns[start:stop] = rng.choice(n, size=branching, replace=False)
            data[start:stop] = rng.dirichlet(np.ones(branching))
        transitions.append(
            sp.csr_matrix((data, (row_indices, columns)), shape=(n, n))
        )
    rewards = rng.normal(0.0, 1.0, size=(actions, n))
    return FiniteMDP(transitions, rewards, gamma, name="sparse_garnet")


def latent_switching(
    n: int = 96,
    actions: int = 4,
    latent_rank: int = 5,
    gamma: float = 0.99,
    seed: int = 0,
) -> FiniteMDP:
    """Policies vary by rows but all transitions share a low-dimensional latent range."""
    rng = np.random.default_rng(seed)
    latent = rng.dirichlet(0.35 * np.ones(n), size=latent_rank)
    state_phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base_features = np.column_stack(
        [
            np.sin((j + 1) * state_phase) + 0.2 * rng.normal(size=n)
            for j in range(latent_rank)
        ]
    )
    transitions: List[np.ndarray] = []
    rewards = np.zeros((actions, n), dtype=float)
    for a in range(actions):
        rotation = np.roll(base_features, a, axis=1)
        logits = 1.3 * rotation + 0.25 * rng.normal(size=(n, latent_rank))
        weights = _row_softmax(logits)
        transitions.append(weights @ latent)
        rewards[a] = (
            0.28 * np.cos(state_phase - 2.0 * np.pi * a / actions)
            + 0.08 * np.sin(3.0 * state_phase + a)
            + 0.03 * rng.normal(size=n)
        )
    return FiniteMDP(transitions, rewards, gamma, name="latent_switching")


def nearly_decomposable(
    n: int = 120,
    actions: int = 4,
    clusters: int = 6,
    leakage: float = 0.01,
    gamma: float = 0.99,
    seed: int = 0,
) -> FiniteMDP:
    rng = np.random.default_rng(seed)
    labels = np.arange(n) * clusters // n
    transitions: List[np.ndarray] = []
    phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rewards = np.zeros((actions, n), dtype=float)
    for a in range(actions):
        p = np.zeros((n, n), dtype=float)
        for i in range(n):
            local = np.flatnonzero(labels == labels[i])
            outside = np.flatnonzero(labels != labels[i])
            local_weights = rng.dirichlet(0.4 * np.ones(local.size))
            p[i, local] = (1.0 - leakage) * local_weights
            if outside.size:
                out_idx = rng.choice(outside, size=min(3, outside.size), replace=False)
                p[i, out_idx] += leakage / out_idx.size
            # Action-specific local drift with the same cluster subspace.
            target = local[(np.where(local == i)[0][0] + a + 1) % local.size]
            p[i] *= 0.92
            p[i, target] += 0.08
            p[i] /= p[i].sum()
        transitions.append(p)
        rewards[a] = -0.05 * labels + 0.15 * np.cos(phase + a * np.pi / 3.0) - 0.025 * a
    rewards += 0.01 * rng.normal(size=rewards.shape)
    return FiniteMDP(transitions, rewards, gamma, name="nearly_decomposable")


def controlled_queue(
    n: int = 200,
    actions: int = 4,
    gamma: float = 0.99,
    arrival: float = 0.30,
    seed: int = 0,
) -> FiniteMDP:
    """Sparse birth-death queue with controllable service probability."""
    rng = np.random.default_rng(seed)
    transitions: List[sp.csr_matrix] = []
    rewards = np.zeros((actions, n), dtype=float)
    states = np.arange(n, dtype=float)
    for a in range(actions):
        service = 0.08 + 0.18 * a
        rows, cols, data = [], [], []
        for i in range(n):
            up = min(n - 1, i + 1)
            down = max(0, i - 1)
            p_up = arrival * (1.0 - service)
            p_down = (1.0 - arrival) * service
            p_stay = 1.0 - p_up - p_down
            bucket = {}
            for j, prob in ((up, p_up), (down, p_down), (i, p_stay)):
                bucket[j] = bucket.get(j, 0.0) + prob
            for j, prob in bucket.items():
                rows.append(i)
                cols.append(j)
                data.append(prob)
        transitions.append(sp.csr_matrix((data, (rows, cols)), shape=(n, n)))
        rewards[a] = -0.015 * states - 0.11 * (a ** 1.4) + 0.002 * rng.normal(size=n)
    return FiniteMDP(transitions, rewards, gamma, name="controlled_queue")


FAMILIES = {"queue": controlled_queue, "cluster": nearly_decomposable,
            "garnet": sparse_garnet, "latent": latent_switching}
