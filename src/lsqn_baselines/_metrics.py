from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np


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


@dataclass
class RunResult:
    method: str
    converged: bool
    value: np.ndarray
    policy: np.ndarray
    outer_iterations: int
    residual: float
    runtime_s: float
    work: WorkCounter
    history: List[Dict[str, float]] = field(default_factory=list)
    message: str = ""

    def summary(self) -> Dict[str, float]:
        out = {
            "method": self.method,
            "converged": bool(self.converged),
            "outer_iterations": int(self.outer_iterations),
            "residual": float(self.residual),
            "runtime_s": float(self.runtime_s),
            "message": self.message,
        }
        out.update(self.work.as_dict())
        return out
