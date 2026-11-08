"""Solve a two-state MDP without the experiment harness."""
import numpy as np
from lsqn import MDP, solve

model = MDP(
    transitions=[np.array([[0.8, 0.2], [0.1, 0.9]]),
                 np.array([[0.4, 0.6], [0.3, 0.7]])],
    rewards=np.array([[1.0, 0.0], [0.2, 0.7]]), gamma=0.9,
)
result = solve(model, v0=model.common_lower_bound())
print("Converged:", result["converged"])
print("Bellman residual (infinity norm):", result["residual"])
print("Outer iterations:", result["iterations"])
print("Learning queries:", result["stats"]["enrichments"])
print("Value:", result["value"])
if not result["converged"]:
    raise SystemExit(1)
