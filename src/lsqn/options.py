"""Options extracted from the archived implementation. See core-port-notes.txt."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SolverOptions:
    """LSQN options; original CGBOptions field names remain accepted.

    ``initial_scale`` initializes H0 = initial_scale * I, including zero.
    No arbitrary-matrix initial inverse API is provided. Legacy fields are
    retained for source compatibility; the public solver uses full protected
    learning and the residual-sign correction. ``width_fraction`` is tau.
    """

    tol: float = 1e-8
    max_outer: int = 500
    armijo_sigma: float = 1e-4
    backtrack_beta: float = 0.5
    max_backtracks: int = 80
    regularization_scale: float = 1e-8
    descent_fraction: float = 0.5
    step_cap_factor: float = 2.0
    tie_atol: float = 1e-12
    tie_rtol: float = 1e-12
    initial_scale: float = 1.0
    forcing_trigger: float = 0.5
    max_self_secant_enrichments: int = 4
    certified_local_completion: bool = True
    local_completion_residual: float = 1e-3
    local_completion_stable_steps: int = 2
    local_forcing_coefficient: float = 0.5
    local_forcing_exponent: float = 0.5
    projection_alignment_fraction: float = 0.5

    width_fraction: float = 0.1

