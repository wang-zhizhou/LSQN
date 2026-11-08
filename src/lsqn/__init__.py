"""Portable LSQN for finite discounted reward-maximization MDPs."""

from .mdp import FiniteMDP
from .options import SolverOptions
from .solver import BACKENDS, prepare_backend, solve

__version__ = "0.1.0rc3"

MDP = FiniteMDP

__all__ = ["MDP", "FiniteMDP", "SolverOptions", "BACKENDS", "prepare_backend", "solve"]
