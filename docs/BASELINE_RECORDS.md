# Baseline records and settings

The paper compares LSQN with nine baselines. This public package includes seven baseline implementations: VI, PI, MPI-4, MPI-20, QPI-U, QPI-BT, and QPI-S. The historical AA-I-S-5 and iGMRES-PI implementations are omitted because their redistribution permissions have not been confirmed. They have not been replaced or rerun for this public package.

## What is retained

`benchmarks/` retains the archived macOS and Linux results for all ten methods, including unsuccessful runs, residual histories, runtime tables, and provenance. `scripts/plot_benchmarks.py` and `scripts/plot_runtime_tables.py` regenerate the full comparison from these records without importing either omitted implementation.

The main comparison uses four model families, state counts 512, 1024, 2048, and 4096, discounts 0.99 and 0.999, seed 130, and three timing repetitions. The common starting value is `min(reward) / (1 - gamma)` in every state. The stopping test uses the Bellman residual infinity norm with tolerance `1e-8`. Recorded limits, failures, model identities, and machine environments remain in the archived protocols.

## AA-I-S-5

The recorded implementation was a Python port of the stabilized type-I Anderson implementation in [nonexp_global_aa1](https://github.com/cvxgrp/nonexp_global_aa1), commit `457996cc6cebcaad606d24531214aa40a636125e`, using `codes/algorithms/alg_iter.m`, option `aa1-safe`. The upstream copyright notice names Junzi Zhang (2018).

| Setting | Recorded value |
| --- | --- |
| Memory | 5 |
| Powell regularization parameter | 0.01 |
| Restart parameter | 0.001 |
| Safeguard constant | `1e6` |
| Safeguard parameter (epsilon) | `1e-6` |
| Fallback update | `0.1 * v + 0.9 * T(v)` |
| Outer iteration limit | 30,000 |

The historical implementation retained Powell regularization, restart, and the declining-threshold safeguard. It is not included in this distribution.

Reference: Junzi Zhang, Brendan O'Donoghue, and Stephen Boyd. “Globally Convergent Type-I Anderson Acceleration for Nonsmooth Fixed-Point Iterations.” *SIAM Journal on Optimization* 30(4), 3170–3197 (2020). [DOI](https://doi.org/10.1137/18M1232772).

## iGMRES-PI

The recorded implementation adapted `Callback` and `GMRES_PI` from `iPI_solvers.py` in [SIS_iPI](https://gitlab.ethz.ch/gmatilde/SIS_iPI), commit `1eddae2cd01d961469c349e3931e660796cce7ae`.

| Setting | Recorded value |
| --- | --- |
| Forcing parameter | 0.1 |
| GMRES restart length | 100 |
| Outer iteration limit | 1,000 |
| Absolute GMRES tolerance | 0 |

The adapter converted rewards to costs, supplied dense or sparse policy matrices, and used the common Bellman-residual stopping test. It retained the authors' tolerance conversion: the relative Euclidean GMRES tolerance is 0.1 times the initial policy residual in the infinity norm, divided by the Euclidean norm of the policy reward vector. The original routine and its adapter are not included in this distribution.

References:

- Matilde Gargiani, Robin Sieber, Efe Balta, Dominic Liao-McPherson, and John Lygeros. “Inexact Policy Iteration Methods for Large-Scale Markov Decision Processes” (2024). [arXiv:2404.06136](https://arxiv.org/abs/2404.06136).
- Matilde Gargiani, Dominic Liao-McPherson, Andrea Zanelli, and John Lygeros. “Inexact GMRES Policy Iteration for Large-Scale Markov Decision Processes.” *IFAC-PapersOnLine* 56(2), 11249–11254 (2023). [DOI](https://doi.org/10.1016/j.ifacol.2023.10.316).

## New runs and archived comparisons

In this public package, `--methods all` runs LSQN and the seven included baselines. A complete main run therefore contains 768 solves; the public and scaling suites add 24 and 36 LSQN solves, respectively. The eight-method run does not satisfy the existing summarizer's requirement for a formal ten-method ranking. Historical records are never inserted into a new timing batch to fill the missing methods.

The archived full comparisons retain the original ten-method results and rankings. Replotting those records is different from rerunning the two omitted algorithms. The two original implementations remain in the separate private author archive, not in this package or its distribution history. This project does not offer them for redistribution.
