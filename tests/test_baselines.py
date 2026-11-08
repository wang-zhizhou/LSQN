"""Small numerical checks for the seven included comparison baselines."""

import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from lsqn import MDP
from lsqn_baselines import METHODS, solve
from lsqn_baselines import _comparators


def model(sparse=False):
    rng = np.random.default_rng(130)
    n = 16
    transitions = []
    for _ in range(3):
        p = rng.dirichlet(np.ones(n), size=n)
        if sparse:
            p[p < 0.065] = 0.0
            p[np.arange(n), np.arange(n)] += 1.0
            p /= p.sum(axis=1, keepdims=True)
        transitions.append(sp.csr_matrix(p) if sparse else p)
    return MDP(transitions, rng.normal(size=(3, n)), 0.9, name="baseline-unit")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def test_all_included_dense_and_sparse_against_exact_pi(self):
        self.assertEqual(len(METHODS), 7)
        for sparse in (False, True):
            mdp = model(sparse)
            reference, _ = mdp.reference_solution(tol=2e-12)
            for method in METHODS:
                with self.subTest(sparse=sparse, method=method):
                    result = solve(method, mdp, mdp.common_lower_bound())
                    self.assertTrue(result["converged"], result["message"])
                    value = np.asarray(result["value"])
                    tv, policy = mdp.bellman(value)
                    residual = float(np.linalg.norm(value-tv, np.inf))
                    self.assertLessEqual(residual, 1e-8)
                    self.assertEqual(result["residual"], residual)
                    self.assertLessEqual(np.linalg.norm(value-reference, np.inf),
                                         1e-8/(1-mdp.gamma)+2e-10)
                    np.testing.assert_array_equal(result["policy"], policy)
                    self.assertEqual(result["trace"][-1]["k"], result["iterations"])
                    self.assertEqual(result["trace"][-1]["residual_inf"], residual)

    def test_no_initial_shift_and_timeout_propagation(self):
        mdp = model()
        initial = np.linspace(-17.0, 25.0, mdp.n_states)
        for method in METHODS:
            with self.subTest(method=method):
                def callback(record):
                    self.assertEqual(record["event"], "initial")
                    self.assertEqual(record["k"], 0)
                    np.testing.assert_array_equal(record["value"], initial)
                    raise TimeoutError("controlled initial interruption")
                with self.assertRaisesRegex(TimeoutError, "controlled initial"):
                    solve(method, mdp, initial, callback=callback)

    def test_callbacks_cover_initial_and_accepted_values(self):
        mdp = model()
        for method in METHODS:
            with self.subTest(method=method):
                records = []
                def callback(record):
                    tv, _ = mdp.bellman(record["value"])
                    self.assertEqual(float(np.linalg.norm(record["value"]-tv, np.inf)),
                                     record["residual_inf"])
                    records.append((record["event"], record["k"]))
                    record["value"][:] = 1e100
                result = solve(method, mdp, callback=callback)
                self.assertTrue(result["converged"], result["message"])
                self.assertEqual(records[0], ("initial", 0))
                self.assertEqual(records[-1], ("finished", result["iterations"]))
                self.assertEqual([k for event, k in records if event == "accepted"],
                                 list(range(1, result["iterations"]+1)))



    def test_qpi_numerical_failure_is_retained(self):
        mdp = model()
        initial = mdp.common_lower_bound()
        with patch.object(_comparators, "secant_transition_update",
                          side_effect=ArithmeticError("ill-conditioned Woodbury update")):
            result = solve("QPI-S", mdp, initial)
        self.assertFalse(result["converged"])
        self.assertEqual(result["message"], "ill-conditioned Woodbury update")
        self.assertEqual(result["iterations"], 0)
        np.testing.assert_array_equal(result["value"], initial)
        self.assertEqual(result["stats"]["dense_updates"], 0)


if __name__ == "__main__":
    unittest.main()
