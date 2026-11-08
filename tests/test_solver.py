"""Small numerical checks; no archived workspace or large timing run required."""

import unittest

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from lsqn import MDP, SolverOptions, solve
from lsqn._core import InverseModel


def small_mdp(sparse=False, gamma=0.95, n=16):
    rng = np.random.default_rng(8301)
    transitions = []
    for _ in range(3):
        p = rng.uniform(size=(n, n))
        if sparse:
            p[p < 0.65] = 0.0
            p[np.arange(n), np.arange(n)] += 1.0
        p /= p.sum(axis=1, keepdims=True)
        transitions.append(sp.csr_matrix(p) if sparse else p)
    return MDP(transitions, rng.normal(size=(3, n)), gamma, name="unit-mdp")


def exact_pi(mdp):
    """Independent dense policy evaluation, even for a sparse test model."""
    value = mdp.common_lower_bound()
    for _ in range(200):
        action_values = np.vstack([
            mdp.rewards[a] + mdp.gamma * p.dot(value)
            for a, p in enumerate(mdp.transitions)
        ])
        policy = np.argmax(action_values, axis=0)
        rows = np.vstack([
            (mdp.transitions[a].getrow(i).toarray().ravel()
             if sp.issparse(mdp.transitions[a]) else mdp.transitions[a][i])
            for i, a in enumerate(policy)
        ])
        reward = mdp.rewards[policy, np.arange(mdp.n_states)]
        value = np.linalg.solve(np.eye(mdp.n_states) - mdp.gamma * rows, reward)
        tv, _ = mdp.bellman(value)
        if np.linalg.norm(value-tv, np.inf) <= 2e-12:
            return value
    raise AssertionError("small reference PI did not converge")


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def assert_certificates(self, mdp, result, options):
        self.assertEqual(len(result["history"]), result["iterations"])
        self.assertEqual(result["stats"]["enrichments"],
                         sum(h["enrichments"] for h in result["history"]))
        self.assertEqual([t["k"] for t in result["trace"]],
                         list(range(result["iterations"] + 1)))
        for h in result["history"]:
            self.assertLessEqual(h["enrichments"], mdp.n_states)
            self.assertLessEqual(h["forcing"], h["target"] * (1 + 1e-8))
            self.assertFalse(h["capped"])
            r2 = h["residual2_before"] ** 2
            self.assertLessEqual(h["slope"],
                                 -options.descent_fraction * r2 * (1-1e-10))
            self.assertLessEqual(
                0.5 * h["residual2_after"] ** 2,
                0.5 * r2 + options.armijo_sigma * h["alpha"] * h["slope"]
                + 8 * np.finfo(float).eps * r2,
            )
            grid = np.log(h["alpha"]) / np.log(options.backtrack_beta)
            self.assertAlmostEqual(grid, round(grid), places=11)

    def test_dense_and_sparse_against_exact_pi(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                mdp = small_mdp(sparse=sparse, gamma=0.99)
                options = SolverOptions()
                result = solve(mdp, options, v0=mdp.common_lower_bound())
                self.assertTrue(result["converged"], result["message"])
                value = np.asarray(result["value"])
                tv, _ = mdp.bellman(value)
                residual = float(np.linalg.norm(value-tv, np.inf))
                self.assertLessEqual(residual, options.tol)
                self.assertAlmostEqual(residual, result["residual"], places=13)
                reference = exact_pi(mdp)
                self.assertLessEqual(np.linalg.norm(value-reference, np.inf),
                                     options.tol / (1-mdp.gamma) + 2e-10)
                self.assert_certificates(mdp, result, options)

    def test_singular_zero_initial_inverse(self):
        mdp = small_mdp(sparse=True, n=12)
        options = SolverOptions(initial_scale=0.0)
        result = solve(mdp, options, v0=mdp.common_lower_bound())
        self.assertTrue(result["converged"], result["message"])
        self.assertGreater(result["history"][0]["enrichments"], 0)
        self.assertLessEqual(np.linalg.norm(np.asarray(result["value"])-exact_pi(mdp), np.inf),
                             options.tol/(1-mdp.gamma) + 2e-10)
        self.assert_certificates(mdp, result, options)

    def test_sign_correction_and_backtracking_certificates(self):
        rng = np.random.default_rng(8)
        n = 8
        transitions = [rng.dirichlet(0.1*np.ones(n), size=n) for _ in range(4)]
        mdp = MDP(transitions, rng.normal(size=(4, n)), 0.99)
        v0 = 100*rng.normal(size=n)
        options = SolverOptions()
        result = solve(mdp, options, v0=v0)
        self.assertTrue(result["converged"], result["message"])
        self.assertGreater(result["stats"]["sign_corrections"], 0)
        self.assertGreater(result["stats"]["backtracks"], 0)
        self.assertTrue(any(h["mu"] > 0 for h in result["history"]))
        self.assert_certificates(mdp, result, options)

    def test_protected_orthogonal_secants_survive_dense_conversion(self):
        rng = np.random.default_rng(2718)
        n = 12
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        jacobian = np.eye(n) + 0.05 * rng.normal(size=(n, n))
        exact_images = np.linalg.solve(jacobian, q)
        model = InverseModel(n, beta=0.0, fast=True)
        for j in range(n):
            model.update(exact_images[:, j]-model.apply(q[:, j]), q[:, j])
            for previous in range(j+1):
                np.testing.assert_allclose(
                    model.apply(q[:, previous]), exact_images[:, previous],
                    atol=3e-13, rtol=3e-13,
                )
        self.assertEqual(model.conversions, 1)

    def test_callback_tracks_values_without_exposing_solver_arrays(self):
        mdp = small_mdp()
        records = []
        def callback(record):
            tv, _ = mdp.bellman(record["value"])
            self.assertAlmostEqual(float(np.linalg.norm(record["value"]-tv, np.inf)),
                                   record["residual_inf"], places=12)
            records.append((record["event"], record["k"], record["accepted_step"]))
            record["value"][:] = 1e100
        result = solve(mdp, callback=callback)
        self.assertTrue(result["converged"])
        self.assertEqual(records[0][:2], ("initial", 0))
        self.assertEqual(records[-1][:2], ("finished", result["iterations"]))
        accepted = [r for r in records if r[0] == "accepted"]
        self.assertEqual(len(accepted), result["iterations"])
        self.assertEqual([r[2] for r in accepted], result["history"])
        self.assertIsNone(records[-1][2])

    def test_timeout_callback_exposes_last_accepted_value(self):
        mdp = small_mdp()
        last = {}
        def callback(record):
            last.clear()
            last.update(record)
            if record["event"] == "accepted":
                raise TimeoutError("test interruption")
        with self.assertRaisesRegex(TimeoutError, "test interruption"):
            solve(mdp, callback=callback)
        self.assertEqual(last["k"], 1)
        tv, _ = mdp.bellman(last["value"])
        self.assertAlmostEqual(float(np.linalg.norm(last["value"]-tv, np.inf)),
                               last["residual_inf"], places=12)

    def test_iteration_limit_records_accepted_step(self):
        mdp = small_mdp()
        result = solve(mdp, SolverOptions(max_outer=1))
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(result["message"], "iteration limit")
        self.assertEqual(result["trace"][-1]["k"], 1)
        tv, _ = mdp.bellman(np.asarray(result["value"]))
        self.assertAlmostEqual(float(np.linalg.norm(np.asarray(result["value"])-tv, np.inf)),
                               result["residual"], places=13)

    def test_zero_residual_stops_without_learning(self):
        mdp = MDP([np.eye(3)], np.zeros((1, 3)), 0.9)
        result = solve(mdp)
        self.assertTrue(result["converged"])
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(result["stats"]["enrichments"], 0)
        self.assertEqual(result["trace"][0]["residual_inf"], 0.0)


if __name__ == "__main__":
    unittest.main()
