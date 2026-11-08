"""The block32 backend preserves every update and the paper's acceptance rules."""

import unittest

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from lsqn import BACKENDS, MDP, SolverOptions, prepare_backend, solve
from lsqn._blocked import BlockedInverseModel


def small_mdp(sparse=False):
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
    return MDP(transitions, rng.normal(size=(3, n)), 0.99)


class OptimizedTests(unittest.TestCase):
    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def test_every_update_survives_conversion_and_multiple_panel_flushes(self):
        rng = np.random.default_rng(20261011)
        for n, dense in ((1, False), (7, False), (31, False), (9, True)):
            with self.subTest(n=n, dense=dense):
                model = BlockedInverseModel(n, beta=0.7, dense=dense, block_size=32)
                expected = 0.7*np.eye(n)
                total = 2*n+70
                for step in range(total):
                    a, c = rng.normal(size=(2, n))
                    expected += np.outer(a, c)
                    model.update(a, c)
                    a[:] = 1e90
                    c[:] = -1e90
                    x = rng.normal(size=n)
                    original = x.copy()
                    np.testing.assert_allclose(model.apply(x), expected@x,
                                               rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(model.materialize(), expected,
                                               rtol=2e-12, atol=2e-12)
                    np.testing.assert_array_equal(x, original)
                    arrays = [model.h] if model.h is not None else [model.a, model.c]
                    if model.pending_count:
                        arrays += [model._pending_a[:, :model.pending_count],
                                   model._pending_c[:, :model.pending_count]]
                    self.assertTrue(all(a.flags.f_contiguous for a in arrays))
                pending_updates = total if dense else total-(n+1)//2
                self.assertEqual(model.updates, total)
                self.assertEqual(model.conversions, 0 if dense else 1)
                self.assertEqual(model.flushes, pending_updates//32)
                self.assertEqual(model.pending_count, pending_updates % 32)

    def test_default_is_explicit_paper_backend_and_invalid_name_fails(self):
        self.assertIn("reference", BACKENDS)
        mdp = small_mdp()
        default = solve(mdp)
        explicit = solve(mdp, backend="block32_cache")
        self.assertEqual(default["backend"], "block32_cache")
        for key in ("value", "policy", "history", "stats", "iterations"):
            self.assertEqual(default[key], explicit[key])
        with self.assertRaisesRegex(ValueError, "unknown LSQN backend"):
            solve(mdp, backend="missing-backend")

    def test_cache_refresh_and_certificates_on_dense_sparse_and_singular_h0(self):
        for sparse in (False, True):
            for initial_scale in (0.0, 1.0):
                with self.subTest(sparse=sparse, initial_scale=initial_scale):
                    mdp = small_mdp(sparse)
                    options = SolverOptions(initial_scale=initial_scale)
                    result = solve(mdp, options, mdp.common_lower_bound(),
                                   backend="block32_cache")
                    reference = solve(mdp, options, mdp.common_lower_bound(),
                                      backend="reference")
                    self.assertTrue(result["converged"], result["message"])
                    self.assertTrue(reference["converged"], reference["message"])
                    self.assertEqual(result["stats"]["cached_proposal_updates"],
                                     result["stats"]["enrichments"])
                    self.assertGreater(result["stats"]["cache_refreshes"], 0)
                    self.assertEqual(result["stats"]["model_applies"],
                                     result["iterations"]+result["stats"]["cache_refreshes"])
                    self.assertEqual(reference["stats"]["cached_proposal_updates"], 0)
                    value = np.asarray(result["value"])
                    tv, _ = mdp.bellman(value)
                    self.assertLessEqual(np.linalg.norm(value-tv, np.inf), options.tol)
                    self.assertLessEqual(np.linalg.norm(value-reference["value"], np.inf),
                                         2*options.tol/(1-mdp.gamma))
                    for h in result["history"]:
                        r2 = h["residual2_before"]**2
                        self.assertLessEqual(h["enrichments"], mdp.n_states)
                        self.assertLessEqual(h["forcing"], h["target"]*(1+1e-8))
                        self.assertFalse(h["capped"])
                        self.assertLessEqual(h["slope"], -0.5*r2*(1-1e-10))
                        self.assertLessEqual(0.5*h["residual2_after"]**2,
                                             0.5*r2+1e-4*h["alpha"]*h["slope"]
                                             +8*np.finfo(float).eps*r2)
                        self.assertAlmostEqual(np.log2(h["alpha"]),
                                               round(np.log2(h["alpha"])), places=11)

    def test_orthogonal_secants_remain_protected_in_pending_panels(self):
        rng = np.random.default_rng(2718)
        n = 16
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        jacobian = np.eye(n)+0.05*rng.normal(size=(n, n))
        images = np.linalg.solve(jacobian, q)
        model = BlockedInverseModel(n, beta=0.0, block_size=32)
        for j in range(n):
            model.update(images[:, j]-model.apply(q[:, j]), q[:, j])
            for i in range(j+1):
                np.testing.assert_allclose(model.apply(q[:, i]), images[:, i],
                                           atol=3e-13, rtol=3e-13)
        self.assertGreater(model.pending_count, 0)
        self.assertEqual(model.conversions, 1)



class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit compilation belongs to test preparation, never solve().
        from lsqn import _native
        _native.prepare()

    def setUp(self):
        self.thread_limit = threadpool_limits(limits=1)
        self.thread_limit.__enter__()

    def tearDown(self):
        self.thread_limit.__exit__(None, None, None)

    def test_runtime_results_certificates_and_per_solve_isolation(self):
        first = None
        for sparse, backend in [(False, "cache_reuse"), (True, "native_query"),
                                (False, "native"), (True, "fused_sparse")]:
            prepare_backend(backend)
            mdp = small_mdp(sparse)
            for scale in (0., 1.):
                options = SolverOptions(initial_scale=scale)
                events = []
                actual = solve(mdp, options, mdp.common_lower_bound(),
                               callback=events.append, backend=backend)
                reference = solve(mdp, options, mdp.common_lower_bound(), backend="reference")
                self.assertTrue(actual['converged'], actual['message'])
                self.assertLessEqual(np.max(np.abs(np.array(actual['value'])-reference['value'])),
                                     2*options.tol/(1-mdp.gamma))
                tv, _ = mdp.bellman(np.asarray(actual['value']))
                self.assertLessEqual(np.max(np.abs(np.array(actual['value'])-tv)), options.tol)
                self.assertEqual(events[0]['event'], 'initial')
                self.assertEqual(events[-1]['event'], 'finished')
                self.assertEqual(actual['stats']['enrichments'],
                                 sum(h['enrichments'] for h in actual['history']))
                self.assertEqual(actual['stats']['cached_proposal_updates'],
                                 actual['stats']['enrichments'])
                for h in actual['history']:
                    rr = h['residual2_before']**2
                    self.assertLessEqual(h['enrichments'], mdp.n_states)
                    self.assertLessEqual(h['forcing'], h['target']*(1+1e-8))
                    self.assertFalse(h['capped'])
                    self.assertLessEqual(h['slope'], -.5*rr*(1-1e-10))
                    self.assertLessEqual(.5*h['residual2_after']**2,
                        .5*rr+1e-4*h['alpha']*h['slope']+8*np.finfo(float).eps*rr)
                if backend == 'native_query':
                    self.assertGreater(actual['stats']['native_learning_queries'], 0)
                if first is None:
                    first = actual
        repeated = solve(small_mdp(), SolverOptions(initial_scale=0.),
                         small_mdp().common_lower_bound(), backend='cache_reuse')
        for key in ['value', 'policy', 'history', 'stats']:
            self.assertEqual(first[key], repeated[key])

    def test_native_policy_rows_and_support_certificate(self):
        from lsqn._runtime import Bindings, NativePolicyCache, configuration
        rng = np.random.default_rng(141)
        for sparse in (False, True):
            mdp = small_mdp(sparse)
            cache = NativePolicyCache(Bindings())
            for j in range(8):
                pi = rng.integers(0, mdp.n_actions, size=mdp.n_states)
                expected = mdp.policy_matrix(pi)
                actual = cache.get(mdp, pi)
                np.testing.assert_array_equal(actual.toarray() if sparse else actual,
                                              expected.toarray() if sparse else expected)
                self.assertIs(cache.get(mdp, pi), actual)
            cfg = configuration(mdp, 'fused_sparse')
            for extras in (0., .1, .7):
                pi = rng.integers(0, mdp.n_actions, size=mdp.n_states)
                mask = rng.random((mdp.n_actions, mdp.n_states)) < extras
                mask[pi, np.arange(mdp.n_states)] = True
                env = cfg['envelope_factory'](mdp, mask, pi,
                    dict(policy_products=0, support_row_products=0))
                f, x = rng.normal(size=(2, mdp.n_states))
                jx = env.j(x)
                np.testing.assert_allclose(jx, x-mdp.gamma*mdp.policy_matrix(pi).dot(x),
                                           rtol=1e-13, atol=1e-13)
                self.assertAlmostEqual(env.pairing_cached(f, x, jx), env.pairing(f, x), places=11)

    def test_native_query_panels_and_coordinate_completion(self):
        from lsqn._runtime import configuration, original_query
        rng = np.random.default_rng(144)
        for n in (1, 3, 16, 32):
            matrix = sp.eye(n, format='csr', dtype=float)
            mdp = MDP([matrix], np.zeros((1, n)), .99)
            for m in sorted(set([0, min(1, n-1), min(6, n-1)])):
                for updates in (0, n//2+9, n//2+31):
                    cfg = configuration(mdp, 'native_query')
                    env = cfg['envelope_factory'](mdp, np.ones((1, n), dtype=bool),
                        np.zeros(n, dtype=np.int64), dict(policy_products=0, support_row_products=0))
                    model = BlockedInverseModel(n, block_size=32)
                    for _ in range(updates):
                        model.update(rng.normal(size=n)/n, rng.normal(size=n))
                    U = np.asfortranarray(np.linalg.qr(rng.normal(size=(n, m)))[0])
                    Q = np.asfortranarray(np.linalg.qr(rng.normal(size=(n, m)))[0])
                    S = np.asfortranarray(rng.normal(size=(n, m)))
                    for p in (rng.normal(size=n), np.zeros(n)):
                        jp = env.j(p)
                        expected = original_query(model, env, p, jp, U, Q, S, True)
                        actual = cfg['learning_query'](model, env, p, jp, U, Q, S, True)
                        for a, b in zip(actual, expected):
                            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
                        if not np.any(p):
                            self.assertGreater(env.stats['coordinate_completion_queries'], 0)

    def test_unsupported_native_layout_and_callback_timeout_are_explicit(self):
        from lsqn._native import NativeBackendError
        with self.assertRaisesRegex(NativeBackendError, 'requires CSR'):
            solve(small_mdp(), backend='native_query')
        mdp = small_mdp()
        mdp.transitions[0] = np.asfortranarray(mdp.transitions[0])
        with self.assertRaisesRegex(NativeBackendError, 'C-contiguous'):
            solve(mdp, backend='native')
        for field in ('data', 'indices', 'indptr'):
            strided_mdp = small_mdp(True)
            p = strided_mdp.transitions[0]
            values = getattr(p, field)
            backing = np.empty(2*values.size, dtype=values.dtype)
            backing[::2] = values
            setattr(p, field, backing[::2])
            with self.assertRaisesRegex(NativeBackendError, 'C-contiguous'):
                solve(strided_mdp, backend='native_query')
        records = []
        def callback(record):
            records.append(record)
            if record['event'] == 'accepted':
                raise TimeoutError('runtime test deadline')
        with self.assertRaisesRegex(TimeoutError, 'runtime test deadline'):
            solve(small_mdp(True), callback=callback, backend='native_query')
        self.assertEqual([r['event'] for r in records], ['initial', 'accepted'])
        self.assertIsNotNone(records[-1]['accepted_step'])


if __name__ == "__main__":
    unittest.main()
