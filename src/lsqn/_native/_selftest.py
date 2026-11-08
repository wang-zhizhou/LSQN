"""Numerical identities checked before native benchmark execution."""
import ctypes
import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits


def run():
    from . import get_library, blas_pointers, status, validate_dimensions

    lib = get_library()
    dot, gemv = blas_pointers()
    rng = np.random.default_rng(17391)
    checks = []
    n, m = 17, 4
    validate_dimensions(n, m)

    def close(name, actual, expected, tolerance=2e-12):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
        checks.append(name)

    with threadpool_limits(limits=1):
        actions = [np.ascontiguousarray(rng.random((n, n))) for _ in range(3)]
        policy = np.ascontiguousarray(rng.integers(0, len(actions), size=n), dtype=np.int64)
        pointers = np.asarray([a.ctypes.data for a in actions], dtype=np.uintp)
        changed = np.ones(n, dtype=np.uint8)
        matrix = np.zeros((n, n))
        lib.scb_copy_policy_rows(matrix.ctypes.data, pointers.ctypes.data,
                                 policy.ctypes.data, changed.ctypes.data, n)
        expected_matrix = np.vstack([actions[int(policy[i])][i] for i in range(n)])
        np.testing.assert_array_equal(matrix, expected_matrix)
        checks.append("exact_policy_row_copy")
        policy[::3] = (policy[::3] + 1) % len(actions)
        changed[:] = 0
        changed[::3] = 1
        lib.scb_copy_policy_rows(matrix.ctypes.data, pointers.ctypes.data,
                                 policy.ctypes.data, changed.ctypes.data, n)
        expected_matrix = np.vstack([actions[int(policy[i])][i] for i in range(n)])
        np.testing.assert_array_equal(matrix, expected_matrix)
        checks.append("exact_changed_row_copy")
        x = rng.normal(size=n)
        output = np.empty(n)
        lib.scb_dense_policy(pointers.ctypes.data, policy.ctypes.data, x.ctypes.data,
                              output.ctypes.data, n, dot)
        close("dense_policy_blas_dot", output, expected_matrix @ x)
        rows = np.array([0, 3, 9, 16], dtype=np.int64)
        selected = np.empty(len(rows))
        lib.scb_dense_rows(actions[0].ctypes.data, rows.ctypes.data, x.ctypes.data,
                            selected.ctypes.data, n, len(rows), dot)
        close("selected_dense_rows", selected, actions[0][rows] @ x)

        u = np.asfortranarray(np.linalg.qr(rng.normal(size=(n, m)))[0])
        s = np.asfortranarray(rng.normal(size=(n, m)))
        work = np.empty(n)
        actual = x.copy()
        expected = x.copy()
        for _ in range(2):
            expected -= u @ (u.T @ expected)
        lib.scb_project(u.ctypes.data, actual.ctypes.data, work.ctypes.data, n, m, gemv)
        close("two_projection_passes", actual, expected)
        z, w = rng.normal(size=n), rng.normal(size=n)
        ez, ew = z.copy(), w.copy()
        for _ in range(2):
            coefficients = u.T @ ez
            ez -= u @ coefficients
            ew -= s @ coefficients
        lib.scb_protect(u.ctypes.data, s.ctypes.data, z.ctypes.data, w.ctypes.data,
                         work.ctypes.data, n, m, gemv)
        close("two_protection_passes_z", z, ez)
        close("two_protection_passes_w", w, ew)
        a = np.asfortranarray(rng.normal(size=(n, 3)) / n)
        c = np.asfortranarray(rng.normal(size=(n, 3)))
        lib.scb_lowrank(a.ctypes.data, c.ctypes.data, x.ctypes.data, output.ctypes.data,
                         work.ctypes.data, n, 3, .8, gemv)
        close("complete_lowrank_apply", output, .8*x + a @ (c.T @ x))

        sparse = []
        for dense in actions:
            dense = dense * (dense > .65)
            dense[np.arange(n), np.arange(n)] += 1
            dense /= dense.sum(axis=1, keepdims=True)
            sparse.append(sp.csr_matrix(dense))
        assert all(p.indices.dtype == np.int32 and p.indptr.dtype == np.int32 for p in sparse)
        data = np.asarray([p.data.ctypes.data for p in sparse], dtype=np.uintp)
        indices = np.asarray([p.indices.ctypes.data for p in sparse], dtype=np.uintp)
        indptr = np.asarray([p.indptr.ctypes.data for p in sparse], dtype=np.uintp)
        selected_matrix = sp.vstack([sparse[int(policy[i])][i] for i in range(n)], format="csr")
        lib.scb_sparse_policy(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                               policy.ctypes.data, x.ctypes.data, output.ctypes.data, n)
        close("sparse_policy_rows", output, selected_matrix @ x)
        gamma = .99
        lib.scb_sparse_j(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                          policy.ctypes.data, x.ctypes.data, output.ctypes.data, n, gamma)
        jx = x - gamma * (selected_matrix @ x)
        close("sparse_jacobian_rows", output, jx)
        f = rng.normal(size=n)
        mask = np.ones((len(sparse), n), dtype=np.uint8)
        lib.scb_sparse_certificate(data.ctypes.data, indices.ctypes.data, indptr.ctypes.data,
                                    policy.ctypes.data, mask.ctypes.data, f.ctypes.data,
                                    x.ctypes.data, jx.ctypes.data, output.ctypes.data,
                                    n, len(sparse), gamma)
        expected = np.max(np.vstack([f*(x - gamma*(p @ x)) for p in sparse]), axis=0)
        close("complete_sparse_certificate", output, expected)

        for width in (0, m):
            q = np.asfortranarray(u[:, :width])
            image_s = np.asfortranarray(s[:, :width])
            for dense_state in (False, True):
                h = np.asfortranarray(.8*np.eye(n) + a @ c.T) if dense_state else None
                pa = np.asfortranarray(rng.normal(size=(n, 2)) / n) if dense_state else None
                pc = np.asfortranarray(rng.normal(size=(n, 2))) if dense_state else None
                rank, pending = (0, 2) if dense_state else (3, 0)
                jp = x - gamma*(selected_matrix @ x)
                arrays = [x, jp, q, q, image_s, h, a, c, pa, pc,
                          data, indices, indptr, policy]
                table = np.asarray([0 if item is None else item.ctypes.data for item in arrays], dtype=np.uintp)
                workspace = np.empty(5*n)
                code = lib.scb_query(table.ctypes.data, n, width, rank, pending, .8, gamma,
                                      workspace.ctypes.data, gemv, dot)
                if code:
                    raise AssertionError("Native query returned error code " + str(code))
                eu = x.copy()
                for _ in range(2):
                    eu -= q @ (q.T @ eu)
                norm = np.linalg.norm(eu)
                eu /= norm
                ez = jp / norm if width == 0 else eu - gamma*(selected_matrix @ eu)
                ew = eu.copy()
                for _ in range(2):
                    coefficients = q.T @ ez
                    ez -= q @ coefficients
                    ew -= image_s @ coefficients
                norm = np.linalg.norm(ez)
                ez /= norm
                ew /= norm
                applied = h @ ez + pa @ (pc.T @ ez) if dense_state else .8*ez + a @ (c.T @ ez)
                close("protected_query_m%d_dense%d" % (width, dense_state),
                      workspace[:4*n], np.concatenate([eu, ez, ew, ew-applied]), 2e-11)
    return dict(passed=True, checks=checks, checks_passed=len(checks),
                blas_integer_bits=status()["identity"]["blas_integer_bits"])
