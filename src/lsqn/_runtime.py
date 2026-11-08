"""Static port of the final October 14 runtime kernels.

Sources and hashes are recorded in optimized-port-notes.txt. All policy, row,
query and inverse-model buffers are scoped to a single solve. Native library
loading is explicit; these helpers never compile or substitute another backend.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import _sparsetools

from . import _native
from ._core import GroupedEnvelope
from ._policy_cache import PolicyMatrixCache, envelope_class

NATIVE_BACKENDS = ("native", "fused_sparse", "native_query")


class Bindings:
    def __init__(self):
        self.lib = _native.get_library()
        self.dot, self.gemv = _native.blas_pointers()


def validate_model(mdp, backend):
    _native.validate_dimensions(mdp.n_states, mdp.n_actions, 32)
    if any(p.shape != (mdp.n_states, mdp.n_states) for p in mdp.transitions):
        raise _native.NativeBackendError("Native transition shapes must equal (n_states, n_states).")
    if backend == "native_query" and not mdp.sparse:
        raise _native.NativeBackendError("native_query requires CSR transitions.")
    if mdp.sparse:
        supported = all(sp.isspmatrix_csr(p) and p.dtype == np.float64
                        and p.indices.dtype == np.int32 and p.indptr.dtype == np.int32
                        and all(a.flags.c_contiguous for a in (p.data, p.indices, p.indptr))
                        for p in mdp.transitions)
    else:
        supported = all(p.dtype == np.float64 and p.flags.c_contiguous
                        for p in mdp.transitions)
    if not supported:
        raise _native.NativeBackendError(
            "Native runtime backends require float64 C-contiguous dense matrices "
            "or float64 CSR matrices with int32 indices and row pointers; "
            "all CSR data/index/pointer arrays must be C-contiguous.")


class NativePolicyCache(PolicyMatrixCache):
    def __init__(self, bindings):
        super().__init__()
        self.bindings = bindings

    def get(self, mdp, policy):
        if mdp.sparse:
            return super().get(mdp, policy)
        if self.mdp is not None and self.mdp is not mdp:
            raise ValueError("Different MDP")
        if self.mdp is None:
            self.action_ptrs = np.asarray([p.ctypes.data for p in mdp.transitions], dtype=np.uintp)
        self.mdp = mdp
        policy = np.ascontiguousarray(policy, dtype=np.int64)
        changed = np.ones(mdp.n_states, dtype=bool) if self.policy is None else policy != self.policy
        if not np.any(changed):
            self.hits += 1
            return self.matrix
        self.builds += 1
        self.refreshed_rows += int(np.count_nonzero(changed))
        if self.matrix is None:
            self.matrix = np.empty((mdp.n_states, mdp.n_states), dtype=np.float64)
        self.bindings.lib.scb_copy_policy_rows(
            self.matrix.ctypes.data, self.action_ptrs.ctypes.data,
            policy.ctypes.data, changed.ctypes.data, mdp.n_states)
        self.policy = policy.copy()
        return self.matrix


def row_envelope(original, cache):
    cached = envelope_class(original, cache)

    class RowEnvelope(cached):
        def __init__(self, mdp, mask, policy, stats):
            super().__init__(mdp, mask, policy, stats)
            self.policy = policy

        def pairing_cached(self, f, x, jx):
            n = self.mdp.n_states
            if self.active_count == n:
                return float(f @ jx)
            if self.mdp.sparse or self.active_count-n > n//2:
                return self.pairing(f, x)
            values = f*jx
            for action, p in enumerate(self.mdp.transitions):
                idx = np.flatnonzero(self.mask[action] & (self.policy != action))
                if idx.size:
                    extra = f[idx]*(x[idx]-self.mdp.gamma*p[idx, :].dot(x))
                    values[idx] = np.maximum(values[idx], extra)
                    self.stats['support_row_products'] += int(idx.size)
            self.stats['partial_certificates'] = self.stats.get('partial_certificates', 0)+1
            return float(values.sum())
    return RowEnvelope


def matvec(p, x):
    if not (sp.isspmatrix_csr(p) and p.dtype == np.float64
            and x.dtype == np.float64 and x.ndim == 1):
        return p.dot(x)
    result = np.zeros(p.shape[0], dtype=np.float64)
    _sparsetools.csr_matvec(p.shape[0], p.shape[1], p.indptr, p.indices, p.data, x, result)
    return result


def kernel_envelope(cache):
    base = row_envelope(GroupedEnvelope, cache)

    class KernelEnvelope(base):
        def j(self, x):
            self.stats['policy_products'] += 1
            return x-self.mdp.gamma*matvec(self.p, x)

        def pairing(self, f, x, upper=True):
            if not self.mdp.sparse:
                return super().pairing(f, x, upper=upper)
            n = self.mdp.n_states
            if self.active_count == n:
                self.stats['support_row_products'] += n
                return float(f @ (x-self.mdp.gamma*matvec(self.p, x)))
            self.stats['support_row_products'] += n*self.mdp.n_actions
            vals = f[None, :]*(x[None, :]-self.mdp.gamma*np.vstack(
                [matvec(p, x) for p in self.mdp.transitions]))
            if upper:
                return float(np.max(np.where(self.mask, vals, -np.inf), axis=0).sum())
            return float(np.min(np.where(self.mask, vals, np.inf), axis=0).sum())
    return KernelEnvelope


class ActionRows:
    def __init__(self, mdp, bindings):
        self.mdp, self.bindings = mdp, bindings
        self.n, self.sparse, self.supported = mdp.n_states, mdp.sparse, True
        if self.sparse:
            self.data = np.asarray([p.data.ctypes.data for p in mdp.transitions], dtype=np.uintp)
            self.indices = np.asarray([p.indices.ctypes.data for p in mdp.transitions], dtype=np.uintp)
            self.indptr = np.asarray([p.indptr.ctypes.data for p in mdp.transitions], dtype=np.uintp)
        else:
            self.data = np.asarray([p.ctypes.data for p in mdp.transitions], dtype=np.uintp)

    def policy(self, pi):
        return PolicyProduct(self, np.ascontiguousarray(pi, dtype=np.int64))

    def selected(self, action, rows, x):
        p = self.mdp.transitions[action]
        if self.sparse:
            return p[rows, :].dot(x)
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        out = np.empty(len(rows), dtype=np.float64)
        self.bindings.lib.scb_dense_rows(p.ctypes.data, rows.ctypes.data, x.ctypes.data,
                                        out.ctypes.data, self.n, len(rows), self.bindings.dot)
        return out


class PolicyProduct:
    def __init__(self, action_rows, policy):
        self.rows, self.policy = action_rows, policy

    def dot(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        r = self.rows
        out = np.empty(r.n, dtype=np.float64)
        if r.sparse:
            r.bindings.lib.scb_sparse_policy(
                r.data.ctypes.data, r.indices.ctypes.data, r.indptr.ctypes.data,
                self.policy.ctypes.data, x.ctypes.data, out.ctypes.data, r.n)
        else:
            r.bindings.lib.scb_dense_policy(r.data.ctypes.data, self.policy.ctypes.data,
                                          x.ctypes.data, out.ctypes.data, r.n, r.bindings.dot)
        return out


def direct_envelope(cache, bindings):
    base = kernel_envelope(cache)
    storage = {}

    class DirectEnvelope(base):
        def __init__(self, mdp, mask, policy, stats):
            if 'rows' not in storage:
                storage['rows'] = ActionRows(mdp, bindings)
            self.rows = storage['rows']
            if not mdp.sparse:
                super().__init__(mdp, mask, policy, stats)
            else:
                self.mdp, self.mask, self.stats, self.policy = mdp, mask, stats, policy
                self.active_count = int(mask.sum())
                self.p = self.rows.policy(policy)

        def j(self, x):
            self.stats['policy_products'] += 1
            if self.mdp.sparse:
                r = self.rows
                out = np.empty_like(x)
                bindings.lib.scb_sparse_j(r.data.ctypes.data, r.indices.ctypes.data,
                    r.indptr.ctypes.data, self.policy.ctypes.data, x.ctypes.data,
                    out.ctypes.data, r.n, self.mdp.gamma)
                return out
            return x-self.mdp.gamma*self.p.dot(x)

        def pairing_cached(self, f, x, jx):
            n = self.mdp.n_states
            if self.active_count == n:
                return float(f @ jx)
            if self.mdp.sparse:
                r, out = self.rows, np.empty_like(x)
                mask = np.ascontiguousarray(self.mask)
                bindings.lib.scb_sparse_certificate(
                    r.data.ctypes.data, r.indices.ctypes.data, r.indptr.ctypes.data,
                    self.policy.ctypes.data, mask.ctypes.data, f.ctypes.data,
                    x.ctypes.data, jx.ctypes.data, out.ctypes.data, n,
                    self.mdp.n_actions, self.mdp.gamma)
                self.stats['support_row_products'] += self.active_count-n
                return float(out.sum())
            if self.active_count-n > n//2:
                return self.pairing(f, x)
            values = f*jx
            for action in range(self.mdp.n_actions):
                idx = np.flatnonzero(self.mask[action] & (self.policy != action))
                if idx.size:
                    extra = f[idx]*(x[idx]-self.mdp.gamma*self.rows.selected(action, idx, x))
                    values[idx] = np.maximum(values[idx], extra)
                    self.stats['support_row_products'] += int(idx.size)
            return float(values.sum())
    return DirectEnvelope


def original_query(H, env, p, jp, U, Q, S, protect):
    """Original coordinate-completion branch, also used by the fused source."""
    n = len(p)
    u = p.copy()
    reuse = U.shape[1] == 0
    for _ in range(2):
        u -= U @ (U.T @ u)
    if np.linalg.norm(u) <= np.finfo(float).eps*max(1., np.linalg.norm(p)):
        reuse = False
        idx = int(np.argmax(1-np.sum(U*U, axis=1)))
        u = np.eye(1, n, idx).ravel()-U @ U[idx, :]
        for _ in range(2):
            u -= U @ (U.T @ u)
    un = np.linalg.norm(u)
    if un <= np.finfo(float).tiny:
        raise ArithmeticError('primal basis breakdown')
    u /= un
    y = jp/un if reuse else env.j(u)
    z, w = y.copy(), u.copy()
    if protect:
        for _ in range(2):
            coeff = Q.T @ z
            z -= Q @ coeff
            w -= S @ coeff
    zn = np.linalg.norm(z)
    if zn <= np.finfo(float).tiny:
        raise ArithmeticError('image basis breakdown')
    z /= zn
    w /= zn
    return u, z, w, w-H.apply(z)


def native_query(bindings):
    def query(H, env, p, jp, U, Q, S, protect):
        if not (protect and env.mdp.sparse and all(x.flags.f_contiguous for x in (U, Q, S))
                and (H.h is None or H.h.flags.f_contiguous)):
            raise _native.NativeBackendError("native_query received an unsupported internal layout.")
        if not hasattr(env, 'query_workspace'):
            env.query_workspace = np.empty(4*len(p)+max(len(p), H.block_size))
        arr = [p, jp, U, Q, S, H.h, H.a, H.c, H._pending_a, H._pending_c,
               env.rows.data, env.rows.indices, env.rows.indptr, env.policy]
        ptr = np.asarray([0 if x is None else x.ctypes.data for x in arr], dtype=np.uintp)
        code = bindings.lib.scb_query(
            ptr.ctypes.data, len(p), U.shape[1], H.a.shape[1], H.pending_count,
            H.beta, env.mdp.gamma, env.query_workspace.ctypes.data, bindings.gemv, bindings.dot)
        # This is the source algorithm's coordinate-completion guard, not a
        # substitute implementation for unavailable native code.
        if code == 1:
            env.stats['coordinate_completion_queries'] = env.stats.get('coordinate_completion_queries', 0)+1
            return original_query(H, env, p, jp, U, Q, S, protect)
        if code:
            raise ArithmeticError('native query basis breakdown')
        if U.shape[1]:
            env.stats['policy_products'] += 1
        else:
            env.stats['reused_learning_products'] = env.stats.get('reused_learning_products', 0)+1
        env.stats['native_learning_queries'] = env.stats.get('native_learning_queries', 0)+1
        return tuple(env.query_workspace[i*len(p):(i+1)*len(p)] for i in range(4))
    return query


def configuration(mdp, backend):
    """Create fresh, empty per-solve cache state; all model data setup is timed."""
    if backend == 'cache_reuse':
        cache = PolicyMatrixCache()
        return dict(envelope_factory=envelope_class(GroupedEnvelope, cache),
                    certificate_mode='singleton')
    bindings = Bindings()
    validate_model(mdp, backend)
    cache = NativePolicyCache(bindings)
    result = dict(certificate_mode='cached')
    if backend == 'native':
        result.update(envelope_factory=kernel_envelope(cache), reuse_learning_image=True)
        return result
    result['envelope_factory'] = direct_envelope(cache, bindings)
    if backend == 'native_query':
        result['learning_query'] = native_query(bindings)
        return result
    if backend != 'fused_sparse':
        raise ValueError('unknown runtime backend: '+str(backend))

    def project(U, x, work):
        if U.shape[1]:
            bindings.lib.scb_project(U.ctypes.data, x.ctypes.data, work.ctypes.data,
                                     *U.shape, bindings.gemv)

    def protect(Q, S, z, w, work):
        if Q.shape[1]:
            bindings.lib.scb_protect(Q.ctypes.data, S.ctypes.data, z.ctypes.data,
                w.ctypes.data, work.ctypes.data, *Q.shape, bindings.gemv)

    result.update(reuse_learning_image=True, basis_projector=project, basis_protector=protect)
    return result
