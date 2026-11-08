"""Public interface to the paper's portable block32 and reference implementations."""

from functools import partial
import numpy as np

from ._core import solve_core
from ._blocked import BlockedInverseModel
from .options import SolverOptions

BACKENDS = ("block32_cache", "reference", "cache_reuse", "native", "fused_sparse", "native_query")


def prepare_backend(backend="block32_cache"):
    """Load a backend before timing; never compile or build model-dependent data.

    Native backends require a prior explicit ``lsqn._native.prepare()`` or the
    supplied preparation script. A missing or incompatible build raises.
    """
    if backend not in BACKENDS:
        raise ValueError("unknown LSQN backend: " + str(backend))
    if backend not in ("block32_cache", "reference"):
        from . import _runtime
        if backend in _runtime.NATIVE_BACKENDS:
            _runtime.Bindings()



def solve(mdp, options=None, v0=None, callback=None, *, backend="block32_cache"):
    """Solve a finite discounted MDP using full protected inner learning.

    ``mdp`` may be :class:`lsqn.MDP` or an object implementing its interface.
    ``options.initial_scale`` sets H0 to a scalar multiple of the identity;
    an arbitrary initial inverse matrix is not accepted by this interface.
    If ``v0`` is omitted, the original solver's zero initialization is used.
    The paper's main/public runners explicitly pass ``mdp.common_lower_bound()``.

    The return value is a dictionary containing ``converged``, ``message``,
    ``value``, ``policy``, ``residual``, ``iterations``, ``runtime_s``, ``stats``,
    ``history`` and ``trace``. ``history`` has one entry per accepted step,
    including ``enrichments`` (the number of inner-learning queries), the
    attained forcing ratio/target, line-search step, and descent certificate.
    ``trace`` records the initial and accepted outer values' residual norms.

    An optional ``callback(record)`` is called initially, after each accepted
    outer step, and at normal termination. Records contain ``event``
    (``initial``, ``accepted``, or ``finished``), ``k``, ``value`` (a copy),
    ``residual2``, ``residual_inf``, ``elapsed_s``, and cumulative ``queries``.
    An accepted event also contains ``accepted_step``. Finished events add
    ``converged`` and ``message``. The terminal event may repeat the last k.
    Callback work is included in runtime. A callback-raised ``TimeoutError``
    propagates to the caller, which can retain the last emitted value.

    ``backend='block32_cache'`` (default) is the implementation used for all
    four archived residual figures: nontruncating panels of 32 inverse updates,
    BLAS-3 merging, incremental proposal products, and the original full-model
    refresh before forcing acceptance. It requires NumPy/SciPy, with no C
    helper build. ``backend='reference'`` preserves the original uncached
    implementation. Unknown or unavailable backends fail instead of falling back.
    Final runtime backends are ``cache_reuse`` (Cluster), ``native`` (Latent),
    ``fused_sparse`` (Garnet), and ``native_query`` (Queue). The last three
    require explicit native preparation. Native dense inputs must be float64
    C-contiguous; sparse inputs must be float64 CSR with int32 indices.
    ``native_query`` requires sparse inputs. These implementations preserve
    the original guards, thresholds and stopping criteria. Their floating-point
    summation order can differ, so equal mathematical operations do not imply
    identical trajectories across machines or backends.
    """
    options = SolverOptions() if options is None else options
    if backend not in BACKENDS:
        raise ValueError("unknown LSQN backend: " + str(backend))
    if v0 is not None:
        v0 = np.asarray(v0, dtype=float)
        if v0.shape != (mdp.n_states,) or not np.all(np.isfinite(v0)):
            raise ValueError("v0 must be a finite vector with one entry per state")
    if not np.isfinite(options.initial_scale):
        raise ValueError("initial_scale must be finite")
    settings = {}
    if backend not in ("block32_cache", "reference"):
        from ._runtime import configuration
        settings = configuration(mdp, backend)
    result = solve_core(
        mdp, opts=options, v0=v0,
        width_fraction=options.width_fraction,
        callback=callback,
        inverse_model_factory=(partial(BlockedInverseModel, block_size=32)
                               if backend != "reference" else None),
        cache_proposal=backend != "reference",
        **settings,
    )
    result["backend"] = backend
    return result
