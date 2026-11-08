"""Deferred, nontruncating storage of the SCB inverse rank-one updates.

This module changes only how an inverse model is represented and evaluated.
It never changes the vectors supplied to ``update(a, c)`` and represents every
update in H <- H + a c.T.  Pending terms are merged into a dense base by BLAS-3
when a panel fills.  The associativity change can change floating-point rounding
and therefore is not a promise of bit-for-bit identical optimization iterates.

The public attributes used by the frozen solver's counters are retained.  Peak
storage counts include the pending panels but exclude temporary BLAS workspace.
"""

import numpy as np
from scipy.linalg.blas import dgemm


class _ColumnBuffer:
    """Growing column storage, with no compression or discarded columns."""

    def __init__(self, n):
        self.n = n
        self.used = 0
        self.capacity = 0
        self.data = np.empty((n, 0), order="F")

    @property
    def view(self):
        return self.data[:, :self.used]

    def append(self, vector):
        if self.used == self.capacity:
            self.capacity = min(self.n, max(4, 2 * self.capacity))
            replacement = np.empty((self.n, self.capacity), order="F")
            replacement[:, :self.used] = self.view
            self.data = replacement
        self.data[:, self.used] = vector
        self.used += 1
        return self.view


class BlockedInverseModel:
    """Identity-plus-low-rank initially, then dense base plus pending updates.

    Before dense conversion, H = beta I + a c.T, just as in the frozen model.
    Afterwards, H = h + pending_a pending_c.T.  ``h`` alone is the flushed base,
    not the complete mathematical model while there are pending terms; callers
    must use ``apply`` (or ``materialize`` for diagnostics), not access ``h``.

    ``block_size`` is an implementation/storage parameter, not a learning limit.
    An arbitrary number of updates is retained through successive exact-algebra
    flushes.  No rank truncation, damping, or approximation is performed.
    """

    def __init__(self, n, beta=1.0, dense=False, fast=False, block_size=32):
        if int(n) != n or n <= 0:
            raise ValueError("n must be a positive integer")
        if int(block_size) != block_size or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.n = int(n)
        self.beta = float(beta)
        self.fast = fast
        self.block_size = int(block_size)
        self.h = self.beta * np.eye(self.n, order="F") if dense else None
        self.ab, self.cb = _ColumnBuffer(self.n), _ColumnBuffer(self.n)
        self.a = np.empty((self.n, 0))
        self.c = np.empty((self.n, 0))
        self._pending_a = None
        self._pending_c = None
        self._pending_count = 0
        self.peak = self.n * self.n if dense else 0
        self.allocated_peak = self.peak
        self.updates = 0
        self.conversions = 0
        self.flushes = 0

    @property
    def pending_count(self):
        return self._pending_count

    def _record_storage(self):
        if self.h is None:
            live = self.a.size + self.c.size
            allocated = self.ab.data.size + self.cb.data.size
        else:
            live = self.h.size + 2 * self.n * self._pending_count
            allocated = self.h.size
            if self._pending_a is not None:
                allocated += self._pending_a.size + self._pending_c.size
        self.peak = max(self.peak, live)
        self.allocated_peak = max(self.allocated_peak, allocated)

    def apply(self, vector):
        if self.h is None:
            return self.beta * vector + self.a @ (self.c.T @ vector)
        result = self.h @ vector
        if self._pending_count:
            a = self._pending_a[:, :self._pending_count]
            c = self._pending_c[:, :self._pending_count]
            result += a @ (c.T @ vector)
        return result

    def _convert_to_dense(self):
        base = self.beta * np.eye(self.n, order="F")
        self.h = dgemm(
            1.0, self.a, self.c, beta=1.0, c=base,
            trans_b=True, overwrite_c=True,
        )
        self.a = np.empty((self.n, 0))
        self.c = np.empty((self.n, 0))
        self.ab, self.cb = _ColumnBuffer(self.n), _ColumnBuffer(self.n)
        self.conversions += 1
        self._record_storage()

    def _flush_pending(self):
        if not self._pending_count:
            return
        a = self._pending_a[:, :self._pending_count]
        c = self._pending_c[:, :self._pending_count]
        self.h = dgemm(
            1.0, a, c, beta=1.0, c=self.h,
            trans_b=True, overwrite_c=True,
        )
        self._pending_count = 0
        self.flushes += 1

    def update(self, a, c):
        self.updates += 1
        if self.h is None:
            self.a = self.ab.append(a)
            self.c = self.cb.append(c)
            self._record_storage()
            # Match the frozen model's mathematical representation switch.
            if 2 * self.a.shape[1] >= self.n:
                self._convert_to_dense()
            return

        if self._pending_a is None:
            self._pending_a = np.empty((self.n, self.block_size), order="F")
            self._pending_c = np.empty((self.n, self.block_size), order="F")
        # Copy the update vectors: caller-side mutation must not change H.
        self._pending_a[:, self._pending_count] = a
        self._pending_c[:, self._pending_count] = c
        self._pending_count += 1
        self._record_storage()
        if self._pending_count == self.block_size:
            self._flush_pending()

    def materialize(self):
        """Return a full diagnostic copy without altering the representation."""
        if self.h is None:
            return self.beta * np.eye(self.n) + self.a @ self.c.T
        result = self.h.copy()
        if self._pending_count:
            a = self._pending_a[:, :self._pending_count]
            c = self._pending_c[:, :self._pending_count]
            result += a @ c.T
        return result


# Drop-in class name for the experimental runner's isolated module namespace.
InverseModel = BlockedInverseModel
