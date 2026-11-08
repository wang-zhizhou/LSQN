"""Per-call observations; there is no shared mutable tracing state."""

import time

import numpy as np


def noop(*args):
    pass


class Recorder:
    def __init__(self, callback):
        self.start = time.perf_counter()
        self.callback = callback
        self.trace = []
        self.value = None
        self.residual = None
        self.k = 0

    def observe(self, value, residual, k, accepted_step=None):
        self.value = np.asarray(value).copy()
        self.residual = np.asarray(residual).copy()
        self.k = int(k)
        row = dict(k=self.k, elapsed_s=time.perf_counter()-self.start,
                   residual2=float(np.linalg.norm(residual)),
                   residual_inf=float(np.linalg.norm(residual, np.inf)))
        duplicate = bool(self.trace and self.trace[-1]["k"] == self.k)
        if duplicate:
            self.trace[-1] = row
        else:
            self.trace.append(row)
        if self.callback is not None and not duplicate:
            self.callback(dict(
                row, event="initial" if k == 0 else "accepted",
                value=self.value.copy(), queries=0,
                accepted_step=dict(accepted_step) if accepted_step is not None else None,
            ))

    def finish(self, mdp, result):
        value = np.asarray(result["value"])
        if self.value is None or not np.array_equal(value, self.value):
            # The archived VI/QPI outer-budget exit can follow its last observed
            # value. Keep its recurrence/iteration count but report the returned
            # value's actual residual. Normal converged runs reuse the observation.
            tv, _ = mdp.bellman(value)
            residual = value-tv
        else:
            residual = self.residual
        k = int(result["iterations"])
        row = dict(k=k, elapsed_s=time.perf_counter()-self.start,
                   residual2=float(np.linalg.norm(residual)),
                   residual_inf=float(np.linalg.norm(residual, np.inf)))
        if self.trace and self.trace[-1]["k"] == k:
            self.trace[-1] = row
        else:
            self.trace.append(row)
        if self.callback is not None:
            self.callback(dict(
                row, event="finished", value=value.copy(), queries=0,
                accepted_step=None, converged=bool(result["converged"]),
                message=result["message"],
            ))
        result.update(residual=row["residual_inf"], trace=self.trace,
                      runtime_s=time.perf_counter()-self.start)
        return result
