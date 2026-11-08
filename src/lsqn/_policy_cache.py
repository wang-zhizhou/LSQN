"""Exact policy-matrix assembly/reuse, scoped to one solve of a fixed MDP.

No transition approximation, low-rank truncation, or policy modification.
The cached matrix may be updated in place only after its previous envelope
has finished being used. This is the lifecycle of the frozen SCB outer loop.
"""
import numpy as np
import scipy.sparse as sp


class PolicyMatrixCache:
    def __init__(self):
        self.mdp = None
        self.policy = None
        self.matrix = None
        self.stacked = None
        self.builds = 0
        self.hits = 0
        self.refreshed_rows = 0

    def get(self, mdp, policy):
        if self.mdp is not None and self.mdp is not mdp:
            raise ValueError('Policy cache must not be shared across MDPs')
        self.mdp = mdp
        policy = np.asarray(policy, dtype=np.int64)
        changed = (np.ones(mdp.n_states, dtype=bool) if self.policy is None
                   else policy != self.policy)
        if not np.any(changed):
            self.hits += 1
            return self.matrix
        self.builds += 1
        self.refreshed_rows += int(np.count_nonzero(changed))
        if mdp.sparse:
            if self.stacked is None:
                self.stacked = sp.vstack(mdp.transitions, format='csr')
            selected = policy * mdp.n_states + np.arange(mdp.n_states)
            self.matrix = self.stacked[selected, :]
        else:
            if self.matrix is None:
                self.matrix = np.empty((mdp.n_states, mdp.n_states))
            for action in range(mdp.n_actions):
                idx = np.flatnonzero(changed & (policy == action))
                if idx.size:
                    self.matrix[idx] = mdp.transitions[action][idx]
        self.policy = policy.copy()
        return self.matrix


def envelope_class(original, cache):
    class CachedEnvelope(original):
        def __init__(self, mdp, mask, policy, stats):
            self.mdp, self.mask, self.stats = mdp, mask, stats
            self.p = cache.get(mdp, policy)
            self.active_count = int(mask.sum())
    return CachedEnvelope
