"""Optional official-model regeneration, using Gymnasium exactly 1.1.1.

Gymnasium is not required to run the included frozen models.
Extracted from original run_supplement.public_model; only imports/version check changed.
Source SHA256: 4dba0f50642355ec7fb2095de020a80ce76c7c1ccc82dadc729b5e6d424f452c
"""
import numpy as np
import scipy.sparse as sp
from lsqn import MDP as FiniteMDP

def public_model(name,gamma):
    import gymnasium as gym
    if gym.__version__ != "1.1.1":
        raise RuntimeError("Regeneration requires Gymnasium exactly 1.1.1")
    opts={'map_name':name.split('-')[-1],'is_slippery':True} if name.startswith('FrozenLake') else {}
    env=gym.make('FrozenLake-v1' if opts else name,**opts).unwrapped
    n,a=int(env.observation_space.n),int(env.action_space.n)
    # Redirect every terminated transition to an added zero-reward absorbing state.
    # No sampling, rollout horizon, reset distribution, or TimeLimit wrapper is used.
    rewards=np.zeros((a,n+1)); matrices=[]
    for action in range(a):
        rows=[];cols=[];weights=[]
        for s in range(n):
            for prob,t,reward,done in env.P[s][action]:
                rows.append(s);cols.append(n if done else t);weights.append(prob)
                rewards[action,s]+=prob*reward
        rows.append(n);cols.append(n);weights.append(1.)
        matrices.append(sp.csr_matrix((weights,(rows,cols)),shape=(n+1,n+1)))
    mdp=FiniteMDP(matrices,rewards,gamma,name)
    # Verify model assembly against the original Gymnasium transition tuples.
    for seed in [0,1,2]:
        v=np.random.default_rng(seed).normal(size=n+1);v[n]=0
        direct=np.array([[sum(p*(r+(0 if done else gamma*v[t]))
                          for p,t,r,done in env.P[s][action]) for s in range(n)] for action in range(a)])
        np.testing.assert_allclose(mdp.action_values(v)[:,:n],direct,atol=1e-12)
    meta=dict(name=name,gymnasium=gym.__version__,original_states=n,augmented_states=n+1,
              actions=a,discount=gamma,added_sink=n,kwargs=opts,
              initialization='constant reward lower bound on ALL augmented states',
              terminal_handling='terminated transitions redirected to zero-reward absorbing sink',
              construction_verified_against_official_tuples=True)
    env.close()
    return mdp,meta

