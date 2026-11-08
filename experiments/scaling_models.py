"""Original independent FrozenLake scaling model construction.

Frozen bundled models are used by default. These functions allow regeneration.
Source SHA256: 20b25d0042aa43ac947c4400e4642f5c8ccfe0aad94569bdba89fcbc3ce1afc6
"""
import numpy as np
import scipy.sparse as sp

SOURCES = {
    'FrozenLake': 'https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/frozen_lake.py',
    'Taxi': 'https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/taxi.py',
    'CliffWalking': 'https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/toy_text/cliffwalking.py',
}


MAPS = {
    4: ['SFFF', 'FHFH', 'FFFH', 'HFFG'],
    8: ['SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF',
        'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG'],
}


def assemble(table, gamma, name, terminal_states):
    from lsqn import MDP as FiniteMDP
    n, actions = len(table), len(table[0])
    rewards = np.zeros((actions, n))
    matrices = []
    for a in range(actions):
        rows, cols, weights = [], [], []
        for s in range(n):
            assert abs(sum(x[0] for x in table[s][a])-1) < 1e-13
            for prob, dest, reward in table[s][a]:
                rows.append(s); cols.append(dest); weights.append(prob)
                rewards[a, s] += prob*reward
        matrices.append(sp.csr_matrix((weights, (rows, cols)), shape=(n, n)))
    mdp = FiniteMDP(matrices, rewards, gamma, name)
    # Independent tuple-based check of the assembled Bellman operator.
    rng = np.random.default_rng(90210)
    for _ in range(3):
        v = rng.normal(size=n)
        direct = np.array([[sum(p*(r+gamma*v[t]) for p,t,r in table[s][a])
                            for s in range(n)] for a in range(actions)])
        np.testing.assert_allclose(mdp.action_values(v), direct, atol=1e-12, rtol=1e-12)
    return mdp, dict(terminal_states=list(terminal_states), n_states=n,
                    n_nonterminal=n-len(terminal_states), n_actions=actions,
                    transition_nnz=sum(p.nnz for p in matrices))


def valid_map(board):
    n = len(board); seen = {(0, 0)}; todo = [(0, 0)]
    while todo:
        r, c = todo.pop()
        if (r, c) == (n-1, n-1): return True
        for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if 0 <= rr < n and 0 <= cc < n and board[rr][cc] != 'H' and (rr,cc) not in seen:
                seen.add((rr,cc)); todo.append((rr,cc))
    return False


def random_map(side, seed, frozen_probability=.9):
    rng = np.random.default_rng(seed)
    for attempt in range(100000):
        board = rng.choice(['F','H'], (side,side), p=[frozen_probability,1-frozen_probability])
        board[0,0] = 'S'; board[-1,-1] = 'G'
        board = [''.join(row) for row in board]
        if valid_map(board): return board
    raise RuntimeError('Failed to generate a connected map')


def frozen_lake(board, gamma):
    side = len(board); n = side*side
    assert all(len(row) == side for row in board)
    shifts = [(0,-1),(1,0),(0,1),(-1,0)]
    terminals = [s for s in range(n) if board[s//side][s%side] in 'HG']
    table = [[[] for _ in range(4)] for _ in range(n)]
    for s in range(n):
        row, col = divmod(s, side)
        for a in range(4):
            if s in terminals:
                table[s][a] = [(1., s, 0.)]
                continue
            for b in ((a-1)%4, a, (a+1)%4):
                dr, dc = shifts[b]
                rr = min(side-1, max(0,row+dr)); cc = min(side-1,max(0,col+dc))
                table[s][a].append((1/3, rr*side+cc, float(board[rr][cc]=='G')))
    # Public 4x4 map, state 0, left: two outcomes remain at 0, one moves down.
    if board == MAPS[4]:
        assert table[0][0] == [(1/3,0,0.),(1/3,0,0.),(1/3,4,0.)]
    mdp, meta = assemble(table, gamma, 'FrozenLake-slippery', terminals)
    meta.update(board=board, slippery=True, source=SOURCES['FrozenLake'])
    return mdp, meta


