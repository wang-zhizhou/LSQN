"""MPI and QPI-BT/S extracted from the local equation-based comparators.

No author-binary claim is made. QPI-S failure guards remain unchanged.
"""
import time
import numpy as np
from ._trace import noop


def _init(mdp,v0):
    return np.array(mdp.common_lower_bound() if v0 is None else v0,dtype=float,copy=True)



def solve_mpi_cached(mdp,v0=None,tol=1e-8,max_outer=30000,depth=4,observe=noop):
    """Same MPI recurrence as frozen baseline, assembling P once per policy."""
    start=time.perf_counter();v=_init(mdp,v0);history=[]
    stats=dict(bellman_calls=0,policy_products=0)
    for k in range(max_outer+1):
        tv,pi=mdp.bellman(v);stats['bellman_calls']+=1
        res=float(np.linalg.norm(tv-v,np.inf))
        observe(v, v-tv, k, history[-1] if k and history else None)
        if res<=tol or k==max_outer:break
        P=mdp.policy_matrix(pi);reward=mdp.policy_reward(pi);x=tv.copy()
        for _ in range(depth-1):
            x=reward+mdp.gamma*P.dot(x);stats['policy_products']+=1
        history.append(dict(iteration=k,residual=res,bellman_calls=stats['bellman_calls'],policy_products=stats['policy_products']))
        v=x
    return dict(converged=res<=tol,message='converged' if res<=tol else 'outer budget',iterations=k,
                residual=res,value=v.tolist(),policy=pi.tolist(),history=history,stats=stats,runtime_s=time.perf_counter()-start)



def qpi_gain(mdp,v,tv,policy,d):
    gamma=mdp.gamma
    def g(x):return x+gamma/(1-gamma)*np.mean(x)
    w=tv-mdp.policy_reward(policy)-gamma*np.mean(v)
    gw=g(w);u=v-np.mean(v);denom=u@(v-gw)
    out=g(d)
    if abs(u@v)>1e-28 and abs(denom)>1e-28:
        out=out+(u@d/denom)*gw
    return out



def secant_transition_update(P,H,C,B,gamma):
    """Solve min ||Pnew-P||F subject Pnew C=B; small rank-revealing SVD."""
    left,s,right=np.linalg.svd(C,full_matrices=False)
    keep=s>1e-12*max(s[0],np.finfo(float).tiny)
    if not np.any(keep):raise ArithmeticError('empty constraint span')
    # V is the orthonormal domain basis. U is the correction on that basis.
    V=left[:,keep]
    U=(B@right[keep,:].T)/s[keep][None,:]-P@V
    feasibility=np.linalg.norm((P@C+U@(V.T@C))-B)
    if feasibility>1e-8*max(1.,np.linalg.norm(B)):
        raise ArithmeticError('inconsistent numerical constraints')
    HU=H@U;VH=V.T@H;small=np.eye(U.shape[1])-gamma*(V.T@HU)
    if not np.all(np.isfinite(small)) or np.linalg.cond(small)>1e14:
        raise ArithmeticError('ill-conditioned Woodbury update')
    Hnew=H+gamma*HU@np.linalg.solve(small,VH)
    return P+U@V.T,Hnew,feasibility



def solve_qpi_bt(mdp,v0=None,tol=1e-8,max_outer=30000,secant=False,observe=noop):
    start=time.perf_counter();v=_init(mdp,v0);n=mdp.n_states;gamma=mdp.gamma
    stats=dict(bellman_calls=0,policy_products=0,backtracks=0,dense_updates=0)
    history=[];last=None;P=H=None
    def bell(x):stats['bellman_calls']+=1;return mdp.bellman(x)
    tv,pi=bell(v)
    def finish(ok,msg):
        observe(v, v-tv, len(history), history[-1] if history else None)
        return dict(converged=ok,message=msg,iterations=len(history),residual=float(np.linalg.norm(v-tv,np.inf)),
                    value=v.tolist(),policy=pi.tolist(),history=history,stats=stats,runtime_s=time.perf_counter()-start)
    if secant:
        P=np.full((n,n),1/n);H=np.eye(n)+gamma/(1-gamma)*P
    for k in range(max_outer):
        f=v-tv;res=float(np.linalg.norm(f,np.inf))
        observe(v, f, k, history[-1] if k and history else None)
        if res<=tol:return finish(True,'converged')
        try:
            if secant:
                ones=np.ones(n)
                if last is None:
                    C=np.column_stack((ones,v));B=np.column_stack((ones,(tv-mdp.policy_reward(pi))/gamma))
                else:
                    vp,tp,pip=last;s=v-vp;y=(tv-tp)/gamma
                    if np.array_equal(pi,pip):
                        C=np.column_stack((ones,v,s));B=np.column_stack((ones,(tv-mdp.policy_reward(pi))/gamma,y))
                    else:C=np.column_stack((ones,s));B=np.column_stack((ones,y))
                P,H,feas=secant_transition_update(P,H,C,B,gamma);stats['dense_updates']+=1
                direction=H@f-f
            else:direction=qpi_gain(mdp,v,tv,pi,f)-f
            alpha=1.
            for bt in range(81):
                candidate=tv-alpha*direction;tc,pc=bell(candidate)
                if np.linalg.norm(candidate-tc,np.inf)<=.5*(1+gamma)*res:break
                alpha*=.5
            else:return finish(False,'backtracking budget')
        except (ArithmeticError,np.linalg.LinAlgError) as exc:return finish(False,str(exc))
        history.append(dict(iteration=k,residual=res,alpha=alpha,bellman_calls=stats['bellman_calls'],
                            switches=0 if last is None else int(np.count_nonzero(pi!=last[2]))))
        stats['backtracks']+=bt;last=(v.copy(),tv.copy(),pi.copy());v,tv,pi=candidate,tc,pc
    return finish(False,'outer budget')

