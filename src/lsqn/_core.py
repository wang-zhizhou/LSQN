"""Portable extraction of the original protected-learning LSQN solver.

The arithmetic/control-flow source is preserved from the archived solver;
changes are local imports and passive diagnostics/callbacks. The public API
fixes the protected-learning/sign-correction path used by the main protocol.
See core-port-notes.txt for source hashes and the exact change scope.
"""
import time
import numpy as np
from scipy.linalg.blas import dger
from .options import SolverOptions as CGBOptions
from ._support import Envelope, _active_actions


class ColumnBuffer:
    """Geometrically growing column storage; no mathematical truncation."""
    def __init__(self,n):
        self.n=n;self.used=0;self.capacity=0
        self.data=np.empty((n,0),order='F')
    @property
    def view(self): return self.data[:,:self.used]
    def append(self,x):
        if self.used==self.capacity:
            self.capacity=min(self.n,max(4,2*self.capacity))
            data=np.empty((self.n,self.capacity),order='F')
            data[:,:self.used]=self.view
            self.data=data
        self.data[:,self.used]=x;self.used+=1
        return self.view


class InverseModel:
    """H = beta I + A C^T until dense storage becomes cheaper. No truncation."""
    def __init__(self, n, beta=1., dense=False, fast=False):
        self.n, self.beta = n, float(beta)
        self.fast=fast
        self.h = self.beta*np.eye(n,order='F' if fast else 'C') if dense else None
        self.ab,self.cb=ColumnBuffer(n),ColumnBuffer(n)
        self.a, self.c = np.empty((n,0)), np.empty((n,0))
        self.peak = n*n if dense else 0
        self.allocated_peak=self.peak
        self.updates = 0
        self.conversions = 0

    def apply(self, f):
        if self.h is not None: return self.h@f
        return self.beta*f + self.a@(self.c.T@f)

    def update(self, a, c):
        self.updates += 1
        if self.h is not None:
            if self.fast:
                self.h=dger(1.,a,c,a=self.h,overwrite_a=1)
            else:
                self.h += np.outer(a,c)
            return
        if self.fast:
            self.a,self.c=self.ab.append(a),self.cb.append(c)
        else:
            self.a = np.column_stack((self.a,a))
            self.c = np.column_stack((self.c,c))
        self.peak = max(self.peak,self.a.size+self.c.size)
        self.allocated_peak=max(self.allocated_peak,
            self.ab.data.size+self.cb.data.size if self.fast else self.a.size+self.c.size)
        if 2*self.a.shape[1] >= self.n:
            self.h = self.beta*np.eye(self.n)+self.a@self.c.T
            if self.fast: self.h=np.asfortranarray(self.h)
            self.a,self.c = np.empty((self.n,0)),np.empty((self.n,0))
            self.ab,self.cb=ColumnBuffer(self.n),ColumnBuffer(self.n)
            self.peak = max(self.peak,self.h.size)
            self.allocated_peak=max(self.allocated_peak,self.h.size)
            self.conversions += 1


class GroupedEnvelope:
    """Separable pairings without assembling one sparse row per active action.

With multiple active actions, evaluate full transition products for each action.
Counts include even the computed rows that the mask subsequently discards.
"""
    def __init__(self,mdp,mask,policy,stats):
        self.mdp,self.mask,self.stats=mdp,mask,stats
        self.p=mdp.policy_matrix(policy)
        self.active_count=int(mask.sum())
    def j(self,x):
        self.stats['policy_products']+=1
        return x-self.mdp.gamma*self.p.dot(x)
    def pairing(self,f,x,upper=True):
        n=self.mdp.n_states
        if self.active_count==n:
            self.stats['support_row_products']+=n
            return float(f@(x-self.mdp.gamma*self.p.dot(x)))
        self.stats['support_row_products']+=n*self.mdp.n_actions
        values=f[None,:]*(x[None,:]-self.mdp.gamma*np.vstack([p.dot(x) for p in self.mdp.transitions]))
        if upper: return float(np.max(np.where(self.mask,values,-np.inf),axis=0).sum())
        return float(np.min(np.where(self.mask,values,np.inf),axis=0).sum())


def solve_core(mdp, opts=None, v0=None, dense=False, protect=True, full_completion=True, persistent=False, correction='sign', width_fraction=.1, grouped_support=True, memory_mode='retain', cache_proposal=False, fast_model=True, buffered_basis=True, callback=None, inverse_model_factory=None, envelope_factory=None, reuse_learning_image=False, certificate_mode='original', basis_projector=None, basis_protector=None, learning_query=None):
    opts = opts or CGBOptions(tol=1e-8,max_outer=500)
    Model = InverseModel if inverse_model_factory is None else inverse_model_factory
    EnvelopeFactory = GroupedEnvelope if envelope_factory is None else envelope_factory
    n = mdp.n_states
    projection_work = np.empty(n) if basis_projector is not None else None
    if memory_mode not in ('retain','reset','no_trajectory','consistent_trajectory'):
        raise ValueError('unknown memory mode')
    if memory_mode!='retain' and persistent:
        raise ValueError('memory ablations require nonpersistent inner bases')
    if persistent and buffered_basis: raise ValueError('buffering requires fresh per-policy bases')
    if correction not in ('projection','sign'): raise ValueError('unknown correction')
    if grouped_support and correction!='sign': raise ValueError('grouped support requires sign correction')
    if width_fraction is not None and not (0<width_fraction<1-opts.descent_fraction):
        raise ValueError('width fraction must be between zero and 1 - descent fraction')
    if not (0<opts.forcing_trigger<1 and 0<opts.descent_fraction<=1-opts.forcing_trigger):
        raise ValueError('invalid forcing/descent parameters')
    if opts.step_cap_factor < 1+opts.forcing_trigger: raise ValueError('invalid cap')
    if not (0<opts.armijo_sigma<.5 and 0<opts.backtrack_beta<1): raise ValueError('invalid line search')
    start = time.perf_counter()
    H = Model(n,opts.initial_scale,dense,fast_model)
    retired_updates=retired_conversions=retired_peak=0
    stats = dict(bellman_calls=0,policy_products=0,policy_transpose_products=0,
                 support_row_products=0,gradient_requests=0,qp_calls=0,
                 enrichments=0,model_applies=0,corrections=0,backtracks=0,
                 peak_inner_columns=0,peak_protected_columns=0,sign_corrections=0,
                 trajectory_updates=0,model_resets=0,skipped_switch_updates=0,
                 cached_proposal_updates=0,cache_refreshes=0)
    stats['peak_basis_allocated_scalars']=0
    history=[]
    trace=[]
    v=np.zeros(n) if v0 is None else np.array(v0,dtype=float,copy=True)
    def emit(event, **extra):
        row=dict(k=len(history), elapsed_s=time.perf_counter()-start,
                 residual2=float(np.linalg.norm(f)),
                 residual_inf=float(np.linalg.norm(f,np.inf)))
        if trace and trace[-1]['k']==row['k']:
            trace[-1]=row
        else:
            trace.append(row)
        if callback is not None:
            record=dict(row, event=event, value=v.copy(),
                        accepted_step=dict(history[-1]) if event=='accepted' else None,
                        queries=int(stats['enrichments']))
            record.update(extra)
            callback(record)
    def bellman(x):
        stats['bellman_calls']+=1
        q=mdp.action_values(x); pi=q.argmax(axis=0)
        return x-q[pi,np.arange(n)],pi,q
    def proposal():
        stats['model_applies']+=1
        return -H.apply(f)
    def finish(ok,message):
        stats.update(peak_model_scalars=max(retired_peak,H.peak),
                     model_updates=retired_updates+H.updates,
                     conversions=retired_conversions+H.conversions)
        stats['all_model_applies']=stats['model_applies']+stats['model_updates']
        stats['model_allocated_scalars_last_model_peak']=H.allocated_peak
        emit('finished', converged=bool(ok), message=message)
        return dict(converged=ok,message=message,iterations=len(history),
                    residual=float(np.linalg.norm(f,np.inf)),runtime_s=time.perf_counter()-start,
                    stats=stats,history=history,trace=trace,value=v.tolist(),policy=pi.tolist())
    f,pi,q=bellman(v); previous=None; stable=0
    emit('initial')
    U,Q,S=[np.empty((n,0)) for _ in range(3)]
    for k in range(opts.max_outer):
        R=float(np.linalg.norm(f)); ri=float(np.linalg.norm(f,np.inf))
        if ri<=opts.tol: return finish(True,'converged')
        if not np.isfinite(R): return finish(False,'nonfinite residual')
        if memory_mode=='reset' and k>0:
            retired_updates+=H.updates
            retired_conversions+=H.conversions
            retired_peak=max(retired_peak,H.peak)
            H=Model(n,opts.initial_scale,dense,fast_model)
            stats['model_resets']+=1
        switches=0 if previous is None else int(np.count_nonzero(previous!=pi))
        stable=stable+1 if previous is not None and switches==0 else 0
        if grouped_support:
            best=q.max(axis=0)
            width=(opts.tie_atol+opts.tie_rtol*np.maximum(1.,abs(best)))*min(1.,R) if width_fraction is None else width_fraction*R/np.sqrt(n)
            env=EnvelopeFactory(mdp,best[None,:]-q<=width,pi,stats)
            active_count=env.active_count
        else:
            active=(_active_actions(q,opts.tie_atol*min(1.,R),opts.tie_rtol*min(1.,R))
                    if width_fraction is None else _active_actions(q,width_fraction*R/np.sqrt(n),0.))
            env=Envelope(mdp,active,pi,stats)
            active_count=len(env.states)
        local=full_completion or (opts.certified_local_completion and ri<=opts.local_completion_residual
               and stable>=opts.local_completion_stable_steps)
        eta=opts.forcing_trigger
        if local: eta=min(eta,opts.local_forcing_coefficient*R**opts.local_forcing_exponent)
        eta=max(eta,10*np.finfo(float).eps)
        budget=n if local else min(n,opts.max_self_secant_enrichments)
        # U: orthonormal queried primal directions; Q: orthonormal image directions.
        # S: exact inverse images, J S = Q and H Q = S during this frozen iteration.
        if not persistent or switches>0:
            U,Q,S=[np.empty((n,0)) for _ in range(3)]
            ub,qb,sb=[ColumnBuffer(n) for _ in range(3)]
        budget=min(budget,n-U.shape[1])
        p=proposal(); jp=env.j(p); forcing=np.linalg.norm(f+jp)/R
        initial_forcing=float(forcing)
        enrich=0
        try:
            for j in range(budget):
                if forcing<=eta: break
                if learning_query is not None:
                    u,z,w,change=learning_query(H,env,p,jp,U,Q,S,protect)
                else:
                    reuse_first_image=reuse_learning_image and U.shape[1]==0
                    u=p.copy()
                    if basis_projector is None:
                        for _ in range(2): u-=U@(U.T@u)
                    else:
                        basis_projector(U,u,projection_work)
                    if np.linalg.norm(u)<=np.finfo(float).eps*max(1.,np.linalg.norm(p)):
                        reuse_first_image=False
                        idx=int(np.argmax(1-np.sum(U*U,axis=1)))
                        u=np.eye(1,n,idx).ravel()-U@U[idx,:]
                        if basis_projector is None:
                            for _ in range(2): u-=U@(U.T@u)
                        else:
                            basis_projector(U,u,projection_work)
                    un=np.linalg.norm(u)
                    if un<=np.finfo(float).tiny: return finish(False,'primal basis breakdown')
                    u/=un; y=(jp/un if reuse_first_image else env.j(u))
                    if reuse_learning_image:
                        stats['reused_learning_products']=stats.get('reused_learning_products',0)+int(reuse_first_image)
                    z=y.copy(); w=u.copy()
                    if protect:
                        if basis_protector is None:
                            for _ in range(2):
                                coeff=Q.T@z; z-=Q@coeff; w-=S@coeff
                        else:
                            basis_protector(Q,S,z,w,projection_work)
                    zn=np.linalg.norm(z)
                    if zn<=np.finfo(float).tiny: return finish(False,'image basis breakdown')
                    z/=zn; w/=zn
                    change=w-H.apply(z)
                H.update(change,z)
                U=ub.append(u) if buffered_basis else np.column_stack((U,u))
                if protect:
                    Q=qb.append(z) if buffered_basis else np.column_stack((Q,z))
                    S=sb.append(w) if buffered_basis else np.column_stack((S,w))
                stats['peak_basis_allocated_scalars']=max(stats['peak_basis_allocated_scalars'],
                    (ub.data.size+qb.data.size+sb.data.size if buffered_basis else U.size+Q.size+S.size)
                    +(0 if projection_work is None else projection_work.size))
                enrich+=1
                if cache_proposal:
                    p=p-change*(z@f)
                    stats['cached_proposal_updates']+=1
                else:
                    p=proposal()
                jp=env.j(p); forcing=np.linalg.norm(f+jp)/R
                # Refresh once a cached proposal passes, so finite-precision
                # acceptance still certifies the actual stored-model action.
                if cache_proposal and forcing<=eta:
                    p=proposal(); jp=env.j(p); forcing=np.linalg.norm(f+jp)/R
                    stats['cache_refreshes']+=1
            stats['enrichments']+=enrich
            stats['peak_inner_columns']=max(stats['peak_inner_columns'],enrich)
            stats['peak_protected_columns']=max(stats['peak_protected_columns'],Q.shape[1])
            if local and forcing>eta*(1+1e-8): return finish(False,'local forcing not attained')
            cap=opts.step_cap_factor*np.sqrt(n)*R/(1-mdp.gamma)
            # Uniform forcing already bounds ||p|| by (1+eta) ||F|| / sigma_lower.
            # Only legacy, bounded-budget experiments need an explicit cap.
            capped=(not full_completion) and np.linalg.norm(p)>cap
            if capped: p*=cap/np.linalg.norm(p)
            target=opts.descent_fraction*R*R
            d=p
            if certificate_mode=='cached' and grouped_support and not capped:
                slope=env.pairing_cached(f,d,jp)
            elif certificate_mode=='singleton' and grouped_support and active_count==n and not capped:
                slope=float(f@jp)
            else:
                slope=env.pairing(f,d)
            mu=0.
            if slope>-target:
                if correction=='sign':
                    g=R*np.sign(f)
                    required=(1-mdp.gamma)*R*np.linalg.norm(f,1)
                    stats['sign_corrections']+=1
                else:
                    g=env.gradient(f)
                    required=opts.projection_alignment_fraction*(1-mdp.gamma)**2/n*R*R
                m=env.pairing(f,g,upper=False)
                if not np.isfinite(m) or m<required*(1-1e-10):
                    return finish(False,'alignment failure')
                mu=(slope+target)/m; d=p-mu*g; slope=env.pairing(f,d)
                if slope>-target:
                    mu+=(slope+target)/m; d=p-mu*g; slope=env.pairing(f,d)
                stats['corrections']+=1
            if not np.isfinite(slope) or slope>-target+1e-10*target:
                return finish(False,'direction certificate failed')
            alpha=1.
            for bt in range(opts.max_backtracks+1):
                candidate=v+alpha*d; fc,pic,qc=bellman(candidate)
                if .5*(fc@fc)<=.5*R*R+opts.armijo_sigma*alpha*slope: break
                alpha*=opts.backtrack_beta
            else: return finish(False,'line search failed')
            stats['backtracks']+=bt
            history.append(dict(iteration=k,residual=ri,alpha=alpha,mu=float(mu),
                                residual2_before=R,residual2_after=float(np.linalg.norm(fc)),slope=float(slope),
                                forcing=float(forcing),target=float(eta),local=bool(local),
                                enrichments=enrich,switches=switches,capped=bool(capped),
                                active_rows=active_count,initial_forcing=initial_forcing,
                                bellman_calls=stats['bellman_calls'],policy_products=stats['policy_products'],
                                support_row_products=stats['support_row_products']))
            s,y=candidate-v,fc-f
            skip_switch=(memory_mode=='consistent_trajectory' and not np.array_equal(pi,pic))
            if skip_switch: stats['skipped_switch_updates']+=1
            if memory_mode!='no_trajectory' and not skip_switch and y@y>np.finfo(float).tiny:
                if persistent and np.array_equal(pi,pic):
                    z,w=y.copy(),s.copy()
                    for _ in range(2):
                        coeff=Q.T@z; z-=Q@coeff; w-=S@coeff
                    u=s.copy()
                    for _ in range(2): u-=U@(U.T@u)
                    zn,un=np.linalg.norm(z),np.linalg.norm(u)
                    if un>1e-10*np.linalg.norm(s) and zn>1e-10*np.linalg.norm(y):
                        z/=zn; w/=zn
                        H.update(w-H.apply(z),z)
                        Q=np.column_stack((Q,z)); S=np.column_stack((S,w))
                        U=np.column_stack((U,u/un))
                        stats['peak_protected_columns']=max(stats['peak_protected_columns'],Q.shape[1])
                    # If dependent, no update: protected image already implies this secant.
                else:
                    H.update(s-H.apply(y),y/(y@y))
                    stats['trajectory_updates']+=1
            previous=pi.copy(); v,f,pi,q=candidate,fc,pic,qc
            emit('accepted')
        except (ArithmeticError,np.linalg.LinAlgError) as exc:
            return finish(False,str(exc))
    return finish(bool(np.linalg.norm(f,np.inf)<=opts.tol),'iteration limit')
