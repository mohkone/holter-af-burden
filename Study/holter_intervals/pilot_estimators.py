"""Finite-population design and patient-independent model calibration."""
import hashlib
import numpy as np
from scipy.optimize import isotonic_regression
from scipy.special import ndtri

METHODS=('raw','calibrated','uniform_review','uniform_corrected','proportional_corrected','informed_corrected')
LEVELS=13
STRATA=6

def stable_seed(*parts):
    key='|'.join(str(x) for x in parts).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little')

def strata_for_minutes(idx, raw):
    continuous=(np.diff(idx)==1)&(np.abs(np.diff(raw))>=.5-1e-10)
    edge=np.r_[False,continuous]|np.r_[continuous,False]
    band=np.where(raw<=1/6+1e-10,0,np.where(raw>=5/6-1e-10,2,1))
    return 2*band+edge.astype(int)

def levels(raw):
    code=np.rint(raw*12).astype(int)
    if (code<0).any() or (code>12).any() or not np.allclose(code/12,raw,rtol=0,atol=2e-7):
        raise ValueError('Scores must come from six binary/half votes')
    return code

def record_stats(raw,strata,y):
    key=levels(raw)*STRATA+strata
    return np.array([np.bincount(key,weights=a,minlength=LEVELS*STRATA).reshape(LEVELS,STRATA)/len(y)
                     for a in (np.ones(len(y)),y,y*y)])

def calibrate(stat):
    n,sy,sy2=stat
    n=np.maximum(n,0.)
    counts=n.sum(axis=1); loc=np.flatnonzero(counts>1e-9)
    phat=isotonic_regression(sy.sum(axis=1)[loc]/counts[loc],weights=counts[loc]).x
    p=np.clip(np.interp(np.arange(LEVELS),loc,phat),0,1)
    sz=n.sum(axis=0); r=(sy-n*p[:,None]).sum(axis=0)
    r2=(sy2-2*sy*p[:,None]+n*p[:,None]**2).sum(axis=0)
    overall=max(0,float(r2.sum()/sz.sum()-(r.sum()/sz.sum())**2))
    shat=np.divide(r*r,sz,out=np.zeros_like(r),where=sz>1e-10)
    var=(np.maximum(0,r2-shat)+.1*overall)/(sz+.1)
    return p,np.maximum(var,.03**2)

def allocate(n,var,budget):
    n=np.asarray(n,dtype=int)
    if np.any(n<0) or budget>n.sum():raise ValueError('Invalid population or budget')
    take=np.minimum(2,n)
    if take.sum()>budget:raise ValueError('Budget cannot cover active strata')
    while take.sum()<budget:
        improve=np.zeros(len(n));ok=take<n
        improve[ok]=n[ok]**2*var[ok]/(take[ok]*(take[ok]+1.))
        improve[~ok]=-np.inf
        take[int(np.argmax(improve))]+=1
    return take

def add_contribution(vals, pred, sel, state, N, population, offset):
    """Update estimates, design variances and bounded-outcome Hoeffding factors."""
    draws=vals[sel]-pred[sel]; m=sel.shape[1];W=population/N
    state[0]+=W*draws.mean(axis=1)
    if m<population:
        state[1]+=W**2*(1-m/population)*draws.var(axis=1,ddof=1)/m
        state[2]+=W**2*(1+offset)**2/m

def simulate(raw, st, y, p_by_level, var, seed, reps=1000,budget=20):
    y=np.asarray(y,dtype=float);N=len(y)
    p=p_by_level[levels(raw)]
    sets=[np.flatnonzero(st==h) for h in range(STRATA)]
    n=np.array(list(map(len,sets)))
    plans=[allocate(n,v,budget) for v in [np.ones(STRATA),var]]
    # Common independent random priorities implement SRS without replacement
    # for the entire list and independent within-stratum SRS for both allocations.
    rng=np.random.default_rng(seed); res=[]
    for start in range(0,reps,100):
        nr=min(reps-start,100); key=rng.random((nr,N))
        subset=np.argpartition(key,kth=budget-1,axis=1)[:,:budget]
        states=[np.zeros((3,nr)) for _ in range(4)]
        zero=np.zeros_like(p)
        add_contribution(y,zero,subset,states[0],N,N,0)
        for state in states[1:]:state[0].fill(p.mean())
        add_contribution(y,p,subset,states[1],N,N,float(np.ptp(p)))
        for h,ix in enumerate(sets):
            if not len(ix):continue
            k=max(plan[h] for plan in plans)
            weights=key[:,ix];part=np.argpartition(weights,kth=k-1,axis=1)[:,:k]
            priorities=np.take_along_axis(weights,part,axis=1)
            ranked=ix[np.take_along_axis(part,np.argsort(priorities,axis=1),axis=1)]
            for j,plan in enumerate(plans):
                sub=ranked[:,:plan[h]]
                add_contribution(y,p,sub,states[j+2],N,len(ix),float(np.ptp(p[ix])))
        res.append(np.stack(states))
    full=np.concatenate(res,axis=2)
    return p,full,plans

def summaries(raw, y, p, full, alpha=.10):
    truth=float(np.mean(y))
    rows=[]
    def scalar(x):
        return dict(mae_pp=100*abs(x-truth),signed_error_pp=100*(x-truth),
                    bias_unclipped_pp=100*(x-truth),mean_prediction_pp=100*x,
                    mae_mcse_pp=0.,cover=np.nan,width_pp=np.nan,zero_width=np.nan,
                    bounded_cover=np.nan,bounded_width_pp=np.nan,empty_interval=np.nan)
    for x in [np.mean(raw),np.mean(p)]:rows.append(scalar(x))
    for est,var,bound in full:
        pred=np.clip(est,0,1);err=np.abs(pred-truth)
        half=ndtri(1-alpha/2)*np.sqrt(var)
        interval=lambda h: (np.maximum(est-h,0),np.minimum(est+h,1))
        lo,hi=interval(half)
        blo,bhi=interval(np.sqrt(.5*np.log(2/alpha)*bound))
        covered=lambda a,b: ((a<=truth+1e-12)&(b>=truth-1e-12)).mean()
        rows.append(dict(mae_pp=float(100*err.mean()),signed_error_pp=float(100*(pred.mean()-truth)),
                         bias_unclipped_pp=float(100*(est.mean()-truth)),mean_prediction_pp=100*pred.mean(),
                         mae_mcse_pp=100*np.std(err,ddof=1)/np.sqrt(len(err)),cover=covered(lo,hi),
                         width_pp=100*np.maximum(0,hi-lo).mean(),zero_width=np.mean(np.maximum(0,hi-lo)<=1e-12),
                         empty_interval=np.mean(hi<lo-1e-12),
                         bounded_cover=covered(blo,bhi),bounded_width_pp=100*np.maximum(0,bhi-blo).mean()))
    return [dict(method=m,**v) for m,v in zip(METHODS,rows)]
