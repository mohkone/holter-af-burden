"""Calibrate on LTA alone, lock the parameters, then evaluate fixed external datasets."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import ndtri
from pilot_estimators import stable_seed,allocate,levels,strata_for_minutes,calibrate,record_stats
from finite import bounded_intervals,profile_intervals

ROOT=Path(__file__).resolve().parent
RES=ROOT/'results'
REPEATS=1000
METHODS=('normal','hoeffding','direct_kl','profile','fixed_radius','scaled_radius')

def read_inputs(det,co):
    tab=pd.read_csv(ROOT/f'inputs/{det}_{co}.csv',dtype={'record':str,'patient':str})
    out=[]
    for row in tab.itertuples(index=False):
        if not row.eligible:continue
        path=ROOT/row.path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row.sha256:raise ValueError(path)
        with np.load(path) as z:
            ix=z['idx'];y=z['y'];raw=z['raw12'].astype(float)/12
        st=strata_for_minutes(np.arange(len(raw)),raw)[ix]
        out.append((row.record,row.patient,y,raw[ix],st))
    return out

def draw(y,st,p,popvar,seed,reps=REPEATS):
    N=len(y);n=np.bincount(st,minlength=6);k=allocate(n,np.ones(6),20)
    group=[np.flatnonzero(st==h) for h in range(6)]
    sums=np.zeros((reps,6));mu=np.full(reps,p.mean());var=np.zeros(reps)
    design=0.;hoeff=0.;q=np.full(6,.5)
    rng=np.random.default_rng(seed)
    for h,ix in enumerate(group):
        if not len(ix):continue
        w=n[h]/N
        q[h]=np.clip(p[ix].mean(),.01,.99)
        if k[h]<n[h]:
            design+=w*w*(1-k[h]/n[h])*popvar[h]/k[h]
            hoeff+=w*w*(1+np.ptp(p[ix]))**2/k[h]
    for start in range(0,reps,100):
        sl=slice(start,min(reps,start+100));u=rng.random((sl.stop-sl.start,N))
        for h,ix in enumerate(group):
            if not len(ix):continue
            selected=ix[np.argpartition(u[:,ix],kth=k[h]-1,axis=1)[:,:k[h]]]
            yread=y[selected];residual=yread-p[selected];w=n[h]/N
            sums[sl,h]=yread.sum(axis=1)
            mu[sl]+=w*residual.mean(axis=1)
            if k[h]<n[h]:var[sl]+=w*w*(1-k[h]/n[h])*residual.var(axis=1,ddof=1)/k[h]
    scale=np.maximum(.001,np.sqrt(.5*(var+design)))
    return dict(mu=mu,var=var,scale=scale,hoeff=hoeff,counts=n,sizes=k,sums=sums,q=q)

def hquant(x,records):
    # Equal-sized rows of independent campaigns within each one-record cluster.
    a=np.asarray(x)
    if a.shape!=(records,REPEATS):raise ValueError(a.shape)
    rank=int(np.ceil(.90*(records+1)*REPEATS))
    if rank>a.size:return np.inf
    return float(np.partition(a.ravel(),rank-1)[rank-1])

def train(det):
    rows=read_inputs(det,'LTA')
    split=sorted(rows,key=lambda r:hashlib.sha256(('holter-int-v1|'+r[0]).encode()).digest())
    if len(split)!=84 or len({s[1] for s in split})!=84:raise ValueError('Unexpected LTA separation')
    fit,cal=split[:42],split[42:]
    by,vr=calibrate(np.stack([record_stats(raw,st,y) for rec,pt,y,raw,st in fit]).sum(0))
    diff=[];scaled=[]
    for rec,pt,y,raw,st in cal:
        state=draw(y,st,by[levels(raw)],vr,stable_seed('holter-interval', 'LTA',rec))
        err=np.abs(np.clip(state['mu'],0,1)-y.mean());diff.append(err);scaled.append(err/state['scale'])
    cfg=dict(detector=det,fit_records=[r[0] for r in fit],calibration_records=[r[0] for r in cal],
             p=by.tolist(),residual_variance=vr.tolist(),radius=hquant(diff,len(cal)),multiplier=hquant(scaled,len(cal)))
    (RES/f'model_{det}.json').write_text(json.dumps(cfg,indent=2)+'\n')

def intervals(state,cfg):
    mu=state['mu'];ctr=np.clip(mu,0,1);a=.1
    def around(x,h):return np.maximum(0,x-h),np.minimum(1,x+h)
    hh=np.sqrt(.5*np.log(2/a)*state['hoeff'])
    n,k,sums=state['counts'],state['sizes'],state['sums'];census=np.array_equal(n,k)
    return {'normal':around(mu,ndtri(.95)*np.sqrt(state['var'])),
            'hoeffding':around(mu,hh),
            'direct_kl':bounded_intervals(n,k,sums,a),
            'profile':profile_intervals(n,k,sums,state['q'],a),
            'fixed_radius':around(ctr,0 if census else cfg['radius']),
            'scaled_radius':around(ctr,0 if census else cfg['multiplier']*state['scale'])}

def evaluate(det,co):
    cfg=json.loads((RES/f'model_{det}.json').read_text());p=np.array(cfg['p']);v=np.array(cfg['residual_variance'])
    rows=[]
    for rec,pt,y,raw,st in read_inputs(det,co):
        state=draw(y,st,p[levels(raw)],v,stable_seed('holter-interval',co,rec))
        truth=y.mean()
        for method,(lo,hi) in intervals(state,cfg).items():
            width=np.maximum(hi-lo,0);covered=(lo<=truth+1e-12)&(hi>=truth-1e-12)
            rows.append(dict(cohort=co,detector=det,record=rec,patient=pt,minutes=len(y),true_percent=100*truth,
                method=method,coverage=covered.mean(),width_pp=100*width.mean(),median_width_pp=100*np.median(width),
                empty=(lo>hi+1e-12).mean(),zero=(width<=1e-12).mean(),miss_low=(truth<lo-1e-12).mean(),miss_high=(truth>hi+1e-12).mean(),
                residual_point_mae_pp=100*np.abs(np.clip(state['mu'],0,1)-truth).mean()))
    pd.DataFrame(rows).to_csv(RES/f'records_{co}_{det}.csv',index=False)

def aggregate():
    frames=[pd.read_csv(x,dtype={'patient':str,'record':str}) for x in sorted(RES.glob('records_*.csv'))]
    d=pd.concat(frames,ignore_index=True)
    d['band']=np.select([d.true_percent<=1e-10,d.true_percent<=5+1e-10,d.true_percent>=100-1e-10],['zero','low','full'],default='intermediate')
    metrics=['coverage','width_pp','empty','zero','miss_low','miss_high','residual_point_mae_pp']
    allrows=[]; comparisons=[];pr=[]
    for (co,det),frame in d.groupby(['cohort','detector']):
        for subgroup,f in [('overall',frame)]+list(frame.groupby('band')):
            patient=f.groupby(['method','patient'])[metrics].mean().reset_index()
            if subgroup=='overall':pr.append(patient.assign(cohort=co,detector=det))
            pts=sorted(set(patient.patient));m=len(pts);rng=np.random.default_rng(stable_seed('holter-int-bootstrap',co,subgroup))
            ix=rng.integers(m,size=(10000,m))
            widths=patient.pivot(index='patient',columns='method',values='width_pp').loc[pts]
            for method,t in patient.groupby('method'):
                t=t.set_index('patient').loc[pts];row=dict(cohort=co,detector=det,band=subgroup,method=method,
                    patients=m,records=f.record.nunique(),**t[metrics].mean().to_dict())
                for field,z in [('coverage',t.coverage.to_numpy()),('width_pp',t.width_pp.to_numpy()),('under80',(t.coverage.to_numpy()<.8).astype(float))]:
                    row[field]=float(z.mean())
                    row[field+'_lo'],row[field+'_hi']=np.quantile(z[ix].mean(axis=1),[.025,.975])
                row['width_ratio']=row['width_pp']/widths.hoeffding.mean()
                row['mean_target']=row['coverage']>=.88 and row['coverage_lo']>=.85 and row['width_pp']<=30 and row['width_ratio']<=.70
                row['individual_target']=row['mean_target'] and row['under80']<=.10
                allrows.append(row)
                for ref in ['hoeffding','direct_kl','fixed_radius','normal']:
                    if ref==method:continue
                    diff=(widths[method]-widths[ref]).to_numpy()
                    lo,hi=np.quantile(diff[ix].mean(axis=1),[.025,.975])
                    comparisons.append(dict(cohort=co,detector=det,band=subgroup,method=method,reference=ref,width_diff=diff.mean(),lo=lo,hi=hi))
    pd.DataFrame(allrows).to_csv(RES/'summary.csv',index=False)
    pd.DataFrame(comparisons).to_csv(RES/'paired_widths.csv',index=False)
    pd.concat(pr).to_csv(RES/'patients.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate','aggregate']);ap.add_argument('--det',choices=['COSEn','ArNet2']);ap.add_argument('--cohort')
    a=ap.parse_args()
    if a.stage=='train':train(a.det)
    elif a.stage=='evaluate':evaluate(a.det,a.cohort)
    else:aggregate()
