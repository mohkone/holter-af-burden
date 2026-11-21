"""Independent regression checks for study-level sampling, calibration and aggregates."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import pandas as pd
from study import draw,hquant,intervals,ROOT,RES,read_inputs
from pilot_estimators import simulate,levels,stable_seed,strata_for_minutes

def small():
    rng=np.random.default_rng(235)
    for N in (20,61,257):
        raw=rng.integers(0,13,N)/12
        st=strata_for_minutes(np.arange(N),raw)
        y=rng.random(N);prob=np.linspace(0,1,13)**1.2;v=np.arange(1,7)*.01
        state=draw(y,st,prob[levels(raw)],v,seed=233,reps=200)
        _,p,plans=simulate(raw,st,y,prob,v,seed=233,reps=200)
        np.testing.assert_allclose(state['mu'],p[2,0],atol=2e-15)
        np.testing.assert_allclose(state['var'],p[2,1],atol=2e-15)
        np.testing.assert_allclose(state['hoeff'],p[2,2],atol=2e-15)
        assert sum(state['sizes'])==20
        if N==20:
            for lo,hi in intervals(state,{'radius':.08,'multiplier':2.}).values():
                np.testing.assert_allclose(lo,y.mean(),atol=1e-10);np.testing.assert_allclose(hi,y.mean(),atol=1e-10)
    scores=np.arange(42000).reshape(42,1000)/42000
    expected=sorted(list(scores.ravel())+[np.inf]*1000)[38699]
    assert hquant(scores,42)==expected
    print('Sampling and variance agree with the independent previous-pilot routine; complete census and hierarchical ranks verified.')

def audit():
    s=pd.read_csv(RES/'summary.csv')
    for ff in sorted(RES.glob('records_*.csv')):
        d=pd.read_csv(ff,dtype={'record':str,'patient':str})
        co,det=d.iloc[0][['cohort','detector']]
        for method,t in d.groupby('method'):
            srow=s[(s.cohort==co)&(s.band=='overall')&(s.detector==det)&(s.method==method)].iloc[0]
            bypt={}
            for row in t.itertuples():
                bypt.setdefault(row.patient,[]).append([row.coverage,row.width_pp])
            pts=np.stack([np.asarray(x).mean(0) for x in bypt.values()])
            np.testing.assert_allclose(pts.mean(0),[srow.coverage,srow.width_pp],atol=1e-11)
            assert abs(np.mean(pts[:,0]<.8)-srow.under80)<1e-12
            assert np.allclose(t.coverage*1000,np.round(t.coverage*1000))
            assert srow.patients==len(bypt) and srow.records==len(t)
        print(co,det,len(set(d.record)),'record rows checked, including all patient-mean coverage/width and under-80 indicators.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',action='store_true');a=ap.parse_args();small()
    if a.results:audit()
