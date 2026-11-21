"""Build portable minute inputs from the prior pilot and new independent cohorts."""
from pathlib import Path
import argparse
import hashlib
import os
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent/'holter_pilot'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(cohort,det):
    first=cohort in ('LTA','SHDB')
    if first:
        meta=pd.read_csv(PILOT/'tables/reference_manifest.csv',dtype=str)
    else:
        manifest='afdb_manifest.csv' if cohort=='AFDB' else 'manifest.csv'
        meta=pd.read_csv(ROOT/'cohorts'/manifest,dtype=str)
    meta=meta[meta.cohort==cohort]
    if not len(meta):raise ValueError('Cohort not yet available: '+cohort)
    records=[]
    for m in meta.itertuples(index=False):
        rec=m.record
        if first:
            with np.load(PILOT/f'derived/reference/{cohort}/{rec}.npz') as z:
                idx,y=z['idx'],z['y']
        else:
            with np.load(ROOT/f'cohorts/derived/{cohort}/{rec}.npz') as z:
                mask=z['valid'].astype(bool)
                idx,y=z['idx'][mask],z['y'][mask]
        if det=='ArNet2':
            with np.load(ROOT/f'second/derived/{cohort}/{rec}.npz') as z:
                di,raw=z['idx'],z['score']
        elif first:
            with np.load(PILOT/f'derived/detector/{cohort}/{rec}.npz') as z:
                di,raw=z['idx'],z['score']
        else:
            sys.path.insert(0,str(PILOT))
            from detector import predict
            with np.load(ROOT/f'cohorts/derived/{cohort}/{rec}.npz') as z:
                out=predict(z['peaks'],int(z['fs']),int(z['n_samples']))
            di,raw=out['idx'],out['score']
        keep=idx<1440;idx=idx[keep];y=y[keep]
        raw=raw[di<1440];di=di[di<1440]
        if not np.array_equal(di,np.arange(len(raw))) or not np.isin(idx,di).all():raise ValueError((cohort,rec,'clock'))
        if np.any(np.diff(idx)<=0) or not np.isfinite(y).all() or np.any((y<0)|(y>1)) or not np.isfinite(raw).all() or not np.allclose(raw,np.rint(raw*12)/12):raise ValueError((cohort,rec,'data'))
        rel=f'inputs/{det}/{cohort}/{rec}.npz';path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,idx=idx.astype(np.int32),y=y.astype(np.float64),raw12=np.rint(12*raw).astype(np.uint8))
        records.append(dict(cohort=cohort,detector=det,record=rec,patient=m.patient,
                            minutes=len(idx),clock=len(di),eligible=len(idx)>=20,path=rel,sha256=digest(path)))
    pd.DataFrame(records).to_csv(ROOT/f'inputs/{det}_{cohort}.csv',index=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cohort');ap.add_argument('detector',choices=['ArNet2','COSEn'])
    args=ap.parse_args();prepare(args.cohort,args.detector)
