"""Apply frozen ArNet2 to new automatic beat arrays using only frequency/time inputs."""
import argparse
import numpy as np
from arnet import HERE, digest, predict, save_manifest
from pathlib import Path

def run(source,cohort):
    files=sorted((source/cohort).glob('*.npz'))
    if not files: raise RuntimeError(f'No automatic beat archives found for {cohort}')
    target=HERE/'derived'/cohort;target.mkdir(exist_ok=True,parents=True)
    rows=[]
    for i,path in enumerate(files):
        rec=path.stem
        # The three fields below are the only arrays read from the source NPZ.
        with np.load(path,allow_pickle=False) as f:
            fs=float(f['fs'].item());n=int(f['n_samples'].item());peaks=f['peaks']
        out=predict(peaks,fs,n);dest=target/(rec+'.npz');np.savez_compressed(dest,**out)
        rows.append(dict(cohort=cohort,record=rec,n_minutes=len(out['idx']),n_peaks=len(peaks),
                         missing_fraction=out['invalid_fraction'].mean(),sha256=digest(dest)))
        if i%10==0:print(cohort,i+1,'/',len(files),flush=True)
    save_manifest(cohort,rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--cohort',required=True)
    a=ap.parse_args();run(a.source,a.cohort)
