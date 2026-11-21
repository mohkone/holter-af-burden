"""Run the authors' fixed ArNet2 signature on automatically detected beat times."""
import argparse
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@lru_cache(maxsize=1)
def load_model():
    # The runtime is optional for replay from deposited minute predictions.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('ARNET2_THREADS','4')))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    path=Path(os.environ.get('ARNET2_MODEL_DIR',HERE/'exported_model'))
    manifest=json.loads((HERE/'sources.json').read_text())['files']
    for name,expected in manifest.items():
        if digest(path/name)!=expected['sha256']:
            raise ValueError(f'Unexpected model weight or graph: {name}')
    return tf.saved_model.load(str(path)).signatures['predict_fixed']

def native_labels(windows):
    if not len(windows):
        return np.empty(0,dtype=np.int32)
    import tensorflow as tf
    fn=load_model()
    n=len(windows)
    out=fn(x=np.asarray(windows,dtype=np.float32),prec_windows=np.arange(n,dtype=np.int32),
           glob_lab=np.ones(n,dtype=np.float32),ids=tf.constant(['record']*n))
    pred=np.asarray(out['pred'])
    if pred.shape!=(n,) or not np.all((pred==0)|(pred==1)) or not np.isfinite(out['probs']).all():
        raise ValueError('Invalid output from fixed upstream model')
    return pred

def to_minutes(peaks,fs,n_samples,labels):
    """Look up predicted class of the 60-RR window covering each query point."""
    n_min=int(n_samples//(60*fs));q=(5+10*np.arange(6*n_min))*fs
    vote=np.full(len(q),.5);valid=np.zeros(len(q),dtype=bool)
    if len(labels):
        rr_index=np.searchsorted(peaks,q,side='right')-1
        valid=(rr_index>=0)&(rr_index<60*len(labels))
        where=np.flatnonzero(valid);ix=rr_index[where]
        steps=(peaks[ix+1]-peaks[ix])/fs
        valid[where]=(steps>=.25)&(steps<=2.5)
        vote[valid]=labels[rr_index[valid]//60]
    return dict(idx=np.arange(n_min,dtype=np.int32),score=vote.reshape(-1,6).mean(axis=1),
                invalid_fraction=(~valid).reshape(-1,6).mean(axis=1))

def predict(peaks,fs,n_samples):
    peaks=np.asarray(peaks,dtype=np.int64)
    if fs<=0 or n_samples<0 or peaks.ndim!=1 or np.any(np.diff(peaks)<0):
        raise ValueError('Timebase/beat input invalid')
    peaks=peaks[(peaks>=0)&(peaks<n_samples)]
    ticks=np.diff(peaks)
    n=len(ticks)//60
    windows=(ticks[:60*n].reshape(n,60)/fs).astype('float32')
    return to_minutes(peaks,fs,n_samples,native_labels(windows))

def run_pilot(pilot,cohort):
    sys.path.insert(0,str(pilot))
    from detector import read_beats
    raw=pilot/'raw_detector'/({'LTA':'LTAFDB','SHDB':'SHDB'}[cohort])
    with (pilot/'tables'/'detector_manifest.csv').open() as file:
        records=[d for d in csv.DictReader(file) if d['cohort']==cohort]
    target=HERE/'derived'/cohort;target.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for i,row in enumerate(records):
        rec=row['record'];fs=float(row['fs']);n=int(row['n_samples'])
        peaks,counts=read_beats(raw/(rec+'.qrs'),fs)
        peaks=peaks[(peaks>=0)&(peaks<n)]
        out=predict(peaks,fs,n);dest=target/(rec+'.npz')
        np.savez_compressed(dest,**out)
        receipts.append(dict(cohort=cohort,record=rec,n_minutes=len(out['idx']),n_peaks=len(peaks),
                             missing_fraction=out['invalid_fraction'].mean(),sha256=digest(dest)))
        if i%10==0:print(cohort,i+1,'/',len(records),flush=True)
    save_manifest(cohort,receipts)
def save_manifest(cohort,rows):
    if not rows:return
    dest=HERE/('manifest_'+cohort+'.csv')
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--cohort',choices=['LTA','SHDB'],required=True)
    args=ap.parse_args(); run_pilot(args.pilot.resolve(),args.cohort)
