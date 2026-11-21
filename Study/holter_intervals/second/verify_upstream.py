"""Re-run the pinned authors' official command on synthetic signals, then compare."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile
import hashlib
import numpy as np
import pandas as pd
from arnet import native_labels, to_minutes

def main(args):
    official=args.runner.resolve();model=args.model.resolve()
    orig=hashlib.sha1(f'blob {official.stat().st_size}\0'.encode()+official.read_bytes()).hexdigest()
    if orig!=args.blob:raise ValueError('Unexpected upstream runner')
    os.environ['ARNET2_MODEL_DIR']=str(model);os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    rng=np.random.default_rng(11217)
    for n,fs in [(47,128),(113,200)]:
        bpm=rng.choice([68,72,79,115,128],size=n)
        noisy=rng.integers(0,2,n)
        seconds=60/bpm[:,None]+rng.normal(0,np.where(noisy,.16,.022)[:,None],size=(n,60))
        rr=(np.maximum(seconds,.36)*fs).round().astype(np.int64)
        peaks=np.r_[4*fs,4*fs+np.cumsum(rr)]
        with tempfile.TemporaryDirectory() as t:
            tmp=Path(t)
            pd.DataFrame({'rr':rr.ravel()/fs,'t':peaks[1:]/fs,'id':np.ones(n*60,dtype=int)}).to_csv(tmp/'input.csv',index=False)
            env=dict(os.environ,TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1')
            subprocess.run([sys.executable,str(official),'--input_file',str(tmp/'input.csv'),
                            '--saved_model',str(model),'--save_output_path',str(tmp)],env=env,check=True,stdout=subprocess.DEVNULL)
            expected=pd.read_csv(tmp/'predictions.csv').pred.to_numpy()
            actual=native_labels(rr/fs)
            assert np.array_equal(expected,actual)
            end=peaks[-1]+21*fs
            labels=to_minutes(peaks,fs,end,actual)['score']
            check=[]
            for t in (5+10*np.arange(6*(end//(60*fs))))*fs:
                w=np.searchsorted(peaks[60::60],t,side='right')
                check.append(.5 if t<peaks[0] or w>=n else expected[w])
            assert np.array_equal(labels,np.asarray(check).reshape(-1,6).mean(1))
            print(f'{n} native labels and {len(labels)} projected minute scores agree with upstream synthetic replay at {fs} Hz.')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--runner',required=True,type=Path);ap.add_argument('--blob',required=True)
    ap.add_argument('--model',required=True,type=Path);main(ap.parse_args())
