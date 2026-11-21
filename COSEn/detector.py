"""Short-RR-window COSEn detector for a simulated one-minute Holter audit.

The inputs are exclusively the original recording header and *automatic* QRS file.
Both cohorts' N marks are retained. No adjudicated beat or rhythm types are inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import numpy as np
from detector_acquire import ROOT, records

PAIR_I, PAIR_J = np.triu_indices(11,k=1)
COHORT_MAP = {'LTAFDB':'LTA','SHDB':'SHDB'}

def read_beats(path, header_fs):
    """Read MIT/WFDB annotation times; skip nonbeat events while preserving timeline."""
    payload = path.read_bytes()
    at = 0
    position = 0
    peaks = []
    event_counts = {}
    while at+2 <= len(payload):
        lo,hi=payload[at],payload[at+1]
        at+=2
        event=hi>>2
        interval=((hi&3)<<8)+lo
        if event==0 and interval==0:
            break
        if event==59:
            # WFDB stores a signed 32-bit displacement as two LE words, high word first.
            if at+4>len(payload): raise ValueError(f'Truncated SKIP in {path}')
            large=int.from_bytes(payload[at+2:at+4]+payload[at:at+2],byteorder='little',signed=True)
            position+=large
            at+=4
        elif event==63:
            if at+interval+(interval%2)>len(payload): raise ValueError(f'Truncated AUX in {path}')
            # Check optional WFDB time-base metadata; other AUX payloads are skipped.
            if position==0:
                val=payload[at:at+interval]
                if val.startswith(b'## time resolution:') and float(val.split(b':')[1])!=header_fs:
                    raise ValueError(f'QRS clock does not match header in {path}')
            at+=interval+(interval%2)
        elif event in (60,61,62):
            pass
        else:
            position+=interval
            event_counts[event]=event_counts.get(event,0)+1
            if event==1:
                peaks.append(position)
    if at!=len(payload) and any(payload[at:]):
        raise ValueError(f'Unexpected nonzero trailing data in {path}')
    result=np.asarray(peaks,dtype=np.int64)
    if len(result)==0 or np.any(np.diff(result)<0):
        raise ValueError(f'Automatic beats absent or unordered in {path}')
    return result,event_counts

def cosen(rr_ticks, fs):
    """COSEn on rows of 12 intervals. All pair distances expressed as integer ticks."""
    x=np.asarray(rr_ticks)
    if x.ndim!=2 or x.shape[1]!=12 or np.any(x<=0): raise ValueError('Need 12 finite positive RR intervals per row')
    d0=np.abs(x[:,PAIR_I]-x[:,PAIR_J])
    dnext=np.abs(x[:,PAIR_I+1]-x[:,PAIR_J+1])
    d01=np.maximum(d0,dnext)
    # Smallest tolerance r >= 30 ms giving at least five length-two matches.
    fifth=np.partition(d01,4,axis=1)[:,4]
    r_ticks=np.maximum(float(fs)*.030,fifth)
    A=np.sum(d01 <= r_ticks[:,None],axis=1)
    B=np.sum(d0 <= r_ticks[:,None],axis=1)
    return np.log(B/A)+np.log(2*r_ticks/x.mean(axis=1))

def predict(peaks, fs, duration):
    """Use six equally spaced interrogation points per complete clock minute."""
    nminutes=int(duration//(60*fs))
    times=(5+10*np.arange(6*nminutes))*fs
    # First beat at/after the midpoint and competitor just before it; earlier wins ties.
    nxt=np.searchsorted(peaks,times).clip(0,len(peaks)-1)
    prev=(nxt-1).clip(min=0)
    center=np.where(abs(peaks[prev]-times)<=abs(peaks[nxt]-times),prev,nxt)
    starts=center-6
    active=(starts>=0)&(starts+12<len(peaks))&(np.abs(peaks[center]-times)<=2.5*fs)
    windows=np.flatnonzero(active)
    raw=np.full(len(times),np.nan)
    if len(windows):
        rr=np.diff(peaks)
        ticks=rr[starts[windows,None]+np.arange(12)]
        plausible=(ticks.min(axis=1)>=.25*fs)&(ticks.max(axis=1)<=2.5*fs)
        raw[windows[plausible]]=cosen(ticks[plausible],fs)
    votes=np.where(np.isfinite(raw),(raw>-1.4).astype(float),.5)
    return {'idx':np.arange(nminutes,dtype=np.int32),'score':votes.reshape(-1,6).mean(axis=1),
            'invalid_fraction':(~np.isfinite(raw)).reshape(-1,6).mean(axis=1)}

def run(cohorts):
    table=[]
    for source in cohorts:
        cohort=COHORT_MAP[source]
        d=ROOT/'raw_detector'/source
        dst=ROOT/'derived'/'detector'/cohort
        dst.mkdir(exist_ok=True,parents=True)
        recs,checks=records(source,True)
        for record in recs:
            header=d/(record+'.hea');qrs=d/(record+'.qrs')
            for file in [header,qrs]:
                if hashlib.sha256(file.read_bytes()).hexdigest()!=checks[file.name]:
                    raise ValueError(f'Incorrect source file {file}')
            first=header.read_text().splitlines()[0].split()
            fs,n=float(first[2].split('/')[0]),int(first[3])
            peaks,types=read_beats(qrs,fs)
            # Some detector outputs include peaks outside the signal end. Never extrapolate them.
            usable=peaks[(peaks>=0)&(peaks<n)]
            pred=predict(usable,fs,n)
            out=dst/(record+'.npz');np.savez_compressed(out,**pred)
            row=dict(cohort=cohort,record=record,fs=fs,n_samples=n,n_minutes=len(pred['idx']),
                     retained_auto_N=len(usable),out_of_bounds=int(len(peaks)-len(usable)),
                     duplicate_peak_times=int(np.sum(np.diff(usable)==0)),excluded_T=int(types.get(19,0)),metadata_types=' '.join(f'{i}:{c}' for i,c in sorted(types.items()) if i!=1),
                     invalid_proportion=float(pred['invalid_fraction'].mean()),sha256=hashlib.sha256(out.read_bytes()).hexdigest())
            table.append(row)
        print(f'{cohort}: {len(recs)} reconstructed',flush=True)
    tablefile=ROOT/'tables'/'detector_manifest.csv'
    if tablefile.exists() and len(cohorts)<len(COHORT_MAP):
        with tablefile.open() as f:
            other=[r for r in csv.DictReader(f) if r['cohort'] not in {COHORT_MAP[c] for c in cohorts}]
        table=other+table
    with tablefile.open('w') as f:
        wr=csv.DictWriter(f,fieldnames=table[0]);wr.writeheader();wr.writerows(table)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=list(COHORT_MAP),action='append')
    run(p.parse_args().cohort or list(COHORT_MAP))
