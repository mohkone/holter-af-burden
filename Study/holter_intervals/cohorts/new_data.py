"""Derive first-day eligible minutes and independent automatic beats for IRIDIA-AF.

The reference burden comes from ECG adjudications. RR annotation CSVs are used
only to audit source clocks after writing predictor-independent positions.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import h5py
from iridia_fetch import main as fetch, ROOT

def read(f):return list(csv.DictReader(f.open()))
def write(out,rows):
    out.parent.mkdir(exist_ok=True,parents=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()

def fractions(starts,stops,n_samples,fs):
    unit=60*fs;idx=np.arange(n_samples//unit,dtype=np.int32)
    start=np.asarray(starts,dtype=np.int64).clip(0,n_samples)
    stop=np.asarray(stops,dtype=np.int64).clip(0,n_samples)
    if len(start)!=len(stop) or np.any(stop<start) or np.any(start[1:]<stop[:-1]):raise ValueError('Labels unordered or overlapping')
    grid=unit*np.arange(len(idx)+1)
    # Integrated time under all expert-marked AF intervals at minute boundaries.
    integrated=np.zeros(len(grid),dtype=np.int64)
    for a,b in zip(start,stop):integrated+=(grid-a).clip(0,b-a)
    return idx,np.diff(integrated)/unit

def iridia():
    raw=ROOT/'raw'/'IRIDIA';out=ROOT/'derived'/'IRIDIA';out.mkdir(exist_ok=True,parents=True)
    md=read(raw/'metadata.csv'); receipt=json.loads((ROOT/'iridia_source_receipt.json').read_text())
    if len(md)!=167 or len({r['patient_id'] for r in md})!=152:raise ValueError('IRIDIA record/subject count')
    records=[];audits=[]
    for row in md:
        rid=row['record_id'];patient=row['patient_id'];fs=200;day=86400*fs
        sources=[raw/(rid+suffix) for suffix in ('_rr_00.h5','_ecg_labels.csv','_rr_labels.csv')]
        for f in sources:
            if sha(f)!=receipt['members'][f.name]['sha256']:raise ValueError('source identity: '+f.name)
        with h5py.File(sources[0]) as h:rr=h['rr'][:]
        if np.any(rr<=0) or np.any(rr%5):raise ValueError('nonpositive or off-grid RR interval: '+rid)
        # Microport RR intervals are milliseconds with a 5-ms sampling grid.
        # The first R boundary is placed at t=0; the first 60 s are never reviewed.
        peaks=np.r_[0,np.cumsum(rr,dtype=np.int64)//5]
        n=min(day,int(row['record_samples']))
        discrepancy=(peaks[-1]-n)/fs
        if abs(discrepancy)>60:raise ValueError('Automatic RR clock differs from ECG time by over a minute: '+rid)
        peaks=peaks[peaks<n].astype(np.int32)
        af=read(sources[1])
        starts=[day*int(a['start_file_index'])+int(a['start_qrs_index']) for a in af]
        stops=[day*int(a['end_file_index'])+int(a['end_qrs_index']) for a in af]
        idx,y=fractions(starts,stops,n,fs);valid=idx>0 # 30-second calibration overlaps minute zero
        for j in np.where(np.diff(peaks)>fs*60)[0]:
            lo,hi=peaks[j:j+2]
            # Short irregular intervals are legitimate predictor inputs. Only
            # unobserved multi-minute runs would compromise the review frame.
            valid[(idx*60*fs<hi)&((idx+1)*60*fs>lo)]=False
        y[~valid]=np.nan
        target=out/(rid+'.npz')
        np.savez_compressed(target,record=rid,patient=patient,fs=fs,n_samples=n,idx=idx,valid=valid,y=y,peaks=peaks)
        # Check timestamps, never fit a correction from human boundary labels.
        rr_labels=read(sources[2]);errors=[]
        if len(af)!=len(rr_labels):raise ValueError('label/QA interval files differ: '+rid)
        cumulative=np.r_[0,np.cumsum(rr,dtype=np.int64)//5]
        for a,b in zip(af,rr_labels):
            for which in ['start','end']:
                if a[which+'_file_index']==b[which+'_file_index']=='0':
                    k=int(b[which+'_rr_index']); q=int(a[which+'_qrs_index'])
                    if 0<=k and k+1<len(cumulative):
                        errors.append(min(abs(cumulative[k]-q),abs(cumulative[k+1]-q))/fs)
        audits.append(dict(record=rid,rr_duration_minus_source_s=discrepancy,pairs_checked=len(errors),
                           median_pair_distance_s=np.median(errors) if errors else '',max_pair_distance_s=max(errors,default=''),
                           largest_RR_s=rr.max()/1000))
        records.append(dict(cohort='IRIDIA',record=rid,patient=patient,source='10.5281/zenodo.8405941',
                            minutes_total=len(idx),minutes_eligible=int(valid.sum()),fs=fs,n_samples=n,source_peaks=len(peaks),
                            npz_sha256=sha(target)))
    write(ROOT/'iridia_clock_audit.csv',audits)
    write(ROOT/'iridia_manifest.csv',records)
    from merge_manifests import merge
    merge()
    print('IRIDIA:',len(records),'records,',len(set(r['patient'] for r in records)),'patients,',sum(r['minutes_eligible'] for r in records),'minutes.')
if __name__=='__main__':
    pp=argparse.ArgumentParser();pp.add_argument('--offline',action='store_true');args=pp.parse_args()
    if not args.offline:fetch()
    iridia()
