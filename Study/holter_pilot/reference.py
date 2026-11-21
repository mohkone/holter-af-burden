"""Acquire and derive AF time fractions from the original rhythm annotations."""
import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'raw'
TABLES = ROOT/'tables'
OUT = ROOT/'derived'/'reference'
COLLECTIONS={'LTA':'ltafdb/1.0.0','SHDB':'shdb-af/1.0.1'}
MIRRORS=('https://physionet-open.s3.amazonaws.com','https://physionet.org/files')
CATALOGUE_SHA256={'LTA':'9ad85f6b46d896b7a5579c70fd05bef1e39b8e2b1aeb2b2bdfde465c4aec3ad1',
                  'SHDB':'a101475f41df8300493b9cba963499334fc5aa118c0b4c01503e7bf7621ac3c9'}
def hash_bytes(x): return hashlib.sha256(x).hexdigest()

def get_file(cohort,name,expected=None,offline=False):
    path=RAW/cohort/name
    if path.is_file() and (not expected or hash_bytes(path.read_bytes())==expected):
        return path
    if offline:
        raise FileNotFoundError(f'Missing or damaged cached {cohort}/{name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    for prefix in (MIRRORS[0],MIRRORS[0],MIRRORS[1]):
        url=f'{prefix}/{COLLECTIONS[cohort]}/{name}'
        if errors:url+='?full=1'
        try:
            with urlopen(Request(url,headers={'User-Agent':'Holter research pilot v1'}),timeout=35) as response:
                data=response.read()
                if response.status!=200: raise OSError(str(response.status))
            if expected and hash_bytes(data)!=expected:raise ValueError('checksum mismatch')
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                f.write(data)
            os.replace(f.name,path)
            return path
        except (OSError,ValueError) as exc:errors.append(f'{type(exc).__name__} {exc}')
    raise OSError(f'{cohort}/{name}: {errors}')

def digest_map(cohort,offline=False):
    path=get_file(cohort,'SHA256SUMS.txt',CATALOGUE_SHA256[cohort],offline)
    return {n.strip().removeprefix('./'):h for h,n in map(str.split,path.read_text().splitlines())}

def datasets(offline=False):
    all_digests={}
    records=[]
    for co in COLLECTIONS:
        hashes=digest_map(co,offline)
        all_digests[co]=hashes
        if co=='LTA':
            names=get_file(co,'RECORDS',hashes.get('RECORDS'),offline).read_text().split()
            assert len(set(names))==84
            records.extend((co,n,n) for n in names)
        else:
            meta=get_file(co,'AdditionalData.csv',hashes['AdditionalData.csv'],offline)
            with meta.open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            marked=[(x['Data_ID'], x['Subject_ID']) for x in rows if x['Annotated'].lower()=='true']
            assert len(rows)==128 and len(marked)==98 and len(set(s for n,s in marked))==93
            if {a.removesuffix('.atr') for a in hashes if a.endswith('.atr')}!={n for n,s in marked}:
                raise ValueError('Metadata versus annotation file mismatch')
            # Per-subject pseudocode; this analysis does not require other clinical fields.
            ordered=sorted(set(s for n,s in marked))
            subjects={s:'S'+str(i+1).zfill(3) for i,s in enumerate(ordered)}
            records.extend((co,n,subjects[s]) for n,s in marked)
    return records,all_digests

def acquire(offline=False):
    records,digests=datasets(offline)
    work=[(co,r+suffix,digests[co][r+suffix]) for co,r,s in records for suffix in ('.hea','.atr')]
    def load(a):
        co,name,digest=a
        p=get_file(co,name,digest,offline)
        return dict(cohort=co,filename=name,bytes=p.stat().st_size,sha256=digest,
                    url=f'{MIRRORS[0]}/{COLLECTIONS[co]}/{name}')
    with ThreadPoolExecutor(max_workers=14) as pool:manifest=list(pool.map(load,work))
    save_csv('reference_sources.csv',manifest)
    meta=[]
    for co,names in [('LTA',['SHA256SUMS.txt','RECORDS','tables.html']),('SHDB',['SHA256SUMS.txt','AdditionalData.csv'])]:
        for name in names:
            expected=digests[co].get(name)
            path=get_file(co,name,expected,offline)
            meta.append(dict(cohort=co,file=name,sha256=hash_bytes(path.read_bytes()),size=path.stat().st_size,
                             url=f'{MIRRORS[0]}/{COLLECTIONS[co]}/{name}',publisher_sha256=expected or ''))
    save_csv('reference_catalogues.csv',meta)
    return records

def save_csv(name,rows):
    TABLES.mkdir(parents=True,exist_ok=True)
    with (TABLES/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Source-defined rhythm transitions are AUX texts prefixed with '('.
# Final reference timestamps delimit ascertainable coverage of a recording.
KNOWN={'LTA':{'N','SVTA','VT','AFIB','B','T','IVR','AB','SBR'},
       'SHDB':{'N','AFIB','AFL','AT','PAT','NOD','AB'}}
UNKNOWN={'?','U','UNKNOWN','NOISE','ARTIFACT','UNKN'}

def decode_annotations(content, return_events=False):
    """Extract labeled rhythm starts, annotation timebase and last actual annotation.

    WFDB short words use little endian, with a high-order six-bit code. SKIP
    holds a signed 32-bit time in two little-endian words, high word first.
    AUX payload words and field-modifier words do not advance the clock.
    """
    if len(content)%2:raise ValueError('Odd annotation file length')
    words=np.frombuffer(content,dtype='<u2')
    if len(words)==0 or words[-1]!=0:raise ValueError('No WFDB end of file')
    codes=words>>10
    primary=(codes>0)&(codes<59)
    delta=(words&1023).astype(np.int64)
    offsets=np.flatnonzero(codes>=59)
    aux=[]
    fs=None
    resume=0
    for i in offsets:
        if i<resume:continue
        code=int(codes[i])
        n=1
        if code==59:
            if i+2>=len(words):raise ValueError('Truncated SKIP')
            skip=(int(words[i+1])<<16)|int(words[i+2])
            if skip>=2**31:skip-=2**32
            delta[i]=skip
            n=3
        elif code==63:
            count=int(words[i]&1023)
            n=1+(count+1)//2
            if i+n>len(words):raise ValueError('Truncated AUX')
            note=content[2*i+2:2*i+2+count].decode('latin1').rstrip('\x00')
            if note.startswith('('):
                aux.append((i,note[1:].strip()))
            if note.startswith('## time resolution:'):
                fs=float(note.split(':')[-1].strip())
            delta[i]=0
        else:
            delta[i]=0
        delta[i+1:i+n]=0
        primary[i:i+n]=False
        resume=i+n
    t=np.cumsum(delta)
    result=([(int(t[i]),name) for i,name in aux],int(t[-1]),fs)
    if return_events:return (*result,t[primary],codes[primary])
    return result

def minute_fractions(starts,rhythms,end,fs,n_samples,cohort):
    """Integrate over [change,next_change); minute length is elapsed 60 seconds."""
    if int(fs)!=fs or fs<=0:raise ValueError('Invalid sample rate')
    unit=int(fs)*60
    nfull=n_samples//unit
    seconds_grid=np.arange(nfull+1,dtype=np.int64)*unit
    start=np.asarray(starts,dtype=np.int64)
    if len(start)==0:raise ValueError('Missing explicit rhythm')
    if np.any(np.diff(start)<0) or start.min()<0 or start[-1]>end:
        raise ValueError('Invalid rhythm order')
    allowed=KNOWN[cohort]|UNKNOWN
    if not set(rhythms)<=allowed:raise ValueError(f'Unexpected rhythms {set(rhythms)-allowed}')
    # Bound the final interval to the final annotated event and the header.
    boundaries=np.clip(np.r_[start,end],0,n_samples)
    durations=np.diff(boundaries)
    def elapsed(mask):
        cumul=np.r_[0,np.cumsum(durations*np.array(mask))]
        idx=np.searchsorted(boundaries,seconds_grid,side='right')-1
        idx=np.clip(idx,0,len(boundaries)-1)
        residual=np.maximum(seconds_grid-boundaries[idx],0)*np.r_[mask,False][idx]
        return np.diff(cumul[idx]+residual)
    ok=elapsed([r in KNOWN[cohort] for r in rhythms])==unit
    af=elapsed([r=='AFIB' for r in rhythms])
    return np.flatnonzero(ok),af[ok].astype(float)/unit,int(np.sum(durations*(np.array(rhythms)=='AFIB')))

def derive(records):
    manifest=[]
    audit=[]
    for co,r,patient in records:
        hea=(RAW/co/(r+'.hea')).read_text().splitlines()[0].split()
        fs=int(hea[2]);n_samples=int(hea[3])
        marks,end,ann_fs,events,codes=decode_annotations((RAW/co/(r+'.atr')).read_bytes(),True)
        if ann_fs is not None and ann_fs!=fs: raise ValueError(f'Timebase mismatch: {co}/{r} {ann_fs} {fs}')
        if not marks:raise ValueError(f'No rhythm {co}/{r}')
        times,annotations=zip(*marks)
        idx,y,total=minute_fractions(times,annotations,end,fs,n_samples,co)
        # A minute that overlaps a full minute without independently annotated
        # beats cannot simulate continuous adjudication of that interval.
        reference_beats=events[np.isin(codes,[1,5,8,13])] if co=='LTA' else events[events>0]
        gap_starts=np.flatnonzero(np.diff(reference_beats)>=fs*60)
        supported=np.ones(len(idx),dtype=bool)
        for j in gap_starts:
            supported &= ~((idx*fs*60<reference_beats[j+1]) & ((idx+1)*fs*60>reference_beats[j]))
        dropped_gaps=int((~supported).sum());idx,y=idx[supported],y[supported]
        p=OUT/co/f'{r}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,idx=idx,y=y)
        manifest.append(dict(cohort=co,record=r,patient=patient,fs=fs,n_samples=n_samples,
              n_full_minutes=n_samples//(fs*60),n_eligible=len(idx),n_excluded=n_samples//(fs*60)-len(idx),
              first_label_sample=times[0],last_annotation_sample=end,unknown_prefix_seconds=times[0]/fs,
              excluded_tail_seconds=max(n_samples-end,0)/fs,annotation_beyond_header_seconds=max(end-n_samples,0)/fs,
              n_gap_over_minute=len(gap_starts),gap_excluded_minutes=dropped_gaps,n_marks=len(marks),rhythms='|'.join(sorted(set(annotations))),total_annotated_af_s=total/fs,
              eligible_af_seconds=y.sum()*60,npz_sha256=hash_bytes(p.read_bytes())))
        from collections import Counter
        audit.extend(dict(cohort=co,record=r,rhythm=label,count=count) for label,count in Counter(annotations).items())
    save_csv('reference_manifest.csv',manifest)
    save_csv('reference_rhythm_counts.csv',audit)
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');a=p.parse_args()
    records=acquire(a.offline);rows=derive(records)
    print('Reference archives',len(rows))
if __name__=='__main__':main()
