"""Supplementary 10-hour MIT-BIH AFDB, separate from the main 24-h confirmation."""
from pathlib import Path
import csv,concurrent.futures,hashlib,urllib.request as u,time
import numpy as np
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'raw'/'AFDB';BASE='https://physionet-open.s3.amazonaws.com/afdb/1.0.0'
def get(name,checks):
    path=RAW/name
    for k in range(4):
        if path.is_file() and (name not in checks or hashlib.sha256(path.read_bytes()).hexdigest()==checks[name]):return path
        with u.urlopen(BASE+'/'+name+('' if not k else '?whole=1'),timeout=25) as r:b=r.read()
        if name in checks and hashlib.sha256(b).hexdigest()!=checks[name]:continue
        path.write_bytes(b);return path
    raise ValueError('Checksum '+name)
def parse(path):
    b=path.read_bytes();offset=0;t=0; event=[];last_t=0;prevcode=0;rh=[];peaks=[]
    while offset<len(b):
        w=int.from_bytes(b[offset:offset+2],'little');offset+=2;code=w>>10;d=w&1023
        if w==0:break
        if code==59:
            v=int.from_bytes(b[offset+2:offset+4]+b[offset:offset+2],'little',signed=True);t+=v;offset+=4
        elif code==63:
            val=b[offset:offset+d].decode('latin1').rstrip('\x00');offset+=d+(d%2)
            if val.startswith('('):rh.append((t,val[1:].strip()))
        elif code in (60,61,62):pass
        else:
            t+=d
            if code==1:peaks.append(t)
    return np.array(peaks,dtype='i8'),rh
def main():
    raw=RAW;raw.mkdir(exist_ok=True,parents=True)
    dig=get('SHA256SUMS.txt',{'SHA256SUMS.txt':'8f3ab7b38bf0591235680a855d6a7f00fe8fdaf2e457fa4c00edad2c735e0d9a'})
    checks={x.split()[1]:x.split()[0] for x in dig.read_text().splitlines()}
    rec=get('RECORDS',checks).read_text().split()
    records=[r for r in rec if r+'.hea' in checks and r not in ['00735','03665']]
    names=[k for r in records for k in [r+'.hea',r+'.atr',r+'.qrs']]+[r+'.dat' for r in ['04043','08405','08434','06453']]+['notes.txt']
    with concurrent.futures.ThreadPoolExecutor(8) as pool:list(pool.map(lambda k:get(k,checks),names))
    dest=ROOT/'derived'/'AFDB';dest.mkdir(exist_ok=True,parents=True);rows=[];gaps=[]
    for rid in records:
        head=(raw/(rid+'.hea')).read_text().splitlines()[0].split();fs=int(head[2]);n=int(head[3]);unit=60*fs
        assert fs==250
        idx=np.arange(n//unit,dtype='i4')
        peak,_=parse(raw/(rid+'.qrs'));peak=peak[(peak>=0)&(peak<n)]
        _,marks=parse(raw/(rid+'.atr'))
        if not len(peak) or not marks:raise ValueError(rid)
        valid=idx*unit>=marks[0][0];y=np.zeros(len(idx));grid=np.arange(len(idx)+1)*unit;cum=np.zeros_like(grid)
        for (start,label),stop in zip(marks,[t for t,_ in marks[1:]]+[n]):
            if label not in {'N','AFIB','J','AFL'}:raise ValueError(label)
            if label=='AFIB':cum+=(grid-start).clip(0,max(0,stop-start))
        y=np.diff(cum)/unit
        if rid in ['04043','08405','08434','06453']:
            triplet=np.frombuffer((raw/(rid+'.dat')).read_bytes(),dtype=np.uint8).reshape(-1,3)
            # Confirm repeated full 2-channel sample pairs, including inserted missing blocks.
            changes=np.flatnonzero(np.any(triplet[1:]!=triplet[:-1],axis=1))+1
            bounds=np.r_[0,changes,len(triplet)]
            for j in np.flatnonzero(np.diff(bounds)>=2500):
                a,b=int(bounds[j]),int(bounds[j+1])
                valid[(idx*unit<b)&((idx+1)*unit>a)]=False;gaps.append((rid,a,b,'flat'))
        # Guard signal-absent tail when headers overstate valid acquisition.
        if n-peak[-1]>60*fs:
            valid[(idx+1)*unit>peak[-1]]=False;gaps.append((rid,peak[-1],n,'tail_no_auto_beats'))
        y[~valid]=np.nan;path=dest/(rid+'.npz')
        np.savez_compressed(path,record=rid,patient=rid,fs=fs,n_samples=n,idx=idx,valid=valid,y=y,peaks=peak.astype(np.int32))
        rows.append(dict(cohort='AFDB',record=rid,patient=rid,source='10.13026/C2MW2D',minutes_total=len(idx),minutes_eligible=int(valid.sum()),fs=fs,n_samples=n,source_peaks=len(peak),npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for filename,header,data in [('afdb_manifest.csv',list(rows[0]),[list(v.values()) for v in rows]),('afdb_missing_signal.csv',['record','first_sample','end_exclusive','reason'],gaps),('afdb_source_receipt.csv',['name','sha256'],[(e,checks[e]) for e in sorted(set(names+['RECORDS']))])]:
        with (ROOT/filename).open('w',newline='') as f:w=csv.writer(f);w.writerow(header);w.writerows(data)
    from merge_manifests import merge
    merge()
    print('AFDB',len(rows),'waveform-available records represented,',sum(r['minutes_eligible'] for r in rows),'minutes')
if __name__=='__main__':main()
