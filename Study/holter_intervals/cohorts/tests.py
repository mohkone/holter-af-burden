"""Independent scalar audit for IRIDIA integration and automatic RR reconstruction."""
import csv
from pathlib import Path
import numpy as np
import h5py
from new_data import fractions
ROOT=Path(__file__).resolve().parent
idx,x=fractions([30,18000],[8000,32000],36000,200)
assert np.allclose(x,[7970/12000,6000/12000,8000/12000])
n=0
for r in csv.DictReader((ROOT/'iridia_manifest.csv').open()):
    rec=r['record'];raw=ROOT/'raw'/'IRIDIA';f=np.load(ROOT/'derived'/'IRIDIA'/(rec+'.npz'))
    with h5py.File(raw/(rec+'_rr_00.h5')) as h:rr=h['rr'][:]
    assert np.array_equal(np.diff(f['peaks']).astype('int64')*5,rr[:len(f['peaks'])-1])
    marks=list(csv.DictReader((raw/(rec+'_ecg_labels.csv')).open()))
    rng=np.random.default_rng(int(rec.split('_')[1])+471)
    for m in rng.choice(f['idx'][f['valid']],size=8,replace=False):
        a,b=int(m)*12000,(int(m)+1)*12000;s=0
        for j in marks:
            lo=int(j['start_file_index'])*17280000+int(j['start_qrs_index'])
            hi=int(j['end_file_index'])*17280000+int(j['end_qrs_index'])
            s+=max(0,min(b,hi)-max(a,lo))
        assert np.isclose(s/12000,f['y'][m],rtol=0,atol=1e-12);n+=1
print(f'All day-0 peak increments agree with official automatic RR files. Scalar intersection agrees on {n} prespecified random eligible minutes across 167 records.')
