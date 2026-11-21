"""Recreate individual thresholds and the exploratory paired coverage contrast."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path,help='Original study ZIP')
    args=ap.parse_args()
    with ZipFile(args.archive) as source:
        path='holter_intervals/results/patients.csv'
        b=source.read(path)
    assert hashlib.sha256(b).hexdigest()=='1e6cda51bd80e92a7e97556541df745853a8991cd5056e9d51613704a205a135'
    data=pd.read_csv(BytesIO(b),dtype={'patient':str})
    data=data.loc[(data.cohort=='IRIDIA') & (data.detector=='ArNet2')]
    pairs=data.pivot(index='patient',columns='method',values='coverage').sort_index()
    old=pairs.fixed_radius
    new=pairs.scaled_radius
    assert len(pairs)==152
    counts=(int(((old>=.8)&(new>=.8)).sum()),int(((old<.8)&(new<.8)).sum()),
            int(((old>=.8)&(new<.8)).sum()),int(((old<.8)&(new>=.8)).sum()))
    assert counts==(138,10,4,0)
    assert ((old==1)&(new==1)).sum()==82
    assert (new<.5).sum()==5 and ((new<.5)&(old>=.8)).sum()==3
    seed=int.from_bytes(hashlib.sha256(b'holter-int-bootstrap|IRIDIA|overall').digest()[:8],'little')
    index=np.random.default_rng(seed).integers(len(pairs),size=(10000,len(pairs)))
    diff=100*(new-old).to_numpy()
    lo,hi=np.quantile(diff[index].mean(axis=1),[.025,.975])
    np.testing.assert_allclose([diff.mean(),lo,hi],[-.0544,-1.7273,1.4169],atol=.00015)
    print('Counts at/above both, below both, below only adaptive, below only fixed:',counts)
    print('Coverage difference in percentage points, adaptive minus fixed:',f'{diff.mean():.4f} [{lo:.4f}, {hi:.4f}]')
    print((pairs.loc[(old>=.8)&(new<.8),['fixed_radius','scaled_radius']]*100).to_string())
if __name__=='__main__': main()
