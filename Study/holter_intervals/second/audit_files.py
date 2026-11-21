"""Verify serialized detector outputs and the predeclared score discretization."""
from pathlib import Path
import csv
import hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def audit():
    messages=[]
    for cohort in ['LTA','SHDB','IRIDIA','AFDB']:
        path=HERE/f'manifest_{cohort}.csv'
        if not path.exists():raise ValueError(f'Unfinished detector cohort: {cohort}')
        with path.open() as f:rows=list(csv.DictReader(f))
        files=sorted((HERE/'derived'/cohort).glob('*.npz'))
        assert {p.stem for p in files}=={r['record'] for r in rows}
        total=0;unknown=0
        for r in rows:
            p=HERE/'derived'/cohort/(r['record']+'.npz')
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
            with np.load(p,allow_pickle=False) as a:
                score=a['score']; idx=a['idx']; valid=a['invalid_fraction']
            assert np.array_equal(idx,np.arange(len(score))) and len(score)==int(r['n_minutes'])
            assert np.all(np.isfinite(score)) and np.all((score>=0)&(score<=1))
            assert np.allclose(score*12,np.round(score*12))
            total+=len(score);unknown+=valid.sum()
        messages.append(f'{cohort}: {len(rows)} records, {total} full native-clock minutes, {100*unknown/total:.4f}% of query points unknown; all file digests, clocks and legal score values passed.')
    return messages
if __name__=='__main__':print('\n'.join(audit()))
