"""Atomic combined record inventory; cohort identities are distinct."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def merge():
    out=[]
    for filename in ('iridia_manifest.csv','afdb_manifest.csv'):
        if (ROOT/filename).exists():
            with (ROOT/filename).open() as f:out.extend(csv.DictReader(f))
    if len({(r['cohort'],r['record']) for r in out})!=len(out):raise ValueError('duplicate inventory key')
    tmp=ROOT/'manifest.csv.tmp'
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    tmp.replace(ROOT/'manifest.csv')
if __name__=='__main__':merge()
