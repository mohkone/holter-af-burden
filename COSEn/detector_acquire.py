"""Acquire automatic R-peak series and headers only, checking SHA-256 before use."""
from pathlib import Path
import hashlib, urllib.request, concurrent.futures, argparse, csv, os, tempfile, time
ROOT = Path(__file__).resolve().parent
COHORTS = {'LTAFDB':('ltafdb/1.0.0','RECORDS'), 'SHDB':('shdb-af/1.0.1','RECORDS.txt')}
ORIGINS = ['https://physionet-open.s3.us-east-1.amazonaws.com/', 'https://physionet.org/files/', 'https://s3.amazonaws.com/physionet-open/']
def fetch(cohort, name, expected=None, offline=False):
    path = ROOT/'raw_detector'/cohort/name
    path.parent.mkdir(exist_ok=True,parents=True)
    if path.exists() and (expected is None or hashlib.sha256(path.read_bytes()).hexdigest()==expected):
        return path
    if offline: raise FileNotFoundError(path)
    for k in range(6):
        uri = ORIGINS[k%len(ORIGINS)] + COHORTS[cohort][0] + '/' + name
        try:
            with urllib.request.urlopen(uri, timeout=18) as response:
                payload = response.read()
            if not payload or (expected and hashlib.sha256(payload).hexdigest()!=expected):
                raise ValueError('incorrect download digest')
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                f.write(payload)
            os.replace(f.name,path)
            return path
        except Exception as e:
            last=e
    raise RuntimeError(f'Could not acquire {cohort}/{name}: {last}')
def records(cohort, offline=False):
    manifest = fetch(cohort,'SHA256SUMS.txt',offline=offline)
    checks={z[1].lstrip('*'):z[0] for line in manifest.read_text().splitlines() if len(z:=line.split())==2}
    recname=COHORTS[cohort][1]
    listing=fetch(cohort,recname,checks.get(recname),offline)
    recs=listing.read_text().split()
    return recs,checks
def run(offline):
    jobs=[]
    for cohort in COHORTS:
        recs,sums=records(cohort,offline)
        for rec in recs:
            for ext in ['hea','qrs']:
                name=rec+'.'+ext
                jobs.append((cohort,name,sums[name],offline))
    with concurrent.futures.ThreadPoolExecutor(14) as pool:
        for _ in pool.map(lambda args:fetch(*args),jobs): pass
    out=ROOT/'tables'/'detector_sources.csv';out.parent.mkdir(exist_ok=True)
    with out.open('w') as f:
        wr=csv.writer(f);wr.writerow(['cohort','filename','sha256','url'])
        for cohort,name,expected,_ in jobs:
            wr.writerow([cohort,name,expected,ORIGINS[0]+COHORTS[cohort][0]+'/'+name])
    print('Confirmed headers and QRS files:',len(jobs))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--offline',action='store_true')
    run(a.parse_args().offline)
