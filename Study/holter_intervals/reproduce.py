"""Rebuild all deposited scientific model and interval results from minute data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DETECTORS = ('COSEn', 'ArNet2')
CORE = ('study.py','finite.py','pilot_estimators.py','test_finite.py','check_study.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze(name):
    for line in (ROOT/name).read_text().splitlines():
        expected, label = line.split(maxsplit=1)
        path = Path(label)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Untrusted frozen path {path}')
        if path.parts[0] == 'holter_intervals':
            path = Path(*path.parts[1:])
        if sha(ROOT/path) != expected:
            raise RuntimeError(f'Frozen source differs: {path}')


def stage_inputs(run, pairs):
    digests = []
    for det,co in sorted(pairs):
        matrix = ROOT/'inputs'/f'{det}_{co}.csv'
        dest = run/matrix.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(matrix,dest)
        with matrix.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows or len({row['record'] for row in rows}) != len(rows):
            raise ValueError(f'Unexpected recording keys in {matrix.name}')
        for row in rows:
            rel = Path(row['path'])
            if ((row['detector'],row['cohort']) != (det,co)
                    or rel.parts[:3] != ('inputs',det,co) or len(rel.parts)!=4
                    or rel.name != row['record']+'.npz'):
                raise ValueError(f'Invalid manifest row in {matrix.name}')
            source = ROOT/rel
            dig = sha(source)
            if dig != row['sha256']:
                raise RuntimeError(f'Input archive differs: {rel}')
            target = run/rel
            target.parent.mkdir(exist_ok=True,parents=True)
            shutil.copy2(source,target)
            if sha(target)!=dig:
                raise RuntimeError(f'Copied archive differs: {rel}')
            digests.append(f'{dig}  {rel.as_posix()}')
    return len(digests), hashlib.sha256(('\n'.join(sorted(digests))+'\n').encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='new empty working directory')
    p.add_argument('--workers',type=int,default=2,help='independent evaluation processes (default 2)')
    args = p.parse_args()
    if args.workers<1: p.error('workers must be positive')
    run = args.output.resolve()
    if run.exists():
        p.error(f'use a new directory for replay: {run}')
    if ROOT==run or ROOT in run.parents:
        p.error('choose a directory separate from the deposited package')
    started=time.monotonic()
    verify_freeze('protocol.sha256')
    verify_freeze('methods_freeze.sha256')
    pairs=[]
    for path in sorted((ROOT/'results').glob('records_*.csv')):
        prefix,co,det=path.stem.split('_')
        if det not in DETECTORS or co == 'LTA':raise ValueError(path)
        pairs.append((det,co))
    if {(d,c) for d in DETECTORS for c in ('SHDB','IRIDIA')}-set(pairs):
        raise RuntimeError('The four planned SHDB/IRIDIA result tables must already be deposited')
    reference = [f'model_{det}.json' for det in DETECTORS]
    reference += [f'records_{co}_{det}.csv' for det,co in pairs]
    reference += ['summary.csv','patients.csv','paired_widths.csv']
    if any(not (ROOT/'results'/n).is_file() for n in reference):
        raise RuntimeError('Incomplete reference result set')
    run.mkdir(parents=True)
    (run/'results').mkdir()
    for name in CORE:
        shutil.copy2(ROOT/name,run/name)
    total, input_hash=stage_inputs(run,set(pairs)|{(d,'LTA') for d in DETECTORS})
    env={**os.environ,**{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}}
    def invoke(cmd):
        a=subprocess.run([sys.executable,*cmd],cwd=run,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if a.returncode:
            raise RuntimeError(' '.join(cmd)+' failed:\n'+a.stdout)
        return a.stdout

    logs=[]
    logs.append(invoke(['test_finite.py']))
    logs.append(invoke(['check_study.py']))
    for det in DETECTORS:
        invoke(['study.py','train','--det',det])
        file=f'model_{det}.json'
        if sha(run/'results'/file)!=sha(ROOT/'results'/file):
            raise RuntimeError(f'Development-only model changed on replay: {file}')
    commands=[['study.py','evaluate','--det',d,'--cohort',c] for d,c in pairs]
    with ThreadPoolExecutor(max_workers=min(args.workers,len(commands))) as pool:
        logs.extend(pool.map(invoke,commands))
    invoke(['study.py','aggregate'])
    logs.append(invoke(['check_study.py','--results']))
    (run/'results'/'replay_tests.log').write_text('\n'.join(logs))
    hashes=[]
    for file in sorted(reference):
        observed=sha(run/'results'/file)
        if observed!=sha(ROOT/'results'/file):
            raise RuntimeError(f'Byte comparison failed: results/{file}')
        hashes.append(f'{observed}  results/{file}')
    result='\n'.join([
        'Isolated offline replay from derived minute archives.',
        f'Python {platform.python_version()} on {platform.system()} {platform.machine()}.',
        f'{total} minute files copied to fresh workspace and individually SHA-256 verified.',
        f'Digest of sorted input-digest index: {input_hash}',
        'Protocol, calculation sources, independent finite-population tests and study audit verified.',
        'Both calibration models reproduced and hash-checked before evaluation.',
        f'{len(pairs)} external cohort/detector evaluations, including all 1,000 campaigns per recording, reproduced.',
        'Patient aggregation, 10,000 bootstrap resamples, interval widths, and individual coverage diagnostics rebuilt.',
        *hashes,
        f'PASS: {len(reference)} scientific model and result files identical byte for byte.',
        f'Elapsed time on this host: {time.monotonic()-started:.2f} s ({args.workers} worker limit).',
    ])+'\n'
    (run/'results'/'replay.txt').write_text(result)
    print(result,end='')

if __name__=='__main__':
    main()
