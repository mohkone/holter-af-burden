"""Reproduce all secondary data tables from the bundled, frozen study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

def digest(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path,help='new working directory')
    args=parser.parse_args(); here=Path(__file__).resolve().parent
    dest=args.output.resolve(); dest.mkdir(parents=True,exist_ok=False)
    with ZipFile(here/'Study/Holter_intervalles_code_et_resultats.zip') as z: z.extractall(dest/'Study')
    study=dest/'Study/holter_intervals'
    runs=[('thresholds','patient_thresholds.py','--study-root','--out'),
          ('thresholds','independent_mc.py','--study-root','--out'),
          ('diagnostics','replay_diagnostics.py','--study','--output'),
          ('transfer','transfer_analysis.py','--study','--out')]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def execute(job):
        folder,script,study_flag,out_flag=job
        subprocess.run([sys.executable,str(here/'Revision'/folder/script),study_flag,str(study),out_flag,str(dest/folder)],check=True,env=env)
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(execute,runs))
    checked=[]
    for source in sorted((here/'Revision').rglob('*.csv*')):
        out=dest/source.relative_to(here/'Revision')
        if not out.is_file() or digest(out)!=digest(source):
            raise ValueError(f'Regenerated table differs: {source.name}')
        checked.append(source.name)
    lookup = Path('Editorial/borderline_mc_comparison.csv')
    subprocess.run([sys.executable, str(here/'Editorial/make_borderline_table.py'), '--thresholds',
                    str(dest/'thresholds'), '--out', str(dest/lookup)], check=True, env=env)
    if digest(dest/lookup) != digest(here/lookup):
        raise ValueError('Regenerated editorial lookup differs from the supplied file')
    subprocess.run([sys.executable,str(here/'Figure2/make_figure2.py'),'--draws',str(dest/'diagnostics/five_cases_draws.csv.gz'),'--out',str(dest/'Figure2')],check=True,env=env)
    subprocess.run([sys.executable,'-m','unittest','test_detector'],check=True,cwd=here/'COSEn',env=env)
    print(f'Regenerated and matched all {len(checked)} supplementary tables and the editorial lookup; regenerated Figure 2 and verified COSEn tests.')

if __name__=='__main__':main()
