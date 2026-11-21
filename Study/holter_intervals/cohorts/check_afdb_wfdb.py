"""Independent optional audit of supplementary AFDB binary decoding."""
from pathlib import Path
import csv
import numpy as np
import wfdb
from afdb_extra import parse,ROOT,RAW
nbeat=0;nmark=0;nf=0
for row in csv.DictReader((ROOT/'afdb_manifest.csv').open()):
    rid=row['record'];wave=str(RAW/rid);a=wfdb.rdann(wave,'atr');b=wfdb.rdann(wave,'qrs')
    ref=[(int(t),txt.lstrip('(').rstrip('\x00')) for t,txt in zip(a.sample,a.aux_note) if txt.startswith('(')]
    got= parse(RAW/(rid+'.atr'))[1]
    assert ref==got
    c,_=parse(RAW/(rid+'.qrs'))
    assert np.array_equal(c,b.sample[np.array(b.symbol)=='N'])
    nbeat+=len(c);nmark+=len(ref);nf+=1
print('Independent WFDB',wfdb.__version__,': all',nmark,'manual rhythm marker positions and',nbeat,'automatic N positions match exactly in',nf,'AFDB records.')
