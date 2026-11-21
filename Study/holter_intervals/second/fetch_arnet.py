"""Download the pinned public model, checking each upstream Git object and SHA-256."""
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import argparse

def fetch(target):
    spec=json.loads(Path(__file__).with_name('sources.json').read_text())
    prefix=f'https://raw.githubusercontent.com/aim-lab/AFtoolkit/{spec["rev"]}/{spec["subdir"]}/'
    for name,metadata in spec['files'].items():
        dest=target/name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==metadata['sha256']:
            continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        with urlopen(prefix+name,timeout=90) as resp: data=resp.read()
        hashed=hashlib.sha256(data).hexdigest()
        oid=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if hashed!=metadata['sha256'] or oid!=metadata['git_blob'] or len(data)!=metadata['bytes']:
            raise RuntimeError(f'Unexpected content in {name}')
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as tmp:
            tmp.write(data); p=Path(tmp.name)
        p.replace(dest)
if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,default=Path(__file__).with_name('exported_model'))
    fetch(a.parse_args().output)
