"""Stage the version 3 supplement and check the repository's archived sources."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile

SUPPLEMENT_SHA256 = 'aeaedec957170fd049c0fda811029d7b95162f4dd6865d320ffedd8cc3a320d8'
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('supplement', type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory for the replay inputs')
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists() or target == ROOT or ROOT in target.parents:
        parser.error('choose a new output directory outside the repository')
    with args.supplement.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SUPPLEMENT_SHA256:
            parser.error('expected the original Holter_AF_burden_supplement_v3.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        stage = Path(scratch) / 'stage'
        with ZipFile(args.supplement) as supplement:
            supplement.extractall(stage)
        with ZipFile(stage / 'Study/Holter_intervalles_code_et_resultats.zip') as core:
            core.extractall(stage / 'Study')
        checked = 0
        for row in (ROOT / 'FROZEN_SOURCES.sha256').read_text().splitlines():
            expected, relative = row.split(maxsplit=1)
            local = (ROOT / relative).read_bytes()
            if hashlib.sha256(local).hexdigest() != expected:
                raise ValueError(f'Repository source differs from the frozen file: {relative}')
            if relative.startswith('Study/holter_pilot/'):
                continue  # These two upstream preparation files come from the earlier pilot.
            if (stage / relative).read_bytes() != local:
                raise ValueError(f'Frozen source differs between releases: {relative}')
            checked += 1
        if target.exists():
            raise FileExistsError(target)
        os.rename(stage, target)
    print(f'{checked} version 3 source files verified. Replay directory: {target}')

if __name__ == '__main__':
    main()
