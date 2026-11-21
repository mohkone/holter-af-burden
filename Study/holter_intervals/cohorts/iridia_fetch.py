"""Select the release directory and compressed members with HTTP byte ranges."""
import argparse, concurrent.futures, hashlib, io, json, os, re, struct, tempfile, time
from pathlib import Path
import urllib.request as url
from zipfile import ZipFile
import zlib
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw'/'IRIDIA'
LENGTH=10532846900
ZIP='iridia-af-records-v1.0.1.zip'
URLS=[f'https://zenodo.org/records/8405941/files/{ZIP}', f'https://zenodo.org/api/records/8405941/files/{ZIP}/content']
META='https://zenodo.org/records/8405941/files/iridia-af-metadata-v1.0.1.csv?download=1'
def fetch(start,end):
    failure=None
    for retry in range(5):
        uri=URLS[retry%len(URLS)]+f'?member={start}-{end}'
        try:
            req=url.Request(uri,headers={'Range':f'bytes={start}-{end}','User-Agent':'Holter interval study'})
            with url.urlopen(req,timeout=28) as resp:
                if resp.status!=206 or resp.headers['Content-Range']!=f'bytes {start}-{end}/{LENGTH}':raise ValueError(f'range {resp.status} {resp.headers.get("Content-Range")}')
                raw=resp.read(end-start+2)
            if len(raw)!=end-start+1:raise IOError('short range')
            return raw
        except Exception as ex:
            failure=ex
            time.sleep(.3*2**retry)
    raise OSError(f'Failed range {start}–{end}: {failure}')
def get_index():
    RAW.mkdir(exist_ok=True,parents=True)
    tail=RAW/'tail'
    if not tail.exists():tail.write_bytes(fetch(LENGTH-300000,LENGTH-1))
    if hashlib.sha256(tail.read_bytes()).hexdigest()!='ace8df542d380dc968874776f73e0c9e6702f773860ccdaeb140e993e5ddaccf':raise ValueError('Pinned ZIP byte-range directory digest differs')
    with tempfile.TemporaryFile() as sparse:
        sparse.seek(LENGTH-tail.stat().st_size);sparse.write(tail.read_bytes());sparse.flush()
        with ZipFile(sparse) as z:index=z.infolist()
    if len(index)!=1080:raise ValueError('release directory changed')
    return index
def main():
    RAW.mkdir(exist_ok=True,parents=True)
    metadata=RAW/'metadata.csv'
    if not metadata.exists():metadata.write_bytes(url.urlopen(META,timeout=20).read())
    if hashlib.md5(metadata.read_bytes()).hexdigest()!='f0102de28265ed6459956ac6cd397464':raise ValueError('release metadata digest')
    idx=get_index()
    wanted=[i for i in idx if i.filename.endswith(('_rr_00.h5','_ecg_labels.csv','_rr_labels.csv'))]
    by_offset=sorted(wanted,key=lambda v:v.header_offset)
    batches=[]
    for item in by_offset:
        start=item.header_offset;end=start+30+len(item.filename.encode())+512+item.compress_size
        if batches and start<=batches[-1][1]+4096:
            batches[-1][1]=max(batches[-1][1],end);batches[-1][2].append(item)
        else:batches.append([start,end,[item]])
    def populate(spec):
        start,end,infos=spec
        needed=[j for j in infos if not (RAW/Path(j.filename).name).is_file() or zlib.crc32((RAW/Path(j.filename).name).read_bytes())!=j.CRC]
        if not needed:return
        raw=fetch(start,min(LENGTH-1,end))
        for j in needed:
            shift=j.header_offset-start; local=raw[shift:shift+30];mark=struct.unpack_from('<4s',local)[0]
            if mark!=b'PK\x03\x04':raise ValueError('bad local header')
            ln,le=struct.unpack_from('<HH',local,26)
            name=raw[shift+30:shift+30+ln].decode()
            if name!=j.filename:raise ValueError('member name differs')
            b=raw[shift+30+ln+le:shift+30+ln+le+j.compress_size]
            out=zlib.decompress(b,-15) if j.compress_type==8 else b
            if len(out)!=j.file_size or zlib.crc32(out)!=j.CRC:raise ValueError('member CRC mismatch')
            tmp=RAW/(Path(name).name+'.tmp');tmp.write_bytes(out);tmp.replace(RAW/Path(name).name)
    if not by_offset:raise ValueError('empty archive selection')
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        fut=[pool.submit(populate,b) for b in batches]
        for i,f in enumerate(concurrent.futures.as_completed(fut),1):
            try:f.result()
            except Exception as ex:errors.append(str(ex))
            if i%40==0:print(i,len(batches),'errors',len(errors),flush=True)
    if errors:raise RuntimeError(errors[:12])
    entries={Path(j.filename).name:{'crc32':f'{j.CRC:08x}','bytes':j.file_size,'sha256':hashlib.sha256((RAW/Path(j.filename).name).read_bytes()).hexdigest()} for j in by_offset}
    (ROOT/'iridia_source_receipt.json').write_text(json.dumps({'record':8405941,'archive_md5_publisher':'81d58d21c9835a601cd9eb6c68bf9feb','archive_member_check':'ZIP central directory CRC-32, member lengths, and archived SHA-256. Full-archive MD5 not verified in range retrieval.', 'members':entries},indent=2))
    print('Verified members',len(entries))
if __name__=='__main__':main()
