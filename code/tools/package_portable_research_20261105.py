"""Preserve all available chemistry experiment evidence in relocatable archives.

No optimizer, GP, oracle evaluator or model client is imported or executed.
Scientific records are never rewritten to improve a result.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'Chemsys-portable'
ARCHIVES = DEST/'archives'
CAT = DEST/'data_catalogue'
SKIP_DIR = {'.git','.venv','node_modules','__pycache__','.pytest_cache'}
SKIP_EXT = {'.pyc','.pyo','.lock','.tmp'}
TEXT_EXT = {'.json','.jsonl','.csv','.tsv','.txt','.md','.py','.ps1','.log','.tex','.bib','.toml','.yaml','.yml','.ini','.xml','.diff'}


def native(path):
    value = os.path.abspath(str(path))
    return '\\\\?\\'+value if os.name == 'nt' and not value.startswith('\\\\?\\') else value


def sha(path):
    h = hashlib.sha256()
    with open(native(path),'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def walk(base):
    stack = [base]
    while stack:
        folder = stack.pop()
        with os.scandir(native(folder)) as iterator:
            entries = sorted(iterator,key=lambda e:e.name)
        for entry in entries:
            path = folder/entry.name
            st = entry.stat(follow_symlinks=False)
            if getattr(st,'st_file_attributes',0)&1024 or entry.is_symlink():
                yield path,st,'reparse_alias'
            elif entry.is_dir(follow_symlinks=False):
                if entry.name not in SKIP_DIR:stack.append(path)
            elif path.suffix.lower() in SKIP_EXT:
                yield path,st,'transient_runtime_file'
            else:
                yield path,st,''


def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(native(path),'w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def known_secrets():
    found = set()
    for name in ('openai-key.txt','openai-apikey.txt','config.txt'):
        p = ROOT/name
        if p.exists():
            text = p.read_text(encoding='utf-8-sig')
            found.update(x.encode() for x in re.findall(r'sk-[A-Za-z0-9_-]{16,}',text))
            try:
                value = json.loads(text)
                if isinstance(value,dict):
                    for k,v in value.items():
                        if re.search('key|token|password',k,re.I) and isinstance(v,str) and len(v)>16:found.add(v.encode())
            except ValueError:pass
    return tuple(sorted(found,key=len,reverse=True))


def clean_text(data,secrets):
    count = 0
    for token in secrets:
        hits = data.count(token)
        if hits:data=data.replace(token,b'[REDACTED_CREDENTIAL]');count+=hits
    # Also remove obsolete, realistically shaped credentials, not API variable
    # names or the obvious non-secret placeholders used by the test suite.
    def redact(match):
        nonlocal count
        value = match.group()
        if any(w in value.lower() for w in (b'example',b'placeholder',b'redacted',b'dummy',b'test',b'your')):return value
        count += 1
        return b'[REDACTED_CREDENTIAL]'
    data = re.sub(rb'\bsk-[A-Za-z0-9_-]{28,}\b',redact,data)
    return data,count


def jobs():
    rows = []
    for p in sorted((ROOT/'experiments/runs').iterdir()):
        if p.is_dir():rows.append(dict(id='run__'+p.name,kind='run',roots=[p.relative_to(ROOT).as_posix()]))
    for name,kind,roots in (
        ('reports','report',['experiments/reports']),
        ('datasets','dataset',['experiments/datasets']),
        ('source_code','source',['code','requirements-bo.txt','run.ps1','config.example.txt']),
        ('original_protocols','protocol',['docs']),
        ('agent_packets','LLM_evidence',['experiments/agent_io']),
        ('historical_manuscripts','paper',['paper']),
        ('milestones','milestone',['experiments/milestones']),
        ('auxiliary_evidence','audit_and_console',['experiments/metadata','experiments/operations','experiments/control','experiments/pr_validation','experiments/logs','experiments/verification'])):
        rows.append(dict(id=name,kind=kind,roots=roots))
    loose = [p.relative_to(ROOT).as_posix() for p in (ROOT/'experiments/runs').iterdir() if p.is_file()]
    rows.append(dict(id='run_console_files',kind='console',roots=loose))
    return rows


def pack(job,secrets):
    archive = ARCHIVES/(job['id']+'.zip')
    index = CAT/'files'/(job['id']+'.csv')
    excluded_path = CAT/'excluded'/(job['id']+'.csv')
    receipt_path = CAT/'receipts'/(job['id']+'.json')
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert sha(archive)==saved['archive_sha256'] and sha(index)==saved['file_index_sha256']
        return saved
    records=[];excluded=[];states=[]
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(native(temp),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for relative in job['roots']:
            source_root=ROOT/relative
            paths = walk(source_root) if source_root.is_dir() else [(source_root,source_root.stat(),'')]
            for p,stat,reason in paths:
                rel=p.relative_to(ROOT).as_posix()
                if reason:
                    excluded.append(dict(source_path=rel,reason=reason));continue
                if p.name in ('openai-key.txt','openai-apikey.txt','.env','auth.json','credentials.json'):
                    excluded.append(dict(source_path=rel,reason='credential_file'));continue
                before=stat.st_mtime_ns
                original_hash=hashlib.sha256();export_hash=hashlib.sha256();redactions=0
                compression=zipfile.ZIP_STORED if p.suffix.lower() in ('.zip','.png','.jpg','.jpeg','.pdf','.npz','.pt','.pth') else zipfile.ZIP_DEFLATED
                info=zipfile.ZipInfo(rel);info.compress_type=compression;info._compresslevel=1
                if p.suffix.lower() in TEXT_EXT:
                    with open(native(p),'rb') as f:data=f.read()
                    original_hash.update(data);data,redactions=clean_text(data,secrets);export_hash.update(data)
                    z.writestr(info,data);size=len(data)
                    if p.name=='state.json':
                        try:
                            s=json.loads(data.decode('utf-8-sig'))
                            history=s.get('history',[])
                            states.append(dict(source_path=rel,study=job['id'].removeprefix('run__'),
                                status=s.get('status','not_recorded'),observations=len(history) if isinstance(history,list) else '',
                                initial=s.get('initial','not_recorded') if not isinstance(s.get('initial'),(list,dict)) else 'structured',
                                query_budget=s.get('queries',s.get('budget','not_recorded')),
                                seed=s.get('seed',s.get('trial_seed','not_recorded')),task=s.get('task','not_recorded'),
                                method=s.get('arm',s.get('method','not_recorded'))))
                        except (ValueError,UnicodeError,TypeError):pass
                else:
                    size=0
                    with open(native(p),'rb') as f,z.open(info,'w',force_zip64=True) as target:
                        for block in iter(lambda:f.read(2**20),b''):
                            original_hash.update(block);export_hash.update(block);target.write(block);size+=len(block)
                after=os.stat(native(p))
                assert before==after.st_mtime_ns and stat.st_size==after.st_size,'Input changed during packaging: '+rel
                records.append(dict(source_path=rel,archive_member=rel,original_bytes=stat.st_size,exported_bytes=size,
                    original_sha256=original_hash.hexdigest(),exported_sha256=export_hash.hexdigest(),
                    credential_redactions=redactions,status='SANITIZED_SECRET_COPY' if redactions else 'BYTE_IDENTICAL'))
    os.replace(native(temp),native(archive))
    write_csv(index,records,['source_path','archive_member','original_bytes','exported_bytes','original_sha256','exported_sha256','credential_redactions','status'])
    write_csv(excluded_path,excluded,['source_path','reason'])
    write_csv(CAT/'states'/(job['id']+'.csv'),states,['source_path','study','status','observations','initial','query_budget','seed','task','method'])
    result=dict(id=job['id'],kind=job['kind'],source_roots=job['roots'],archive=archive.relative_to(DEST).as_posix(),
        file_index=index.relative_to(DEST).as_posix(),archive_sha256=sha(archive),file_index_sha256=sha(index),
        files=len(records),original_bytes=sum(r['original_bytes'] for r in records),exported_bytes=sum(r['exported_bytes'] for r in records),
        archive_bytes=archive.stat().st_size,credential_redactions=sum(r['credential_redactions'] for r in records),
        exclusions=len(excluded),raw_state_files_indexed=len(states))
    receipt_path.parent.mkdir(parents=True,exist_ok=True);receipt_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    ARCHIVES.mkdir(parents=True,exist_ok=True);CAT.mkdir(parents=True,exist_ok=True)
    legacy=DEST/'docs/legacy_portable_20261104';legacy.mkdir(parents=True,exist_ok=True)
    for name in ('MANIFEST.json','README.md','chemsys.py'):
        if not (legacy/name).exists():shutil.copyfile(DEST/name,legacy/name)
    selected=jobs();secrets=known_secrets();results=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(pack,j,secrets) for j in selected]):
            r=future.result();results.append(r)
            print(json.dumps({k:r[k] for k in ('id','files','archive_bytes','credential_redactions')}),flush=True)
    results.sort(key=lambda r:r['id'])
    doc=dict(status='ARCHIVED_AWAITING_FULL_VERIFICATION',created_UTC=datetime.now(timezone.utc).isoformat(),
        scope='All available chemistry optimization runs, reports, input/oracle pools, LLM evidence, source code, original protocols, manuscripts, milestones, and auxiliary audits. No outcome-based cohort exclusion.',
        exclusions='Credentials, transient locks/tmp/bytecode, environment/git/dependency directories, redundant reparse aliases; unrelated Ce-Al-Ge/Materials Project/XRD caches are outside this chemical optimization scope.',
        archives=results,total_files=sum(r['files'] for r in results),original_bytes=sum(r['original_bytes'] for r in results),
        archive_bytes=sum(r['archive_bytes'] for r in results),credential_redactions=sum(r['credential_redactions'] for r in results),
        workers=args.workers,elapsed_seconds=time.monotonic()-start,new_experiments=0,new_GP_fits=0,new_LLM_calls=0)
    (CAT/'archive_catalogue.json').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')
    write_csv(CAT/'archive_catalogue.csv',[{k:r[k] for k in ('id','kind','files','original_bytes','archive_bytes','archive','file_index','archive_sha256','raw_state_files_indexed','credential_redactions')} for r in results])
    print(json.dumps({k:v for k,v in doc.items() if k!='archives'}),flush=True)


if __name__=='__main__':main()
