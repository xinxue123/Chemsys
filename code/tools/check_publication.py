"""Check the exact Git index for accidental private runtime content."""
from io import BytesIO
from pathlib import Path
import json
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]
TOKEN=re.compile(rb'\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b')
ASSIGNED=re.compile(r'''(?i)(?:api_?key|api_?token|password|secret)\s*["']?\s*[:=]\s*["']([^"'\r\n]{8,})["']''')


def suspects(blob):
    kinds=[]
    if TOKEN.search(blob):kinds.append('credential_token_pattern')
    if re.search(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',blob):kinds.append('private_key')
    try:text=blob.decode('utf-8')
    except UnicodeDecodeError:return kinds
    for match in ASSIGNED.finditer(text):
        value=match.group(1)
        if not re.search(r'(?i)(your|dummy|example|placeholder|test|fake|api.key|environ|config|redact|not.a.key)',value) and not any(c in value for c in [' ','\\','/']):
            kinds.append('literal_credential_candidate')
    return sorted(set(kinds))


def main():
    raw=subprocess.check_output(['git','ls-files','-s','-z'],cwd=ROOT)
    entries=[]
    for item in raw.split(b'\0'):
        if not item:continue
        metadata,name=item.split(b'\t',1);mode,oid,stage=metadata.decode().split()
        entries.append((name.decode('utf-8'),mode,oid))
    assert entries,'No staged files'
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    findings=[];total=0;archive_members=0
    try:
        for name,mode,oid in entries:
            if name.startswith(('experiments/','runtime/','chemist_system/')) or name=='config.txt' or Path(name).suffix in {'.pt','.npz','.pyd','.log','.pem','.key'}:
                findings.append({'path':name,'kind':'private_runtime_or_binary_artifact'})
            if mode not in {'100644','100755'}:findings.append({'path':name,'kind':'unexpected_link_or_submodule'})
            proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
            head=proc.stdout.readline().decode().split();size=int(head[2]);blob=proc.stdout.read(size);assert proc.stdout.read(1)==b'\n'
            total+=len(blob)
            for kind in suspects(blob):findings.append({'path':name,'kind':kind})
            if name.endswith('.zip'):
                with zipfile.ZipFile(BytesIO(blob)) as z:
                    for item in z.infolist():
                        if item.is_dir():continue
                        archive_members+=1
                        for kind in suspects(z.read(item)):findings.append({'path':name+'!'+item.filename,'kind':kind})
    finally:
        proc.stdin.close();proc.wait()
    forbidden='code/chemist_system/third_party/reasoning_bo_official/scripts/experiment.py'
    assert forbidden not in {name for name,_,_ in entries}
    assert {'.gitignore','.gitattributes','README.md','code/run.py','run.ps1'}.issubset({name for name,_,_ in entries})
    report={'status':'PASS' if not findings else 'FAIL','staged_files':len(entries),'staged_bytes':total,
        'archive_members_scanned':archive_members,'findings':findings,'excluded_upstream_notification_script':True,
        'private_config_logs_environments_excluded':not any(r['kind']=='private_runtime_or_binary_artifact' for r in findings)}
    folder=ROOT/'runtime/github_publish';folder.mkdir(parents=True,exist_ok=True)
    (folder/'publication_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if findings:raise SystemExit(1)


if __name__=='__main__':main()
