"""Seal the reduced study and its actual language/GP evidence additively."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import v13_targeted_ablation as study
from v13_origin_ablation import queue


def main():
    report=ROOT/'experiments/reports'/study.VARIANT
    if queue.read(report/'audit.json')['status']!='PASS_COMPLETE':raise ValueError('Complete audit required')
    if queue.read(report/'facts.json')['status']!='COMPLETE':raise ValueError('Completed result report required')
    if queue.read(study.base()/'controls_complete.json')['status']!='COMPLETE':raise ValueError('Controller execution failed')
    p=study.check(True);files={}
    def add(path):
        path=Path(path)
        if path.is_file() and path.suffix not in ('.lock','.partial','.pyc') and '__pycache__' not in path.parts:
            files[queue.rel(ROOT,path)]=path
    def tree(path):
        for f in Path(path).rglob('*'):add(f)
    tree(study.base());tree(report)
    entries=set()
    for job in p['jobs']:
        folder=study.original.base()/'calls'/job['id'];tree(folder)
        for request in folder.glob('*/request.json'):
            entries.add(ROOT/queue.read(request)['response_map_entry'])
        tree(study.original.base()/'numerical_diagnostics'/job['id'])
    original=queue.read(study.original.base()/'protocol.json')
    for probe in original['probes']:
        tree(study.original.base()/'probes'/probe['id'])
        folder=study.original.base()/'calls'/probe['id'];tree(folder)
        for request in folder.glob('*/request.json'):entries.add(ROOT/queue.read(request)['response_map_entry'])
    for entry in entries:
        tree(entry.parent)
        for request in entry.parent.glob('a*/request.json'):
            meta=queue.read(request);add(ROOT/meta['input']);add(ROOT/meta['output'])
    for name in p['sources_sha256']:add(ROOT/name)
    source=study.original.check()
    for task,seed in study.CELLS:
        pool=study.original.POOL(ROOT,task);add(pool.source_path)
        with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig') as f:
            import csv
            row=next(r for r in csv.DictReader(f) if r['task']==study.original.SHORT[task]);add(ROOT/row['oracle_path'])
    for name in ('code/tools/v13_origin_nested_dispatch.py','code/tools/v13_targeted_dispatch.py',
        'code/analysis/audit_v13_origin_ablation.py','code/analysis/audit_v13_targeted_ablation.py',
        'code/analysis/report_v13_origin_ablation.py','code/analysis/report_v13_targeted_ablation.py',
        'code/analysis/verify_v13_targeted_report.py',
        'code/analysis/preserve_v13_targeted_ablation.py'):
        add(ROOT/name)
    for name in ('protocol.json','broad_phase_cancelled.json','operational_dispatch_registration.json','seeded_responses.json'):
        add(study.original.base()/name)
    for f in (ROOT/'code/chemist_system/third_party').rglob('LICENSE*'):add(f)
    target=ROOT/'experiments/milestones'/study.VARIANT;target.mkdir(parents=True,exist_ok=True)
    if (target/'receipt.json').exists():
        receipt=queue.read(target/'receipt.json')
        if queue.file_sha(target/'evidence.zip')!=receipt['archive_sha256']:raise ValueError('Sealed archive changed')
        print(json.dumps(receipt));return
    manifest={name:dict(bytes=path.stat().st_size,sha256=queue.file_sha(path)) for name,path in sorted(files.items())}
    queue.write_once(target/'manifest.json',manifest)
    archive=target/'evidence.zip';partial=target/'evidence.zip.partial'
    if archive.exists() or partial.exists():raise ValueError('Retain previous unsealed archive before retrying')
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.write(target/'manifest.json','PRESERVATION_MANIFEST.json')
    with zipfile.ZipFile(partial) as z:
        for name,value in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                while data:=f.read(1<<20):h.update(data)
            if h.hexdigest()!=value['sha256']:raise ValueError('Archive member changed: '+name)
    partial.replace(archive);study.check(True)
    receipt=dict(status='SEALED_AND_VERIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
        files=len(manifest),primary_cells=4,full_replay_trajectories=4,control_trajectories=12,control_queries=240,
        already_completed_prompt_probes=12,no_notes_cancelled=True,all_ZIP_members_verified=True,
        archive_sha256=queue.file_sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=queue.file_sha(target/'manifest.json'),protocol_sha256=p['protocol_sha256'],
        old_releases_and_cancelled_phase_unchanged=True,
        cancelled_phase='Preserved in place; state hashes are protected by targeted protocol; excluded from primary analysis')
    queue.write_once(target/'receipt.json',receipt);queue.write_once(report/'preservation_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
