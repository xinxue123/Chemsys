"""Seal new direct-weight evidence without changing any frozen comparator."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,sys,zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from program_kf_weight import study,VARIANT,CELLS
from v13_origin_ablation import queue


def main():
    p=study.check(True);report=ROOT/'experiments/reports'/VARIANT
    assert queue.read(report/'audit.json')['status']=='PASS_COMPLETE'
    assert queue.read(report/'visual_review.json')['status']=='PASS'
    assert queue.read(study.base()/'traces_complete.json')['status']=='COMPLETE'
    files={}
    def add(path):
        path=Path(path)
        if path.is_file() and path.suffix not in ('.lock','.partial','.pyc') and '__pycache__' not in path.parts:
            files[study.rel(path)]=path
    def tree(path):
        for item in Path(path).rglob('*'):add(item)
    tree(study.base());tree(report)
    for request in (study.base()/'responses').glob('*/*/a*/request.json'):
        r=queue.read(request);add(ROOT/r['input']);add(ROOT/r['output'])
    for name in p['sources_sha256']:add(ROOT/name)
    for name in ('code/analysis/audit_program_kf_weight.py','code/analysis/report_program_kf_weight.py',
                 'code/analysis/preserve_program_kf_weight.py','code/tools/program_kf_schema_repair.py'):
        add(ROOT/name)
    for name in ('code/tools/program_kf_operations.py','code/analysis/plot_program_kf_prefix.py',
                 'code/analysis/audit_v13_origin_ablation.py','code/tools/program_kf_alias_recovery.py',
                 'code/analysis/verify_program_kf_report.py'):
        add(ROOT/name)
    for path in (ROOT/'experiments/logs').glob('program_kf_weight_*.log'):add(path)
    for ref in p['references'].values():
        path=ROOT/ref['state_path'];add(path)
        tree(path.parent/'rounds');add(path.parent/'initial_commit.json')
    for task,seed in CELLS:
        pool=study.frozen.pool_for(ROOT,task);add(pool.source_path)
        with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig') as f:
            row=next(r for r in csv.DictReader(f) if r['task']==study.SHORT[task]);add(ROOT/row['oracle_path'])
    for path in (ROOT/'code/chemist_system/third_party').rglob('LICENSE*'):add(path)
    target=ROOT/'experiments/milestones'/VARIANT;target.mkdir(parents=True,exist_ok=True)
    if (target/'receipt.json').exists():
        receipt=queue.read(target/'receipt.json')
        assert queue.file_sha(target/'evidence.zip')==receipt['archive_sha256'];print(json.dumps(receipt));return
    manifest={name:dict(bytes=path.stat().st_size,sha256=queue.file_sha(path)) for name,path in sorted(files.items())}
    queue.write_once(target/'manifest.json',manifest)
    archive=target/'evidence.zip';partial=target/'evidence.zip.partial'
    if archive.exists() or partial.exists():raise ValueError('Preserve unsealed attempt before retry')
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.write(target/'manifest.json','PRESERVATION_MANIFEST.json')
    with zipfile.ZipFile(partial) as z:
        for name,value in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                while block:=f.read(1<<20):h.update(block)
            if h.hexdigest()!=value['sha256']:raise ValueError('Archive member mismatch: '+name)
    partial.replace(archive);study.check(True)
    receipt=dict(status='SEALED_AND_VERIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
        files=len(manifest),new_complete_trajectories=4,new_experiment_queries=80,
        fixed_historical_prefixes=40,prefix_new_experiment_queries=0,
        all_ZIP_members_verified=True,archive_sha256=queue.file_sha(archive),archive_bytes=archive.stat().st_size,
        protocol_sha256=p['protocol_sha256'],manifest_sha256=queue.file_sha(target/'manifest.json'),
        old_V13_and_other_frozen_controls_unchanged=True)
    queue.write_once(target/'receipt.json',receipt);queue.write_once(report/'preservation_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
