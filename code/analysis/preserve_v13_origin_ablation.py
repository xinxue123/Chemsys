"""Additive, hash-verified evidence archive after the entire ablation passes."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_origin_ablation import study,queue


def main():
    report=ROOT/'experiments/reports'/study.VARIANT
    if queue.read(report/'audit.json')['status']!='PASS_COMPLETE':raise ValueError('Complete native audit required')
    if queue.read(report/'facts.json')['status']!='COMPLETE':raise ValueError('Completed analysis required')
    p=study.check(True);status=queue.status(ROOT)
    if status['complete']!=160 or status['pending_requests']!=0:raise ValueError('Scientific workers still incomplete')
    if queue.read(study.base()/'controls_batch.json')['status']!='COMPLETE':raise ValueError('Batch has execution errors')
    target=ROOT/'experiments/milestones'/study.VARIANT
    target.mkdir(parents=True,exist_ok=True)
    if (target/'receipt.json').exists():
        old=queue.read(target/'receipt.json')
        if queue.file_sha(target/'evidence.zip')!=old['archive_sha256']:raise ValueError('Existing archive changed')
        print(json.dumps(old));return
    files={}
    for directory in (study.base(),ROOT/'experiments/agent_io'/study.VARIANT,report):
        for f in directory.rglob('*'):
            if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.lock','.partial','.pyc'):continue
            files[queue.rel(ROOT,f)]=f
    for name in ('code/experiment_tools/run_v13_origin_ablation.py','docs/v13_origin_ablation_20261103.txt',
                 'code/tools/v13_origin_nested_dispatch.py','code/analysis/audit_v13_origin_ablation.py',
                 'code/analysis/report_v13_origin_ablation.py','code/analysis/preserve_v13_origin_ablation.py'):
        files[name]=ROOT/name
    for name in p['inputs_sha256']:files[name]=ROOT/name
    manifest={name:dict(sha256=queue.file_sha(path),bytes=path.stat().st_size) for name,path in sorted(files.items())}
    queue.write_once(target/'manifest.json',manifest)
    archive=target/'evidence.zip';temporary=target/'evidence.zip.partial'
    if archive.exists() or temporary.exists():raise ValueError('Preserve earlier unsealed archive attempt before retrying')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.write(target/'manifest.json','PRESERVATION_MANIFEST.json')
    with zipfile.ZipFile(temporary) as z:
        for name,entry in manifest.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                while block:=f.read(1<<20):h.update(block)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Archived member changed: '+name)
    temporary.replace(archive);study.check(True)
    receipt=dict(status='SEALED_AND_VERIFIED',created_utc=datetime.now(timezone.utc).isoformat(),
        scientific_protocol_sha256=p['protocol_sha256'],files=len(manifest),factorial_traces=160,
        new_control_trajectories=128,new_control_budgeted_queries=2560,targeted_one_step_probes=12,
        archive_sha256=queue.file_sha(archive),archive_bytes=archive.stat().st_size,
        manifest_sha256=queue.file_sha(target/'manifest.json'),all_archived_member_hashes_verified=True,
        historical_sources_results_and_releases_unchanged=True,
        original_language_responses='Exact seeded copies plus preserved references to prior sealed V13 archives',
        requested_model='gpt-6.1-terra medium; orchestrator receipts, not provider weight attestation')
    queue.write_once(target/'receipt.json',receipt)
    queue.write_once(report/'preservation_receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
