"""Freeze V6 before querying, preserving source and exact-context caches."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study


def copy_verified(source, target):
    if target.exists():
        if study.sha(source) != study.sha(target):
            raise ValueError('Existing destination content differs: '+str(target))
        return 0
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if study.sha(source) != study.sha(target):
        raise ValueError('Copy verification failed')
    return 1


def main():
    reports = ROOT/'experiments/reports/knowledge_loop_20261101'
    receipt_path = reports/'v6_freeze_receipt.json'
    if receipt_path.exists():
        raise ValueError('V6 is already frozen; inspect its receipt, do not refreeze')
    folder = study.paths(ROOT,'v6').resolve()
    if not folder.is_relative_to(ROOT) or list((folder/'runs').glob('*/state.json')):
        raise ValueError('V6 already has activity or an unsafe path')
    p = study.prepare(ROOT,'v6',seed=0)
    old = study.read(study.paths(ROOT,'v5')/'protocol.json')
    confirmed = study.read(study.paths(ROOT,'v5_seed2_confirm')/'protocol.json')
    if p['prompt_sha256'] != old['prompt_sha256'] or p['inputs'] != old['inputs']:
        raise ValueError('V6 must preserve V5 public inputs and prompt')
    # Protected production dependencies must remain byte-identical to V5.
    for name, expected in old['source_sha256'].items():
        if name.endswith('knowledge_study.py') or name.endswith('knowledge_gp.py') or name.endswith('knowledge_loop.py'):
            continue  # Existing, previously audited seed2 input adapter plus V6 runner dispatch.
        if study.sha(ROOT/name) != expected:
            raise ValueError('Unregistered dependency change: '+name)
    for name in ('code/chemist_system/chem/benchmarks/knowledge_gp.py','code/chemist_system/knowledge_loop.py'):
        if study.sha(ROOT/name) != confirmed['source_sha256'][name]:
            raise ValueError('Previously audited numerical/input adapter changed: '+name)
    snapshot = ROOT/'runtime/knowledge_loop/snapshots/v6'
    for name, expected in p['source_sha256'].items():
        if study.sha(ROOT/name) != expected:
            raise ValueError('Source changed during freeze')
        copy_verified(ROOT/name,snapshot/name)
    for name in (p['design_document_path'],p['goal_config_path']):
        copy_verified(ROOT/name,snapshot/name)
    copy_verified(folder/'protocol.json',snapshot/'protocol.json')
    counts = {}
    for kind in ('cache','requests'):
        origin = study.paths(ROOT,'v5')/kind
        copied = files = size = 0
        for source in sorted(origin.rglob('*')):
            if not source.is_file() or source.suffix not in ('.npz','.pt','.json'):
                continue
            target = folder/kind/source.relative_to(origin)
            if not target.resolve().is_relative_to(folder):
                raise ValueError('Cache destination escaped the variant')
            copied += copy_verified(source,target)
            files += 1
            size += source.stat().st_size
        counts[kind] = dict(files=files,copied=copied,bytes=size)
    study.check(ROOT,'v6')
    milestone = subprocess.run(['git','rev-parse','--short','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    result = dict(frozen_before_query=True,time_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['protocol_sha256'],prompt_sha256=p['prompt_sha256'],
        design_sha256=p['design_document_sha256'],source_files=len(p['source_sha256']),
        snapshot=str(snapshot),code_milestone=milestone,tasks=8,arms=['bo','full'],
        seed=0,initial=4,queries=16,prefill=counts,
        reuse_policy='GP exact observed-prefix hash; LLM exact full public context/system/model/effort hash. No run state or decisions copied.')
    receipt_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
