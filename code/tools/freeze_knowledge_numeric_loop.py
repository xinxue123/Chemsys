"""Freeze V8 numerical closed loop before any new model call or experiment."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from freeze_knowledge_anchor_loop import copy_verified


def main():
    reports=ROOT/'experiments/reports/knowledge_loop_20261101'
    receipt=reports/'v8_freeze_receipt.json'
    if receipt.exists():raise ValueError('V8 already frozen; never overwrite its receipt')
    folder=study.paths(ROOT,'v8').resolve()
    if (not folder.is_relative_to(ROOT) or list((folder/'runs').glob('*/state.json'))
            or (folder/'requests').exists()):raise ValueError('Unsafe destination or existing V8 activity')
    p=study.prepare(ROOT,'v8',seed=0)
    old=study.read(study.paths(ROOT,'v7')/'protocol.json')
    if (p['inputs']!=old['inputs'] or p['model']!=old['model']
            or p['reasoning_effort']!=old['reasoning_effort'] or p['prompt_sha256']==old['prompt_sha256']):
        raise ValueError('V8 must preserve inputs/model but register a new quantitative prompt')
    for name,expected in old['source_sha256'].items():
        if name.endswith(('knowledge_study.py','knowledge_loop.py')):continue
        if study.sha(ROOT/name)!=expected:raise ValueError('Protected old dependency changed: '+name)
    snapshot=ROOT/'runtime/knowledge_loop/snapshots/v8'
    for name,expected in p['source_sha256'].items():
        if study.sha(ROOT/name)!=expected:raise ValueError('Source changed during freeze')
        copy_verified(ROOT/name,snapshot/name)
    for name in (p['goal_config_path'],p['design_document_path']):copy_verified(ROOT/name,snapshot/name)
    copy_verified(folder/'protocol.json',snapshot/'protocol.json')
    origin=study.paths(ROOT,'v7')/'cache';files=copied=size=0
    for source in sorted(origin.rglob('*')):
        if not source.is_file() or source.suffix not in ('.pt','.npz','.json'):continue
        target=folder/'cache'/source.relative_to(origin)
        if not target.resolve().is_relative_to(folder):raise ValueError('Cache escapes variant')
        copied+=copy_verified(source,target);files+=1;size+=source.stat().st_size
    study.check(ROOT,'v8')
    milestone=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    result=dict(frozen_before_query=True,time_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['protocol_sha256'],prompt_sha256=p['prompt_sha256'],
        design_sha256=p['design_document_sha256'],source_files=len(p['source_sha256']),
        snapshot=str(snapshot),code_milestone=milestone,seed=0,initial=4,queries=16,tasks=8,
        stage1=['bo','full'],stage2=['verification','no_calibration'],
        stage1_gate='at least5of8strictAUCwins and at least1 actual weighted choice',
        prefill=dict(GP_files=files,copied=copied,bytes=size,LLM_requests=0),
        reuse_policy='GP exact observed-prefix hash only; no old quantitative LLM responses',
        resource_policy='8 independent CPU workers; Torch1thread; fixed GP cache; birth kernel blocks<=256; no JIT')
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
