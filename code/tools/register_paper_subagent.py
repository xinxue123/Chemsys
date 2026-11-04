"""Register a separate request-isolated subagent cohort without outcome access."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from paper_v12 import study as original

def main():
    from paper_v12_subagent import study as adapter
    dest=adapter.directory(ROOT);target=dest/'protocol.json'
    if target.exists():
        adapter.check(ROOT,verify_controls=True)
        print('Existing registered subagent protocol is unchanged.');return
    if any(dest.glob('s[0-9]*')):raise ValueError('Register before any new execution')
    parent=original.check(ROOT,True)
    jobs=[list(j) for j in original.JOBS if j[2]!='bo']
    jobs += [[task,seed,'memory'] for task in original.FOCUS for seed in (0,1,2,3)]
    assert len(jobs)==68 and len({tuple(j) for j in jobs})==68
    protected=deepcopy(parent['protected_controls_sha256'])
    for directory in (original.directory(ROOT),ROOT/'paper/v12'):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix!='.lock':protected[path.relative_to(ROOT).as_posix()]=original.old.sha(path)
    for path in (ROOT/'paper/v12_source.zip',ROOT/'experiments/reports/paper_v12/audit.json'):
        protected[path.relative_to(ROOT).as_posix()]=original.old.sha(path)
    p=deepcopy(parent)
    p.pop('protocol_sha256')
    p.update(variant='paper_v12_subagent',backend='codex_subagent',created_utc=datetime.now(timezone.utc).isoformat(),
             original_protocol_sha256=parent['protocol_sha256'],jobs=jobs,
             model='gpt-6.1-terra',reasoning_effort='medium',requested_model='gpt-6.1-terra',requested_reasoning_effort='medium',
             fork_turns='none',context_policy='fresh_agent_per_request_attempt',
             public_exposure_policy='opaque_exact_system_public_packet',
             decision_instructions_path='docs/paper_v12_subagent_decision_instructions.txt',
             decision_instructions_sha256=original.old.sha(ROOT/'docs/paper_v12_subagent_decision_instructions.txt'),
             adapter_source_sha256=adapter.sources(ROOT),protected_controls_sha256=protected,
             design_path='docs/paper_v12_subagent_protocol_20261101.txt',
             design_sha256=original.old.sha(ROOT/'docs/paper_v12_subagent_protocol_20261101.txt'),
             maximum_new_trajectories=68,maximum_new_queries=1088,maximum_logical_LLM_calls=1246,
             execution='Frozen policies, new subagent request transport; one fresh model-request agent per attempt; two persistent CPU workers.',
             original_API_cohort_complete=False,resolved_model_snapshot_attested=False,token_usage_available=False,
             no_performance_rerolls=True,LLM_calls_at_registration=0,initial_outcomes_accessed=False,
             comparison_boundary='New subagent cohort; same-backend V12/RBO seeds1-3 and focus removals seeds0/3. Original API results remain separate.')
    p['protocol_sha256']=original.digest(p);original.write(target,p)
    snapshot=ROOT/'runtime/knowledge_loop/snapshots/paper_v12_subagent'
    for name,expected in {**p['source_sha256'],**p['adapter_source_sha256']}.items():
        q=snapshot/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,q)
        if original.old.sha(q)!=expected:raise ValueError('Snapshot mismatch')
    q=snapshot/p['design_path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p['design_path'],q)
    shutil.copy2(target,snapshot/'protocol.json')
    adapter.check(ROOT,verify_controls=True)
    print(json.dumps(dict(protocol_sha256=p['protocol_sha256'],jobs=len(jobs),protected_files=len(protected),
                         backend=p['backend'],context_policy=p['context_policy']),indent=2))

if __name__=='__main__':main()
