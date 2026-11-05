"""Validate the authoritative table and retain active/superseded custody."""
from pathlib import Path
import json
import sys
import math
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/tools')]
import numpy as np
import pandas as pd
from chemical_hypothesis_bo import study as s
import chemical_hypothesis_active as active


def main():
    active.activate();old=s.read(s.REPORT/'audit.json');fix=s.read(active.FIX_REPORT/'audit.json')
    assert old['status']=='PASS_REPLAY_AND_CUSTODY' and fix['status']=='PASS_CORRECTED_DOYLE_REPLAY'
    frame=pd.read_csv(s.REPORT/'unified_results.csv');assert len(frame)==40
    assert len(frame[['task','method']].drop_duplicates())==40
    assert not frame[['SearchAUC','final_best','initial_best']].isna().any().any()
    manifest={};active_rows=pd.read_csv(s.REPORT/'audit_trajectories.csv')
    active_rows=pd.concat([active_rows[active_rows.task!='doyle'],pd.DataFrame(fix['rows'])],ignore_index=True)
    active_rows.to_csv(s.REPORT/'active_audit_trajectories.csv',index=False,encoding='utf-8-sig')
    for task in s.native.TASKS:
        for arm in ('fixed','adaptive'):
            folder=active.folder_for(task,arm);state=s.read(folder/'state.json')
            assert state['status']=='COMPLETE' and len(state['history'])==20
            values=[max(x['raw_value'] for x in state['history'][:q+4]) for q in range(1,17)]
            row=frame[(frame.task==s.SHORT[task])&(frame.method==arm)].iloc[0]
            assert abs(sum(values)/16-row.SearchAUC)<1e-9
            assert abs(max(x['raw_value'] for x in state['history'])-row.final_best)<1e-9
            for path in [folder/'state.json',folder/'initial_commit.json',*sorted((folder/'rounds').glob('*_commit.json'))]:manifest[path.relative_to(ROOT).as_posix()]=s.sha(path)
    receipts=[]
    for path in active.responses():
        saved=s.read(path);af=(ROOT/saved['request']).parent;req=s.read(af/'request.json');receipt=s.read(af/'response_receipt.json')
        assignment=s.read(af/'assignment.json');export=s.read(af/'export.json')
        assert s.sha(af/'response.raw.json')==receipt['raw_sha256']
        assert s.sha(af/'assignment.json')==receipt['assignment_sha256']
        assert assignment['spawn_request']==s.read(af/'invocation.json')
        assert assignment['spawn_request']['fork_turns']=='none'
        assert s.read(ROOT/export['output'])==saved['raw']
        receipts.append(dict(task=req['task'],kind=req['kind'],query=req['query'],agent=receipt['agent_id'],
            raw_sha256=receipt['raw_sha256'],request_sha256=req['request_sha256']))
    assert len(receipts)==32 and len({r['agent'] for r in receipts})==32
    pd.DataFrame(receipts).to_csv(s.REPORT/'active_LLM_receipts.csv',index=False,encoding='utf-8-sig')
    feedback=pd.read_csv(s.REPORT/'posthoc_feedback_effects.csv');refresh=pd.read_csv(s.REPORT/'posthoc_refresh_effects.csv')
    decisions=pd.read_csv(s.REPORT/'posthoc_decisions.csv');assert len(decisions)==256
    assert len(decisions[decisions.arm=='adaptive'])==128
    assert len(feedback)==int(decisions.feedback_only_changes_choice.fillna(False).astype(bool).sum())
    p=s.read(active.FIX/'protocol.json')
    for group in ('sources_sha256','inputs_sha256','protected_sha256'):
        for path,digest in p[group].items():assert s.sha(ROOT/path)==digest,path
    s.write(s.REPORT/'active_checkpoint_hashes.json',manifest,replace=True)
    value=dict(status='PASS_ACTIVE_COMPARISON',trajectories=16,active_queries=256,all_executed_queries_with_superseded=288,
        active_LLM_requests=32,total_executed_LLM_requests=36,metrics_rows=40,
        preserved_historical_artifacts=len(p['protected_sha256']),
        active_results_sha256=s.sha(s.REPORT/'unified_results.csv'),
        active_trajectory_manifest_sha256=s.sha(s.REPORT/'active_checkpoint_hashes.json'),
        active_receipts_sha256=s.sha(s.REPORT/'active_LLM_receipts.csv'),
        feedback_only_point_changes=len(feedback),feedback_positive_best_changes=int((feedback.immediate_best_gain>1e-10).sum()),
        feedback_negative_best_changes=int((feedback.immediate_best_gain< -1e-10).sum()),
        refresh_calls=len(refresh),refresh_immediate_point_changes=int(refresh.changed.sum()),
        refresh_positive_best_changes=int((refresh.immediate_best_gain>1e-10).sum()),
        semantic_correction='Only the MTBD common-name annotation in Doyle was corrected; original runs retained.',
        scope='Single seed, independent LLM responses; same-prefix oracle diagnostics are not full causal trajectory ablations.',
        sources={Path(__file__).name:s.sha(__file__),**{n:s.sha(ROOT/'code/tools'/n) for n in ('chemical_hypothesis_active.py','analyze_chemical_hypothesis_bo.py','report_chemical_hypothesis_comparison.py')}})
    s.write(s.REPORT/'active_audit.json',value,replace=True)
    print(json.dumps(value))


if __name__=='__main__':main()
