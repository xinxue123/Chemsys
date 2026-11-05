"""Parallel saved-array, feedback-chain and focal-parameter custody audit."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from kernel_contrast_revision import study,policy


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_one(task,arm):
    torch.set_num_threads(1)
    folder=study.folder_for(task,arm);pool=study.native.load_task(ROOT,task,seed=0)
    final=study.read(folder/'state.json');p=study.read(study.BASE/'protocol.json')
    state=dict(task=task,arm=arm,status='RUNNING',protocol_sha256=p['protocol_sha256'],
        history=deepcopy(final['history'][:4]),decisions=[],parents=[],log_weights=None,initial_log_weights=None)
    errors=[];checkpoints={};feedback_changed=weight_updates=protected_BO=0;match_counts={};model_custody=0
    for query,saved in enumerate(final['decisions'],1):
        cp_path=folder/'rounds'/f'{query:02d}_commit.json';cp=study.read(cp_path)
        assert cp['commit_sha256']==study.models.digest({k:v for k,v in cp.items() if k!='commit_sha256'})
        assert cp['prestate_sha256']==study.models.digest(state)
        assert cp['counters']['operator_compiles_in_loop']==0
        meta=study.read(folder/'rounds'/f'{query:02d}_models.json')
        with np.load(folder/'rounds'/f'{query:02d}_arrays.npz',allow_pickle=False) as z:
            context={k:z[k].copy() for k in ('available','base_logei','tie_priority')}
            context['epsilon']=cp['decision']['epsilon']
            available=np.flatnonzero(context['available']);maximum=context['base_logei'][available].max()
            top=available[context['base_logei'][available]==maximum]
            context['bo_index']=int(top[np.argmin(context['tie_priority'][top])])
            for i,m in enumerate(meta):
                assert m['id']==z['expert_ids'][i]
                m.update(mu=z['expert_mu'][i].copy(),sd=z['expert_sd'][i].copy(),
                    observation_sd=z['expert_observation_sd'][i].copy())
            d,arrays=policy.select(pool,state['history'],context,meta,arm,state['log_weights'],query,
                initial_stored=state['initial_log_weights'])
            np.testing.assert_array_equal(arrays['budget'],z['budget'])
            np.testing.assert_array_equal(arrays['qualified'],z['qualified'])
            errors.append(float(np.max(np.abs(arrays['score']-z['score']))))
            np.testing.assert_allclose(arrays['weights'],z['weights'],rtol=0,atol=1e-12)
        actual=cp['decision']
        for key in ('selected_id','BO_reference_id','reason','representative_pair','fixed_initial_selected_id',
                    'feedback_weights_change_selection','matching'):
            assert d[key]==actual[key],(task,arm,query,key)
        for key in ('score','unmatched_score','maximum_budget_score','effective_expert_count'):
            assert abs(d[key]-actual[key])<1e-10,(task,arm,query,key)
        assert actual['native_logEI_gap']<=actual['epsilon']+1e-12
        observation=final['history'][query+3]
        fb=policy.feedback(actual,observation)
        assert fb==saved['feedback']
        assert saved['candidate_id']==actual['selected_id']==observation['candidate_id']
        receipt=study.read(folder/'rounds'/f'{query:02d}_feedback.json')
        assert receipt==dict(commit_sha256=cp['commit_sha256'],observation=observation,feedback=fb)
        assert cp_path.stat().st_mtime_ns <= (folder/'rounds'/f'{query:02d}_feedback.json').stat().st_mtime_ns
        if arm=='feedback':
            if state['log_weights'] is not None:
                expected=policy.normalize([state['log_weights'][m['id']] for m in meta])
                np.testing.assert_allclose(expected,[actual['log_weights_before'][m['id']] for m in meta],rtol=0,atol=1e-12)
            weight_updates+=fb['maximum_weight_change']>1e-12
        if arm=='soft_fixed':
            assert fb['log_weights_after']==actual['log_weights_before']
            if state['initial_log_weights'] is not None:
                np.testing.assert_allclose([actual['log_weights_before'][m['id']] for m in meta],
                    [state['initial_log_weights'][m['id']] for m in meta],rtol=0,atol=1e-12)
        if actual['feedback_weights_change_selection']:
            feedback_changed+=1;protected_BO+=actual['selected_id']==actual['BO_reference_id']
        if actual['matching'] is not None:
            kind=actual['matching']['kind'];match_counts[kind]=match_counts.get(kind,0)+1
            assert actual['matching']['anchor_id'] in {h['candidate_id'] for h in state['history']}
        native_path=study.BASE/'cache'/study.SHORT[task]/(cp['primary_fit_key']+'.pt')
        native_saved=torch.load(native_path,map_location='cpu',weights_only=False)
        native_state=native_saved['state'];assert state_digest(native_state)==cp['native_state_sha256']
        checkpoints[native_path.relative_to(ROOT).as_posix()]=study.sha(native_path)
        for m in meta:
            path=ROOT/m['checkpoint'];checkpoints[m['checkpoint']]=study.sha(path)
            if arm=='geometry':
                continue
            saved_model=torch.load(path,map_location='cpu',weights_only=False)
            assert saved_model['native_state_sha256']==cp['native_state_sha256']
            assert saved_model['expert']==m['id']
            expected_y={h['candidate_id']:h['y'] for h in state['history']}
            assert dict(zip(saved_model['training_ids'],saved_model['training_y']))==expected_y
            changes={('covar_module.'+c['module']+'.raw_lengthscale').replace('..','.') :c['local_axis'] for c in m['changed_components']}
            for name,value in native_state.items():
                other=saved_model['state'][name]
                if name in changes:
                    keep=torch.ones_like(value,dtype=torch.bool);keep[...,changes[name]]=False
                    assert torch.equal(value[keep],other[keep]),(task,query,m['id'],name)
                    assert torch.isfinite(other).all()
                else:
                    assert torch.equal(value,other),(task,query,m['id'],name)
            assert all(np.isfinite(np.asarray(v)).all() for v in m['parameters'].values())
            model_custody+=1
        state['history'].append(deepcopy(observation));state['decisions'].append(deepcopy(saved))
        state['parents']=cp['next_parents'];state['log_weights']=fb['log_weights_after']
        if query==1:state['initial_log_weights']=actual['log_weights_before']
        state['status']='COMPLETE' if query==16 else 'RUNNING'
    assert state==final
    assert len({h['candidate_id'] for h in state['history']})==20
    assert max(errors)<1e-10
    return dict(task=study.SHORT[task],arm=arm,replayed=16,max_score_error=max(errors),
        weight_updates=weight_updates,feedback_changed=feedback_changed,feedback_follows_BO=protected_BO,
        match_counts=match_counts,sidecar_parameter_checks=model_custody,checkpoint_hashes=checkpoints)


def main():
    p=study.check(True);rows=[];checkpoints={}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures=[executor.submit(verify_one,*job) for job in study.JOBS]
        for future in as_completed(futures):
            value=future.result();checkpoints.update(value.pop('checkpoint_hashes'));rows.append(value)
    # Fixed and adaptive soft weighting have exactly the same startup information.
    startup_checks=[]
    for task in study.native.TASKS:
        s=study.read(study.folder_for(task,'soft_fixed')/'rounds/01_commit.json')['decision']
        f=study.read(study.folder_for(task,'feedback')/'rounds/01_commit.json')['decision']
        assert s['selected_id']==f['selected_id']
        assert s['weights']==f['weights']
        assert s['predictions']==f['predictions']
        startup_checks.append(study.SHORT[task])
    summaries=pd.read_csv(study.REPORT/'results.csv',encoding='utf-8-sig')
    for task,arm in study.JOBS:
        state=study.read(study.folder_for(task,arm)/'state.json')
        expected=np.maximum.accumulate([h['raw_value'] for h in state['history']])[4:].mean()
        actual=summaries[(summaries.task==study.SHORT[task])&(summaries.method==arm)].iloc[0].SearchAUC
        assert abs(expected-actual)<1e-9
    pd.DataFrame(rows).to_csv(study.REPORT/'decision_replay_receipts.csv',index=False,encoding='utf-8-sig')
    study.write(study.REPORT/'participating_checkpoint_hashes.json',checkpoints,replace=True)
    audit=dict(status='PASS_FULL_KERNEL_REVISION_VERIFICATION',tasks=8,arms=4,decision_replays=sum(r['replayed'] for r in rows),
        maximum_score_error=max(r['max_score_error'] for r in rows),
        sidecar_focal_parameter_checks=sum(r['sidecar_parameter_checks'] for r in rows),
        adaptive_weight_updates=sum(r['weight_updates'] for r in rows),
        feedback_changes_next_choice=sum(r['feedback_changed'] for r in rows),
        feedback_follows_BO=sum(r['feedback_follows_BO'] for r in rows),
        matched_S_F_startups=startup_checks,SearchAUC_denominator=16,SearchAUC_recomputed=True,
        protected_files_verified=len(p['protected_sha256']),participating_checkpoints=len(checkpoints),
        new_GP_fits_in_verification=0,new_queries_in_verification=0,new_LLM_calls=0)
    study.write(study.REPORT/'final_verification.json',audit,replace=True)
    print(json.dumps(audit))


if __name__=='__main__':main()
