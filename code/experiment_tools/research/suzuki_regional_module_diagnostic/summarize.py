"""Verify and summarize observed full ladders and retrospective local probes."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as st

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
METHODS=('G0_gate','G1_gate_KF','G2_gate_KF_B4','G3_gate_KF_B4_event')
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def csvwrite(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(case,method):return read(ROOT/f'data/runs/suzuki_regional_ablation/runs/c{case}_s0_assistant_initial_{method}/state.json')
def curve(s):return [max(v['y'] for v in s['history'][:i])*100 for i in range(5,21)]
def metric(s):
    ys=[v['y']*100 for v in s['history']]
    return {'AUC':st.mean(curve(s)),'final':max(ys),'first_best_added_round':ys.index(max(ys))-3}


def main():
    gap={};ladder={};local={};ladder_rows=[];ladder_rounds=[];branch_rows=[];snapshots={}
    data={c:read(OUT/f'case{c}.json') for c in (3,4)}
    for case,d in data.items():
        states={m:load(case,m) for m in METHODS}
        path=ROOT/'data/runs/suzuki_mixed_kalman/base/c3_s0/state.json' if case==3 else ROOT/'data/runs/suzuki_mixed_comparison/runs/c4_s0_assistant_initial_LLINBO_matched/state.json'
        baseline=read(path);new=states[METHODS[3]];baseline_curve=curve(baseline);new_curve=curve(new)
        assert baseline['history'][:4]==new['history'][:4]
        contributions=[(b-a)/16 for a,b in zip(new_curve,baseline_curve)]
        total=st.mean(baseline_curve)-st.mean(new_curve)
        assert abs(sum(contributions)-total)<1e-10
        ranges=[(1,9),(10,16)] if case==3 else [(1,5),(6,10),(11,16)]
        gap[str(case)]={'new':metric(new),'baseline':metric(baseline),'baseline_minus_new_AUC_pp':total,
                       'segments':[{'first_round':lo,'last_round':hi,'AUC_gap_pp':sum(contributions[lo-1:hi]),
                                    'gap_share_percent':sum(contributions[lo-1:hi])/total*100} for lo,hi in ranges]}
        parents=[baseline,*[states[m] for m in METHODS[:-1]]]
        ladder[str(case)]={}
        for parent,child,label in zip(parents,[states[m] for m in METHODS],('initial_policy_difference','KF','B4_given_KF','refresh_given_KF_B4')):
            pc,cc=curve(parent),curve(child);delta=st.mean(cc)-st.mean(pc)
            first=next((t for t in range(1,17) if parent['history'][t+3]!=child['history'][t+3]),None)
            row={'case':case,'increment':label,'parent':parent['method'],'child':child['method'],
                 'AUC_delta_pp':delta,'final_yield_delta_pp':metric(child)['final']-metric(parent)['final'],
                 'first_different_selection_added_round':first}
            ladder[str(case)][label]=row;ladder_rows.append(row)
            for t,(a,b) in enumerate(zip(pc,cc),1):
                ladder_rounds.append({'case':case,'increment':label,'added_round':t,
                    'parent_incumbent_percent':a,'child_incumbent_percent':b,'AUC_delta_contribution_pp':(b-a)/16})
        assert abs(sum(r['AUC_delta_pp'] for r in ladder[str(case)].values())+total)<1e-10
        pool=read(ROOT/f'data/runs/suzuki_guarded/inputs/case{case}/pool.json')
        with np.load(ROOT/f'data/runs/suzuki_guarded/inputs/case{case}/oracle.npz',allow_pickle=False) as f:oracle=dict(zip(pool['ids'],f['values']))
        for row in d['variants']:
            t=row['added_round'];before=max(h['y'] for h in new['history'][:t+3])*100
            y=oracle[row['alternative_id']]*100;actual=oracle[row['actual_id']]*100
            assert abs(y-row['alternative_yield_percent'])<1e-10
            assert abs(max(before,y)-row['alternative_incumbent_percent'])<1e-10
            assert abs(max(before,actual)-max(before,y)-row['actual_minus_alternative_incumbent_pp'])<1e-10
            if row['variant']!='ungated_hold_field':assert row['inside_actual_budget']
        local[str(case)]={}
        for variant in ('base_only','no_KF_hold_region','no_B4_hold_alpha','no_B4_native_bank','no_refresh_this_round','ungated_hold_field'):
            rs=[r for r in d['variants'] if r['variant']==variant]
            local[str(case)][variant]={'compared_rounds':len(rs),'changed_rounds':[r['added_round'] for r in rs if not r['same_choice']],
                'positive_immediate_incumbent_rounds':[r['added_round'] for r in rs if r['actual_minus_alternative_incumbent_pp']>1e-10],
                'negative_immediate_incumbent_rounds':[r['added_round'] for r in rs if r['actual_minus_alternative_incumbent_pp']<-1e-10]}
        if case==4:
            for t,event in enumerate(baseline['decisions'],1):
                branch_rows.append({'added_round':t,'proposed_id':event['proposed_id'],'selected_id':event['selected_id'],
                    'accepted':event['accepted'],'proposal_loss':event['proposal_logei_loss'],'budget':event['budget_logei'],
                    'offline_proposed_yield_percent':float(oracle[event['proposed_id']]*100),
                    'actual_selected_yield_percent':float(oracle[event['selected_id']]*100)})
        for call in new['calls']:
            context=read(ROOT/'data/runs/suzuki_regional_ablation/requests'/call['context_sha256']/'context.json')
            observed=context['public']['observations']
            for i,h in enumerate(new['history'][:call['iteration']+3],1):
                assert observed[i-1]['index']==i
                j=pool['ids'].index(h['candidate_id'])
                assert observed[i-1]['z']==[round(float(x),4) for x in pool['z'][j][:3]]
            snapshots[f'case{case}_round{call["iteration"]}']={'context_sha256':call['context_sha256'],
                 'input_has_named_feature_values':False,'z_order':['residence_time','temperature','catalyst_loading'],
                 'fixed_slots':context['public']['fixed_slots'],'actions':call['plan']['actions']}
    protocol=read(OUT/'analysis_protocol.json')
    for p,h in protocol['source_sha256'].items():assert hashfile(ROOT/p)==h
    assert protocol['script_sha256']==hashfile(OUT/'analyze.py')
    counts={'actual_prefix_replays':sum(len(d['rounds']) for d in data.values()),
            'all_probe_rows_including_controls':sum(len(d['variants']) for d in data.values())}
    counts['single_step_alternatives']=counts['all_probe_rows_including_controls']-counts['actual_prefix_replays']
    result={'gap':gap,'full_ladder_conditional_increments':ladder,'local_probe_summary':local,'counts':counts,
            'coordinate_error_evidence':read(OUT/'coordinate_evidence.json'),'event_inputs':snapshots,
            'limits':['Ordered full-trajectory effects depend on policy history and insertion order.',
                      'Single-step alternatives freeze each actual prefix, not future rollouts; do not sum them into AUC module contributions.',
                      'Targets and oracle residual contrasts are retrospective diagnostics, never candidate-ranking or knowledge inputs.',
                      'One seed and two cases: improvement proposals remain untested.']}
    write(OUT/'results.json',result);csvwrite(OUT/'ladder_metrics.csv',ladder_rows)
    csvwrite(OUT/'ladder_rounds.csv',ladder_rounds);csvwrite(OUT/'LLINBO_case4_branches.csv',branch_rows)
    protected=read(ROOT/'research/suzuki_regional_ablation/release_inventory.json')
    for f in protected['files']:assert hashfile(ROOT/f['path'])==f['sha256']
    write(OUT/'validation_receipt.json',{'status':'PASS',**counts,'sources_unchanged':len(protocol['source_sha256']),
          'previous_release_files_unchanged':len(protected['files']),'new_GP_fits':0,'new_LLM_calls':0,
          'actual_controls_reproduced':32,'operator_compiles_in_loop':0,
          'results_sha256':hashfile(OUT/'results.json'),'verification':'Oracle outcomes/incumbents recomputed; full-ladder identities and event-coordinate inputs checked. Not a new trajectory performance qualification.'})
    print(json.dumps({'gap':gap,'local':local,'counts':counts},ensure_ascii=False))


if __name__=='__main__':main()
