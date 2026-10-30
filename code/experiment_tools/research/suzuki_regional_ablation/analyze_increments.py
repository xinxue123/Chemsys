"""Read-only, full-trajectory exports; never fit a GP or query an oracle/API."""
from pathlib import Path
import csv
import json
import statistics


ROOT = Path('data/runs/suzuki_regional_ablation')
OUT = Path('research/suzuki_regional_ablation')
METHODS = ('G0_gate', 'G1_gate_KF', 'G2_gate_KF_B4', 'G3_gate_KF_B4_event')
REGIMES = ('correct', 'wrong_category', 'mixed', 'assistant_initial')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def csv_write(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve(state):
    return [max(v['y'] for v in state['history'][:4+t])*100 for t in range(1,17)]


def main():
    result = read(OUT/'results.json')
    states = [read(ROOT/'runs'/entry['run_id']/'state.json') for entry in read(ROOT/'manifest.json')]
    assert len(states) == 64 and all(s['status'] == 'COMPLETE' and len(s['history']) == 20 for s in states)
    by_key = {(s['case'],s['regime'],s['method']):s for s in states}
    pools = {c:read(Path(f'data/runs/suzuki_guarded/inputs/case{c}/pool.json')) for c in range(1,5)}
    points = {c:dict(zip(p['ids'],p['points'])) for c,p in pools.items()}
    observations=[]
    for s in states:
        calls={c['iteration']:c for c in s['calls']}
        for index,h in enumerate(s['history']):
            t=index-3;d=s['decisions'][t-1] if t>0 else None
            m=d['measurement'] if d else {}; call=calls.get(t,{})
            row={'case':s['case'],'seed':0,'knowledge_condition':s['regime'],'method':s['method'],
                 'total_query':index+1,'added_query':t if t>0 else '',
                 'candidate_id':h['candidate_id'],**points[s['case']][h['candidate_id']],
                 'yield_percent':h['y']*100,
                 'incumbent_percent':max(v['y'] for v in s['history'][:index+1])*100,
                 'GP_prequery_mean_percent':m.get('prequery_mu_y',0)*100 if d else '',
                 'GP_prequery_observation_sd_pp':m.get('prequery_predictive_sd_y',0)*100 if d else '',
                 'regional_shift_raw_pp':d['raw_shift_y']*100 if d else '',
                 'regional_shift_deployed_pp':d['deployed_shift_y']*100 if d else '',
                 'KF_amplitude':d['amplitude'] if d else '',
                 'KF_bank_updates_before':d['bank_before']['updates'] if d and d['bank_before'] else '',
                 'KF_bank_key':d['bank_key'] if d else '',
                 'innovation_z':m.get('innovation_z',''),
                 'budget_logEI':d['budget_logei'] if d else '',
                 'base_logEI_loss':d['base_logei_loss'] if d else '',
                 'eligible_candidates':d['eligible_candidates'] if d else '',
                 'B4_diagnosed_revoked_ids':';'.join(d['diagnosis']['revoked_ids']) if d else '',
                 'region_retained_ids':';'.join(d['retained_ids']) if d else '',
                 'refresh_status':call.get('status',''),
                 'refresh_reasons':';'.join(call.get('reasons',[])),
                 'knowledge_json':json.dumps(d['knowledge'],ensure_ascii=False,separators=(',',':')) if d else '',
                 'GP_fit_key':d['base_acquisition']['fit_key'] if d else ''}
            observations.append(row)
    csv_write('trajectory_observations.csv',observations)
    pairs=[];summaries=[]
    for c in range(1,5):
        for regime in REGIMES:
            for i,module in zip((1,2,3),('KF','B4_given_KF','event_given_KF_B4')):
                a=by_key[c,regime,METHODS[i]];b=by_key[c,regime,METHODS[i-1]]
                ca,cb=curve(a),curve(b)
                first=next((t for t in range(1,17) if a['history'][3+t]!=b['history'][3+t]),None)
                for t,(va,vb) in enumerate(zip(ca,cb),1):
                    da,db=a['decisions'][t-1],b['decisions'][t-1]
                    pairs.append({'case':c,'knowledge_condition':regime,'module':module,'added_query':t,
                                  'parent_method':b['method'],'added_module_method':a['method'],
                                  'same_observation_prefix_before_query':a['history'][:t+3]==b['history'][:t+3],
                                  'parent_candidate_id':db['selected_id'],'added_module_candidate_id':da['selected_id'],
                                  'parent_incumbent_percent':vb,'added_module_incumbent_percent':va,
                                  'incumbent_delta_pp':va-vb,'AUC_contribution_pp':(va-vb)/16,
                                  'parent_amplitude':db['amplitude'],'added_module_amplitude':da['amplitude'],
                                  'parent_budget_size':db['eligible_candidates'],'added_module_budget_size':da['eligible_candidates']})
                delta=statistics.mean(va-vb for va,vb in zip(ca,cb))
                expected=result['ladder_increments'][f'Case{c}'][regime][f'{a["method"]}_minus_{b["method"]}']['AUC_delta_mean_pp']
                assert abs(delta-expected)<1e-10
                summaries.append({'case':c,'knowledge_condition':regime,'module':module,
                                  'first_different_selection_added_query':first,
                                  'AUC_delta_pp':delta,'final_yield_delta_pp':ca[-1]-cb[-1],
                                  'positive_curve_positions':sum(va>vb+1e-10 for va,vb in zip(ca,cb)),
                                  'negative_curve_positions':sum(va<vb-1e-10 for va,vb in zip(ca,cb)),
                                  'selected_ids_differ_rounds':sum(da['selected_id']!=db['selected_id'] for da,db in zip(a['decisions'],b['decisions'])),
                                  'new_bank_rounds_added_arm':sum(d['bank_before'] is not None and d['bank_before']['updates']==0 for d in a['decisions']),
                                  'B4_applied_revocation_rounds_added_arm':sum(set(d['retained_ids'])!=set(h['id'] for h in d['knowledge']['clauses']) for d in a['decisions']),
                                  'refresh_rounds_added_arm':';'.join(str(call['iteration']) for call in a['calls']),
                                  'singleton_budget_rounds_added_arm':sum(d['eligible_candidates']==1 for d in a['decisions'])})
    csv_write('paired_round_differences.csv',pairs)
    csv_write('module_case_increments.csv',summaries)
    assert len(observations)==1280 and len(pairs)==768 and len(summaries)==48
    vs_gate={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4))]:
        vs_gate[scope]={}
        for method in METHODS[1:]:
            ds=[statistics.mean(curve(by_key[c,g,method]))-statistics.mean(curve(by_key[c,g,METHODS[0]])) for c in cases for g in REGIMES]
            vs_gate[scope][method]={'n':len(ds),'AUC_delta_mean_pp':statistics.mean(ds),
                                  'wins':sum(d>1e-10 for d in ds),'losses':sum(d<-1e-10 for d in ds),'ties':sum(abs(d)<=1e-10 for d in ds)}
    output={'source_protocol_sha256':result['protocol_sha256'],'observation_rows':len(observations),
            'paired_round_rows':len(pairs),'module_case_rows':len(summaries),'versus_gate':vs_gate,
            'limits':['Paired incumbent curves describe complete policy trajectory differences; after the first differing selection these are not matched-prefix single-action effects.',
                      'AUC contributions sum exactly to the total paired trajectory AUC delta; they are not an additive causal decomposition among modules.']}
    (OUT/'increment_analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
