"""Independently validate the delivered CSVs against immutable raw snapshots."""
from hashlib import sha256
from itertools import groupby
from pathlib import Path
import csv
import json
import numpy as np

WORK=Path('research/suzuki_case2_csv_export');ROOT=Path('data/runs/suzuki_kalman_knowledge');OUT=Path('outputs/case2_kf_event_b4_seed0')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def fh(p):return sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def arr(rows,key):return np.array([float(row[key]) for row in rows])


def main():
    manifest=read(WORK/'prepared_manifest.json');state=read(ROOT/'runs/c2_s0/state.json');pool=read('data/runs/suzuki_guarded/inputs/case2/pool.json');ids=pool['ids'];ix={key:j for j,key in enumerate(ids)}
    oracle=load('data/runs/suzuki_guarded/inputs/case2/oracle.npz')['values'];z=np.array(pool['z']);outputs=[]
    for name in ('case2_candidates.csv','case2_bo_predictions.csv','case2_decisions.csv'):
        p=OUT/name;assert p.read_bytes().startswith(b'\xef\xbb\xbf');outputs.append({'file':name,'bytes':p.stat().st_size,'sha256':fh(p)})
    with (OUT/'case2_candidates.csv').open(encoding='utf-8-sig',newline='') as f:candidates=list(csv.DictReader(f))
    assert len(candidates)==4692 and [r['candidate_id'] for r in candidates]==ids
    assert np.allclose(arr(candidates,'oracle_yield_pct_posthoc'),oracle*100,rtol=0,atol=1e-12)
    for row,p in zip(candidates,pool['points']):
        assert row['catalyst']==p['catalyst'] and float(row['temperature_C'])==p['temperature'] and float(row['residence_time_s'])==p['t_res'] and float(row['catalyst_loading_mol_pct'])==p['catalyst_loading']
    assert sum(int(r['initial_design']) for r in candidates)==4 and sum(int(r['evaluated_in_this_run']) for r in candidates)==20
    count=0;rounds=0;decisions=[];max_error=0.
    with (OUT/'case2_bo_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        for t,group in groupby(csv.DictReader(f),key=lambda r:int(r['bo_round'])):
            rows=list(group);event=state['decisions'][t-1];prefix=state['history'][:t+3];seen={v['candidate_id']:v['y'] for v in prefix};j=ix[event['selected_id']]
            assert len(rows)==4692 and [r['candidate_id'] for r in rows]==ids and all(int(r['total_query'])==t+4 and int(r['observations_before'])==t+3 for r in rows)
            assert np.allclose(arr(rows,'oracle_yield_pct_posthoc'),oracle*100,rtol=0,atol=1e-12)
            p=load(ROOT/'cache/c2_s0'/(event['posterior_key']+'.npz'));base=load(ROOT/'cache/c2_s0'/(event['base_score_key']+'.npz'))['scores'];guide=load(ROOT/'cache/c2_s0'/(event['guided_score_key']+'.npz'))['scores']
            for field,expected in [('gp_mean_pct',p['mu_y']*100),('gp_epistemic_sd_pp',p['sd_y']*100),('gp_observation_sd_pp',p['sd_predictive_y']*100),('base_mc_qlogei',base),('guided_mc_qlogei',guide)]:
                error=float(np.max(np.abs(arr(rows,field)-expected)));max_error=max(max_error,error);assert error<1e-12
            available=np.array([k not in seen for k in ids]);order=sorted(np.flatnonzero(available),key=lambda j:(-float(base[j]),ids[j]));ref=order[0];eligible=available&((base[ref]-base)<=event['budget_logei'])
            gorder=sorted(np.flatnonzero(eligible),key=lambda j:(-float(guide[j]),ids[j]));assert gorder[0]==j
            assert sum(int(r['is_selected']) for r in rows)==1 and int(rows[j]['is_selected'])==1
            assert np.array_equal(arr(rows,'available_before'),available.astype(int)) and np.array_equal(arr(rows,'in_budget_set'),eligible.astype(int))
            assert sum(int(r['observed_before']) for r in rows)==len(prefix)
            for row in rows:
                key=row['candidate_id'];known=key in seen
                assert int(row['observed_before'])==int(known)
                if known:assert float(row['observed_yield_pct_before'])==seen[key]*100 and row['base_rank_available']==''
                else:assert row['observed_yield_pct_before']==''
            raw=np.zeros(4692);retained=set(event['diagnosis']['retained_ids'])
            for h in event['hypotheses']:
                if h['id'] not in retained:continue
                if h['variable']=='catalyst':m=np.isin(np.floor(z[:,3]*8).astype(int),h['categories'])
                else:
                    k=('residence_time','temperature','catalyst_loading').index(h['variable']);m=(z[:,k]>=h['bounds'][0])&(z[:,k]<=h['bounds'][1])
                raw+=event['trust'][h['id']]*(h['effect_pp']/100)*(m.astype(float)-m.mean())
            delta=np.maximum(np.minimum(raw,.5*p['sd_y']),-.5*p['sd_y'])
            assert np.allclose(arr(rows,'raw_knowledge_shift_pp'),raw*100,rtol=0,atol=1e-12)
            assert np.allclose(arr(rows,'knowledge_shift_pp'),delta*100,rtol=0,atol=1e-12)
            assert np.allclose(arr(rows,'guided_mean_pct'),(p['mu_y']+delta)*100,rtol=0,atol=1e-12)
            assert abs(float(rows[j]['knowledge_shift_pp'])-event['mean_change_y']*100)<1e-12
            decisions.append(rows[j]);count+=len(rows);rounds+=1
    assert count==75072 and rounds==16
    with (OUT/'case2_decisions.csv').open(encoding='utf-8-sig',newline='') as f:drows=list(csv.DictReader(f))
    assert len(drows)==20 and [int(r['total_query']) for r in drows]==list(range(1,21))
    for i,(row,h) in enumerate(zip(drows,state['history']),1):
        assert row['candidate_id']==h['candidate_id'] and float(row['actual_yield_pct'])==h['y']*100 and row['decision_reason_zh']
        if i<=4:
            assert row['phase']=='initial' and row['gp_mean_pct']==row['guided_mc_qlogei']==row['bo_round']==''
        else:
            t=i-4;event=state['decisions'][t-1];selected=decisions[t-1]
            assert int(row['bo_round'])==t and int(row['guided_rank_in_budget'])==1
            for k in ('gp_mean_pct','gp_epistemic_sd_pp','gp_observation_sd_pp','knowledge_shift_pp','guided_mean_pct','base_mc_qlogei','guided_mc_qlogei','base_gap_logei','budget_logei'):
                assert float(row[k])==float(selected[k])
            assert int(row['eligible_candidates'])==event['eligible_candidates']
            assert bool(int(row['same_prefix_BO_choice_changed']))==(event['selected_id']!=event['reference_id'])
    assert [int(r['bo_round']) for r in drows[4:] if int(r['LLM_called_this_round'])]==[1,3,5,7]
    for p,expected in manifest['inputs'].items():assert fh(p)==expected,p
    receipt={'status':'PASS','candidate_rows':4692,'prediction_rows':count,'decision_rows':20,'BO_rounds':rounds,'selected_BO_rows':16,'initial_rows':4,
        'complete_candidate_round_pairs':True,'max_snapshot_numeric_difference':max_error,'all_source_hashes_unchanged':True,
        'future_oracle_labels_marked_posthoc':True,'no_new_API_calls':True,'no_new_oracle_queries':True,'no_new_GP_fits':True,'outputs':outputs}
    (WORK/'validation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
