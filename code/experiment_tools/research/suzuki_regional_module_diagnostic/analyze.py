"""Retrospective matched-prefix module probes for the current regional G3.

No GP fits, new LLM calls, future rollouts, or oracle-based candidate ranking.
Alternative yields are looked up only AFTER deterministic policy selection.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import hashlib
import json
import statistics

import numpy as np

from chem.benchmarks import regional_ablation_suzuki_engine as e
from chem.benchmarks import regional_ablation_suzuki_study as s
from chem.benchmarks.suzuki_domain import CATALYSTS,digest

OUT=Path('research/suzuki_regional_module_diagnostic')
REGIME='assistant_initial'
METHODS=e.METHODS
CASES=(3,4)
TARGETS={3:['p00221','p00661','p01027','p00651','p00545','p00953','p00613'],
         4:['p02436','p00730','p00946','p00874','p01589']}
PROBE_ROUNDS={3:{1,2,4,6,7,8,10,11,12},4:{1,2,3,5,6,10,11}}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(type(value).__name__)
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=native)+'\n',encoding='utf8')
def filehash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csvwrite(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class ProbeCache(e.RegionalAblationCache):
    def _evaluate(self,history,tilt,value,fit,key,amplitude,spec):
        self.ensure(key+'.npz')
        return super()._evaluate(history,tilt,value,fit,key,amplitude,spec)


def controls(case):
    ladder={m:read(s.ROOT/'runs'/f'c{case}_s0_{REGIME}_{m}'/'state.json') for m in METHODS}
    if case==3:
        baseline=read(s.prior.previous.ROOT/'base'/f'c{case}_s0'/'state.json')
    else:
        baseline=read(s.prior.ROOT/'runs'/f'c{case}_s0_{REGIME}_LLINBO_matched'/'state.json')
    return ladder,baseline


def analyze_case(case):
    pool=s.inputs(case);ids=pool['ids'];ix={k:i for i,k in enumerate(ids)}
    with np.load(s.prior.original.ROOT/'inputs'/f'case{case}'/'oracle.npz',allow_pickle=False) as f:oracle=f['values'].copy()
    ladder,baseline=controls(case);actual=ladder[METHODS[3]]
    sources=[root/folder/f'c{case}_s0' for root in (s.ROOT,s.prior.ROOT,s.prior.previous.ROOT)
             for folder in ('cache','audit_cache','qualification_cache')]
    cache=ProbeCache(pool,OUT/'cache'/f'c{case}',case,sources=sources)
    state=deepcopy(actual);state.update(history=actual['history'][:4],decisions=[],calls=[],banks={},knowledge=deepcopy(actual['initial_knowledge']))
    rounds=[];variants=[];targets=[];clauses=[];refresh=[]
    sy=[h['y']*100 for h in actual['history']];by=[h['y']*100 for h in baseline['history']]
    sc=np.maximum.accumulate(sy);bc=np.maximum.accumulate(by)
    for t,event in enumerate(actual['decisions'],1):
        before=deepcopy(state);call=next((c for c in actual['calls'] if c['iteration']==t),None)
        if call:state['knowledge']=deepcopy(call['plan']['knowledge'])
        decision=e.setup(cache,state,pool,t)
        j=int(decision['selected']);assert ids[j]==event['selected_id']
        assert digest(state['knowledge'])==digest(event['knowledge'])
        assert json.loads(json.dumps(decision['bank']))==event['bank_before']
        assert abs(decision['amplitude']-event['amplitude'])<1e-14
        _,fit=cache.model(state['history']);ys=float(fit['std']);base=decision['base'];scores=decision['score']
        available,eligible=decision['available'],decision['eligible'];inc=max(h['y'] for h in state['history'])
        stats=decision['stats'];actual_after=max(inc,oracle[j])*100
        knowledge=state['knowledge'];all_ids=sorted(h['id'] for h in knowledge['clauses'])
        def add(name,score,mask,alpha=None,retained=None,key=None):
            selected=int(e.common.regional.argmax(score,mask,ids))  # oracle is not an input
            y=float(oracle[selected]);after=max(inc,y)*100
            row={'case':case,'added_round':t,'total_query':t+4,'variant':name,
                 'actual_id':ids[j],'alternative_id':ids[selected],'same_choice':selected==j,
                 'actual_yield_percent':float(oracle[j]*100),'alternative_yield_percent':y*100,
                 'incumbent_before_percent':inc*100,'actual_incumbent_percent':actual_after,
                 'alternative_incumbent_percent':after,'actual_minus_alternative_yield_pp':float((oracle[j]-y)*100),
                 'actual_minus_alternative_incumbent_pp':actual_after-after,
                 'alternative_alpha':alpha,'retained_ids':';'.join(retained or []),
                 'bank_key':key,'alternative_base_score_loss':float(base[decision['reference']]-base[selected]),
                 'inside_actual_budget':bool(eligible[selected]),
                 'GP_mean_percent':float(stats['mu_y'][selected]*100),'GP_latent_SD_pp':float(stats['sd_y'][selected]*100),
                 'alternative_catalyst':pool['points'][selected]['catalyst']}
            variants.append(row);return row
        def field(name,know,retained,alpha,mask=eligible):
            score,_,_=cache.calibrated_score(state['history'],know,retained,alpha)
            return add(name,score,mask,alpha,retained,e.region_key(know,retained))
        add('actual',scores,eligible,decision['amplitude'],decision['retained'],decision['bank_key'])
        add('base_only',base,available,0,[],None)
        field('no_KF_hold_region',knowledge,decision['retained'],1.)
        field('no_B4_hold_alpha',knowledge,all_ids,decision['amplitude'])
        no_b4=deepcopy(state);no_b4['method']=METHODS[1]
        d=e.setup(cache,no_b4,pool,t)
        add('no_B4_native_bank',d['score'],d['eligible'],d['amplitude'],d['retained'],d['bank_key'])
        field('no_KF_no_B4',knowledge,all_ids,1.)
        add('ungated_hold_field',scores,available,decision['amplitude'],decision['retained'],decision['bank_key'])
        if decision['bank'] and decision['bank']['updates']==0 and t>1 and actual['decisions'][t-2]['bank_after']:
            alpha=float(np.clip(actual['decisions'][t-2]['bank_after']['mean'][1],0,1))
            field('carry_previous_posterior_alpha_new_region',knowledge,decision['retained'],alpha)
        if call:
            d=e.setup(cache,before,pool,t)
            row=add('no_refresh_this_round',d['score'],d['eligible'],d['amplitude'],d['retained'],d['bank_key'])
            field('old_knowledge_hold_actual_alpha',before['knowledge'],d['retained'],decision['amplitude'])
            refresh.append({'case':case,'round':t,'reasons':call['reasons'],'knowledge_before':call['knowledge_before'],
                            'knowledge_after':call['plan']['knowledge'],'actions':call['plan']['actions'],
                            'summary':call['plan']['summary'],'counterfactual':row})
        if t in PROBE_ROUNDS[case]:
            for h in knowledge['clauses']:
                without={'confidence':knowledge['confidence'],'clauses':[v for v in knowledge['clauses'] if v['id']!=h['id']]}
                d=e.setup(cache,state,pool,t,knowledge=without)
                add('drop_'+h['id']+'_native_bank',d['score'],d['eligible'],d['amplitude'],d['retained'],d['bank_key'])
        masks=e.membership(pool,knowledge)
        residual=oracle-stats['mu_y']
        for h,mask in zip(knowledge['clauses'],masks):
            inside=available&mask;outside=available&~mask
            clauses.append({'case':case,'round':t,'id':h['id'],'coordinate':h['coordinate'],
                            'bounds':h['bounds'],'categories':h['categories'],'retained':h['id'] in decision['retained'],
                            'inside_unqueried_count':int(inside.sum()),'outside_unqueried_count':int(outside.sum()),
                            'offline_mean_yield_inside_percent':float(oracle[inside].mean()*100) if inside.any() else None,
                            'offline_mean_yield_outside_percent':float(oracle[outside].mean()*100) if outside.any() else None,
                            'offline_residual_contrast_pp':float((residual[inside].mean()-residual[outside].mean())*100) if inside.any() and outside.any() else None})
        for target_id in TARGETS[case]:
            k=ix[target_id];is_available=bool(available[k]);is_eligible=bool(eligible[k])
            violated=[h['id'] for h,mask in zip(knowledge['clauses'],masks) if not mask[k]]
            targets.append({'case':case,'round':t,'target_id':target_id,'catalyst':pool['points'][k]['catalyst'],
                'offline_true_yield_percent':float(oracle[k]*100),'available':is_available,'eligible':is_eligible,
                'base_logEI_loss':float(base[decision['reference']]-base[k]),'budget':decision['budget'],
                'base_rank_available':int(1+sum(available&((base>base[k])|((base==base[k])&(np.asarray(ids)<target_id))))) if is_available else None,
                'guided_rank_eligible':int(1+sum(eligible&((scores>scores[k])|((scores==scores[k])&(np.asarray(ids)<target_id))))) if is_eligible else None,
                'GP_mean_percent':float(stats['mu_y'][k]*100),'GP_latent_SD_pp':float(stats['sd_y'][k]*100),
                'raw_region_shift_pp':float(decision['raw_shift_std'][k]*ys*100),
                'deployed_shift_pp':float(decision['shift_std'][k]*ys*100),
                'violated_current_clauses':';'.join(violated),'violated_retained_clauses':';'.join(v for v in violated if v in decision['retained'])})
        rounds.append({'case':case,'added_round':t,'total_query':t+4,'candidate_id':ids[j],
            'catalyst':pool['points'][j]['catalyst'],'yield_percent':float(oracle[j]*100),
            'new_incumbent_percent':float(sc[t+3]),'baseline_id':baseline['history'][t+3]['candidate_id'],
            'baseline_yield_percent':by[t+3],'baseline_incumbent_percent':float(bc[t+3]),
            'baseline_minus_new_incumbent_pp':float(bc[t+3]-sc[t+3]),'baseline_minus_new_AUC_contribution_pp':float((bc[t+3]-sc[t+3])/16),
            'available_count':int(available.sum()),'eligible_count':int(eligible.sum()),
            'alpha':decision['amplitude'],'bank_updates_before':decision['bank']['updates'] if decision['bank'] else None,
            'retained_ids':';'.join(decision['retained']),'B4_revoked':';'.join(decision['diagnosis'].revoked_ids),
            'GP_mu_percent':float(stats['mu_y'][j]*100),'raw_shift_pp':float(decision['raw_shift_std'][j]*ys*100),
            'deployed_shift_pp':float(decision['shift_std'][j]*ys*100),
            'innovation_z':event['measurement']['innovation_z'],'new_alpha_after':float(np.clip(event['bank_after']['mean'][1],0,1)) if event['bank_after'] else None,
            'refresh':bool(call)})
        if event['bank_after'] is not None:state['banks'][event['bank_key']]=deepcopy(event['bank_after'])
        state['history'].append(actual['history'][t+3]);state['decisions'].append(event)
        if call:state['calls'].append(call)
    assert state['history']==actual['history'] and state['banks']==actual['banks']
    assert cache.counters['fits']==0,cache.counters
    result={'case':case,'rounds':rounds,'variants':variants,'targets':targets,'clauses':clauses,'refresh':refresh,
            'cache':cache.counters,'operator_compiles_in_loop':sum(s.prior.previous.JIT.values())}
    write(OUT/f'case{case}.json',result)
    return {'case':case,'rounds':len(rounds),'alternatives':len(variants),'GP_fits':cache.counters['fits']}


def main():
    if hasattr(__import__('sys').stdout,'reconfigure'):__import__('sys').stdout.reconfigure(encoding='utf8')
    s.check_protocol();OUT.mkdir(parents=True,exist_ok=True)
    sources=[]
    for case in CASES:
        sources.extend(s.ROOT/'runs'/f'c{case}_s0_{REGIME}_{m}'/'state.json' for m in METHODS)
        sources.append(s.prior.previous.ROOT/'base'/f'c{case}_s0'/'state.json' if case==3 else s.prior.ROOT/'runs'/f'c{case}_s0_{REGIME}_LLINBO_matched'/'state.json')
        sources.extend(s.prior.original.ROOT/'inputs'/f'case{case}'/n for n in ('pool.json','oracle.npz'))
    frozen={p.as_posix():filehash(p) for p in sources}
    write(OUT/'analysis_protocol.json',{'source_sha256':frozen,'script_sha256':filehash(__file__),
        'cases':CASES,'regime':REGIME,'design':'32 actual-prefix replays, one-step deterministic module removals; independent full-ladder comparisons. Never sum local effects into trajectory AUC.',
        'outcome_boundary':'Oracle used after ranking and for disclosed retrospective region-quality/target diagnostics only; no knowledge/policy tuning or new trajectories.'})
    with ProcessPoolExecutor(max_workers=2,initializer=s.worker_init) as executor:
        summaries=list(executor.map(analyze_case,CASES))
    data=[read(OUT/f'case{c}.json') for c in CASES]
    for key in ('rounds','variants','targets','clauses'):csvwrite(OUT/(key+'.csv'),[r for d in data for r in d[key]])
    for p,h in frozen.items():assert filehash(p)==h
    s.check_protocol()
    write(OUT/'run_receipt.json',{'status':'PASS','summaries':summaries,'source_files_unchanged':len(frozen),
        'new_GP_fits':sum(d['cache']['fits'] for d in data),'new_LLM_calls':0,
        'operator_compiles_in_loop':sum(d['operator_compiles_in_loop'] for d in data)})
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()
