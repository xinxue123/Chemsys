"""Small, versioned full-trajectory R2 tuning; frozen comparisons stay read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import re
import time
import numpy as np

from . import r2_region_trust_study as trust
from . import r2_region_trust_engine as engine
from . import r2_tuning_policy_v3 as policy
from . import pointwise_hte_llm as llm
from . import reasoning_bo_data as data
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=Path('data/runs/hte_r2_tuning_20261028')
OUT=Path('research/hte_r2_tuning_20261028')
TARGETS={'Perera_Suzuki':{'AUC_floor':98.58875,'best_floor':99.05,'stretch_AUC':98.8325},
    'Doyle_Buchwald_Hartwig':{'stage_AUC':87.457094395,'stage_best':95.,
        'AUC_floor':91.4567435325,'best_floor':98.28722602}}


def folders(variant):
    if not re.fullmatch('[a-z][a-z0-9_]+',variant):raise ValueError('invalid variant ID')
    return ROOT/variant,OUT/variant


def config(variant):
    c=data.read(OUT/'plans'/f'{variant}.json');assert c['id']==variant
    assert c['normal_policy']=='R2_region_trust'
    assert len(set(c['validation_rounds']))==len(c['validation_rounds'])<=3
    assert all(type(t)==int and 1<=t<=16 for t in c['validation_rounds'])
    return c


def sources(variant):
    paths=[Path(__file__),Path(policy.__file__),Path(policy.base.__file__),Path(policy.base.base.__file__),Path('chem/benchmarks/r2_tuning_report_v3.py'),
        Path('tests/test_r2_tuning_v3.py'),Path('pointwise_r2_tuning_v3.py'),OUT/'TASK.md',OUT/'plans'/f'{variant}.json']
    return {str(p.as_posix()):data.filehash(p) for p in paths}


def preservation():
    result=[]
    for folder in (trust.parent.OUT,trust.OUT,*sorted(p for p in OUT.iterdir() if p.is_dir() and (p/'release_inventory.json').exists())):
        inv=data.read(folder/'release_inventory.json');receipt=data.read(folder/'release_receipt.json')
        assert data.filehash(folder/'release_inventory.json')==receipt['inventory_sha256']
        for row in inv['files']:assert data.filehash(row['path'])==row['sha256'],row['path']
        result.append({'source':str(folder),'files':len(inv['files']),'inventory_sha256':receipt['inventory_sha256']})
    return result


def prepare(variant,secret_config='../config.txt'):
    root,out=folders(variant)
    if (out/'protocol.json').exists():return check(variant)
    trust.check();c=config(variant);kept=preservation()
    p={'study':'R2_tuning','variant':variant,'config':c,'targets':TARGETS,'source_sha256':sources(variant),
        'seed':0,'initial_queries':4,'additional_queries':16,'trajectories':2,'q':1,
        'candidate_data':'same full3696/3955 author measured tables; no oracle-based filtering',
        'GP':'unchanged all-category Mixed GP, qLogEI512; same fit/MC seeds and jitter',
        'normal_policy':'same-region trust corrected R2, unchanged KF and B4',
        'gate_exception':'ONLY declared controlled-validation rounds may exceed original logEI-loss gate; same16 total queries',
        'initialization':'reuse exact original common4 and common knowledge',
        'LLM':api.load_config(secret_config).public(),'max_regional_refreshes':3,
        'max_controlled_validation_requests':len(c['validation_rounds']),
        'interpretation':'Development tuning on previously inspected seed0, no independent generalization or significance claim.',
        'protected_controls':kept,'CPU_workers':2,'Torch_threads_per_worker':1,'operator_compiles_in_loop':0,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    for task in data.SPECS:
        data.write(root/'initialization'/task/'ready.json',data.read(trust.parent.ROOT/'initialization'/task/'ready.json'))
    p['protocol_sha256']=digest(p);data.write(out/'protocol.json',p);return p


def check(variant):
    trust.check();_,out=folders(variant);p=data.read(out/'protocol.json');q=deepcopy(p);h=q.pop('protocol_sha256')
    assert digest(q)==h and p['source_sha256']==sources(variant)
    assert p['config']==config(variant) and p['targets']==TARGETS
    return p


def initial(task,variant,p):
    root,_=folders(variant);ready=data.read(root/'initialization'/task/'ready.json')
    return {'task':task,'seed':0,'method':'New_R2','display_method':variant,'variant':variant,
        'run_id':task+'_s0_'+variant,'protocol_sha256':p['protocol_sha256'],
        'history':deepcopy(ready['history']),'knowledge':deepcopy(ready['knowledge']),
        'initial_knowledge':deepcopy(ready['knowledge']),'banks':{},'region_trust':{},'calls':[],
        'validations':[],'decisions':[],'status':'READY','seconds':0.}


def request_for(variant):
    root,_=folders(variant)
    def request(client,public,state,system,validator,phase):
        public=deepcopy(public)
        if phase=='refresh' and state['validations']:
            public['controlled_validation_results']=deepcopy(state['validations'])
            public['comparison_scope']='A one-axis comparison applies to its fixed other conditions; it is not proof of a global axis preference.'
        context={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
            'phase':phase,'seed':0,'variant':variant,'protocol_sha256':state['protocol_sha256']}
        context['context_sha256']=digest(context);folder=root/'requests'/context['context_sha256'];data.write(folder/'context.json',context)
        if (folder/'validated.json').exists():
            saved=data.read(folder/'validated.json');assert saved['plan']==validator(saved['raw']);return saved,context['context_sha256']
        if phase in ('refresh','controlled_validation'):
            for oldroot in tuple(ROOT/v for v in config(variant).get('reuse_variants',[]))+(trust.ROOT,trust.parent.ROOT):
                for path in sorted((oldroot/'requests').glob('*/context.json')):
                    old=data.read(path)
                    if old['task']!=state['task'] or old['method']!='New_R2' or old['phase']!=phase:continue
                    if old['observed_prefix']!=state['history'] or not trust.equivalent(old['public'],public):continue
                    source=path.parent/'validated.json'
                    if not source.exists():continue
                    saved=data.read(source);assert saved['plan']==validator(saved['raw'])
                    saved={**saved,'context_sha256':context['context_sha256']}
                    data.write(folder/'reused.json',{'source':str(source),'source_sha256':data.filehash(source)})
                    data.write(folder/'validated.json',saved);return saved,context['context_sha256']
        saved=llm.nominal.get(client,context,folder,system,validator);return saved,context['context_sha256']
    return request


def validation_call(client,request,pool,state,d,t):
    candidates=policy.build_candidates(pool,state,d)
    if not candidates['candidates']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B',
        'reaction_class':pool['reaction_class'],'objective_unit':pool['objective_unit'],'domain':llm.nominal.domain(pool),
        'observations':llm.nominal.observations(pool,state,include_gp=False),'observation_count':len(state['history']),
        'query_round':t,'remaining_queries':17-t,'current_knowledge':llm.nominal.public_knowledge(pool,state['knowledge']),
        'counterexample_pool':candidates,'normal_BO_reference_id':pool['ids'][d['reference']],
        'normal_R2_proposal_id':pool['ids'][d['selected']],'normal_gate_budget':d['budget'],
        'previous_controlled_comparisons':deepcopy(state['validations'])}
    answer,key=request(client,public,state,policy.SYSTEM,lambda raw:policy.validate(raw,candidates,state['history']),'controlled_validation')
    return {'prepared':candidates,'validated':answer,'context_sha256':key}


def run_one(task,variant,secret_config='../config.txt'):
    trust.parent.warm();p=check(variant);root,_=folders(variant);pool=data.load_pool(task);state=initial(task,variant,p)
    folder=root/'runs'/state['run_id'];path=folder/'state.json'
    if path.exists():state=data.read(path)
    if state['status']=='COMPLETE':return {'task':task,'variant':variant,'status':'COMPLETE','queries':16}
    state.pop('error',None);cache=engine.Cache(pool,root/'cache'/task)
    cache.sources=[trust.ROOT/'cache'/task,trust.parent.ROOT/'cache'/task]
    client=api.APIClient(api.load_config(secret_config));start=time.perf_counter();request=request_for(variant)
    lookup={k:i for i,k in enumerate(pool['ids'])}
    try:
        while len(state['decisions'])<16:
            t=len(state['decisions'])+1;d=engine.setup(cache,state,t);reasons=engine.event_reasons(state,d,t)
            if reasons:
                with patch.object(llm,'request',request):answer,key=llm.regional_call(client,pool,state,d,reasons)
                state['calls'].append({'iteration':t,'reasons':reasons,'context_sha256':key,
                    'before':trust.parent.decision_receipt(d),'validated':answer})
                state['knowledge']=answer['plan']['knowledge'];data.write(path,state);d=engine.setup(cache,state,t)
            normal_selected=pool['ids'][d['selected']];validation=None
            if t in p['config']['validation_rounds']:
                validation=validation_call(client,request,pool,state,d,t)
                if validation:d['selected']=lookup[validation['validated']['plan']['candidate_id']]
            mode='controlled_validation' if validation else 'normal_R2';j=d['selected'];key=pool['ids'][j]
            prefix=deepcopy(state['history']);best=max(h['y'] for h in prefix)
            assert d['available'][j]
            if mode=='normal_R2':assert d['eligible'][j]
            table=folder/'rounds'/f'{t:02d}_candidates.npz';table.parent.mkdir(parents=True,exist_ok=True)
            with table.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
                    raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],**d['stats'])
            table.with_suffix('.tmp').replace(table);target=folder/'rounds'/f'{t:02d}_committed.json'
            data.write(target,{'task':task,'variant':variant,'iteration':t,'execution_mode':mode,'observation_count':len(prefix),
                'selected_id':key,'normal_R2_proposal_id':normal_selected,'history_sha256':digest(prefix),
                'decision':trust.parent.decision_receipt(d),'region_trust_at_commit':deepcopy(state['region_trust']),
                'validation':validation,'candidate_table_sha256':data.filehash(table),'protocol_sha256':p['protocol_sha256']})
            observed=data.reveal(task,[key])[0];measurement=bank=trust_feedback=None
            if d['bank_key']:
                _,v=cache.model(prefix);bank,measurement=engine.original.nominal.update_bank(d['bank'],d['prior'],t,
                    float(d['raw_shift_std'][j])*v['std'],float(d['stats']['mu_y'][j]),float(d['stats']['sd_observation_y'][j]),observed['y'])
                state['banks'][d['bank_key']]=bank;d['_measurement']=measurement
                trust_feedback=engine.feedback(state,d,t,observed['y'],bank)
            controlled=None
            if validation:
                plan=validation['validated']['plan'];anchor=prefix[plan['anchor_observation_index']-1]
                controlled={'query':t,'anchor_observation_index':plan['anchor_observation_index'],
                    'anchor_candidate_id':anchor['candidate_id'],'anchor_yield_percent':round(anchor['y']*100,3),
                    'tested_candidate_id':key,'tested_yield_percent':round(observed['y']*100,3),
                    'observed_difference_pp':round((observed['y']-anchor['y'])*100,3),
                    'changed_axis':plan['changed_axis'],'tested_clause_id':plan['clause_id'],
                    'prequery_prediction':plan['prediction'],'scope':'Same other axes; one comparison, no global causal claim.'}
                state['validations'].append(controlled)
            loss=float(d['base'][d['reference']]-d['base'][j]);exception=bool(not d['eligible'][j])
            if exception:assert validation and p['config']['allow_acquisition_gate_exception_for_validation']
            event={'iteration':t,'selected_id':key,'y':observed['y'],'execution_mode':mode,'normal_R2_proposal_id':normal_selected,
                'incumbent_before':best,'incumbent_after':max(best,observed['y']),'incumbent_gain':max(0.,observed['y']-best),
                'reference_id':pool['ids'][d['reference']],'base_qLogEI':float(d['base'][j]),'guided_qLogEI':float(d['score'][j]),
                'base_opportunity_loss':loss,'budget':d['budget'],'acquisition_gate_exception':exception,
                'eligible_count':int(d['eligible'].sum()),'amplitude':d['amplitude'],'fallback':d['fallback'],
                'retained':d['retained'],'diagnosis':d['diagnosis'].to_dict(),'measurement':measurement,
                'calibration_after':deepcopy(bank),'knowledge':deepcopy(state['knowledge']),'trust':deepcopy(d['trust']),
                'trust_feedback':trust_feedback,'region_trust_after_feedback':deepcopy(state['region_trust']),
                'controlled_comparison':controlled,'validation_context_sha256':validation['context_sha256'] if validation else None,
                'commit_sha256':data.filehash(target)}
            state['history'].append(observed);state['decisions'].append(event);state['status']='COMPLETE' if t==16 else 'RUNNING'
            data.write(path,state);cache.release()
            print(json.dumps({'task':task,'variant':variant,'query':t,'mode':mode,'observed_percent':observed['y']*100,
                'best_percent':max(best,observed['y'])*100,'gate_exception':exception}),flush=True)
    except Exception as error:
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
    finally:
        state['seconds']+=time.perf_counter()-start;data.write(path,state);client.close()
    return {'task':task,'variant':variant,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}


def states(variant):
    root,_=folders(variant);return [data.read(p) for p in (root/'runs').glob('*/state.json')]


def run(variant,secret_config='../config.txt'):
    check(variant)
    with ProcessPoolExecutor(max_workers=2,initializer=trust.parent.warm) as ex:
        for f in as_completed([ex.submit(run_one,t,variant,secret_config) for t in data.SPECS]):print(json.dumps(f.result()),flush=True)
    return {'complete':sum(s['status']=='COMPLETE' for s in states(variant)),'planned':2,'variant':variant}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','run','status','validate','report','seal'])
    ap.add_argument('--variant',default='focused_countercheck_v3');ap.add_argument('--config',default='../config.txt');a=ap.parse_args()
    if a.action=='prepare':p=prepare(a.variant,a.config);result={'protocol_sha256':p['protocol_sha256'],'variant':a.variant}
    elif a.action=='run':result=run(a.variant,a.config)
    elif a.action=='status':result=[{'task':s['task'],'status':s['status'],'queries':len(s['decisions']),'error':s.get('error')} for s in states(a.variant)]
    else:
        from . import r2_tuning_report_v3 as report
        result={'validate':report.validate,'report':report.build,'seal':report.seal}[a.action](a.variant)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
