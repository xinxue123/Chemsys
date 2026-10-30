"""Matched Mixed-GP controls, contextual old references and independent audits."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import torch

from . import mixed_kalman_suzuki_study as s
from . import mixed_kalman_suzuki_engine as e
from . import kalman_knowledge_suzuki_report as previous
from . import agent_score_suzuki_report as metrics
from .suzuki_domain import digest


def rows():
    groups={m:[s.read(s.ROOT/folder/f'c{c}_s0'/'state.json') for c in (1,2,3,4)] for m,folder in zip(s.METHODS,('runs','frozen','base'))}
    for method,group in previous.rows().items():groups['Previous_'+method]=group
    return groups


def summary(group):
    values=[metrics.metric(r) for r in group]
    return {'n':len(group),'completed':sum(r['status']=='COMPLETE' for r in group),'SearchAUC_mean':st.mean(v['AUC'] for v in values),
            'final_yield_mean':st.mean(v['final'] for v in values),'mean_curve':[st.mean(v['curve'][i] for v in values) for i in range(16)]}


def usage():
    answers=[s.read(p) for p in (s.ROOT/'requests').glob('*/attempt*/response.json')]
    return {'logical_calls':sum(len(s.read(p)['calls']) for p in (s.ROOT/'runs').glob('*/state.json')),'received_API_responses':len(answers),
            'HTTP_errors':len(list((s.ROOT/'requests').glob('*/attempt*/transport_error.json'))),
            'schema_failures':len(list((s.ROOT/'requests').glob('*/attempt*/validation_error.json'))),
            'returned_models':sorted({r['model'] for r in answers}),'fingerprints':sorted({r['system_fingerprint'] for r in answers if r.get('system_fingerprint')}),
            **{key:sum(r['usage'].get(key,0) for r in answers) for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'reasoning_tokens_included_in_completion':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in answers)}


def mechanism(group):
    events=[v for row in group for v in row['decisions']]
    if not events:return {'observed_added_queries':0,'failed_trajectories':sum(r['status']=='FAILED' for r in group)}
    result=previous.mechanisms(group)
    result['singleton_budget_rounds']=sum(v['eligible_candidates']==1 for v in events)
    result['eligible_count_by_case']={str(r['case']):[v['eligible_candidates'] for v in r['decisions']] for r in group}
    result['call_rounds_by_case']={str(r['case']):[c['iteration'] for c in r['calls']] for r in group}
    return result


def read_scores(cache,key):
    with np.load(cache.directory/(key+'.npz'),allow_pickle=False) as data:return data['scores'].copy(),json.loads(str(data['metadata']))


def permutation_posterior(cache,prefix,model,value):
    permutation=(19,-3,101,7,0,12,-9,4);xp=torch.tensor(e.encode_mixed(cache.pool,permutation),dtype=torch.double)
    lookup={k:j for j,k in enumerate(cache.ids)};ordered=sorted(prefix,key=lambda h:h['candidate_id'])
    y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double);ystd=(y-y.mean(0,keepdim=True))/y.std(0,unbiased=False,keepdim=True).clamp_min(1e-12)
    relabeled=e.MixedSingleTaskGP(xp[[lookup[h['candidate_id']] for h in ordered]],ystd,cat_dims=[3]);relabeled.load_state_dict(model.state_dict());relabeled.eval()
    means=[];variances=[]
    with torch.no_grad():
        for batch in xp.split(512):
            posterior=relabeled.posterior(batch.unsqueeze(1));means.append(posterior.mean.reshape(-1));variances.append(posterior.variance.reshape(-1))
    return torch.cat(means).numpy(),torch.cat(variances).clamp_min(1e-18).sqrt().numpy()


def audit_case(case):
    p=s.policy;pool=s.inputs(case);ids=pool['ids'];z=np.array(pool['z']);ix={k:j for j,k in enumerate(ids)}
    cache=e.MixedKnowledgeCache(pool,s.ROOT/'audit_cache'/f'c{case}_s0',case,sources=[s.ROOT/'cache'/f'c{case}_s0'])
    checked=0;score_error=0.;filter_error=0.;permutation_error=0.;scores_checked=0;nominal_prefixes=0
    for folder in ('runs','frozen','base'):
        row=s.read(s.ROOT/folder/f'c{case}_s0'/'state.json');state=p.initial_filter();hypotheses=[];enabled=False
        if folder=='frozen' and row.get('frozen_initial_plan'):
            state,hypotheses=p.apply_plan(state,[],row['frozen_initial_plan'],z);enabled=True
        for t,event in enumerate(row['decisions'],1):
            prefix=row['history'][:t+3];queried=np.array([key in {h['candidate_id'] for h in prefix} for key in ids]);available=~queried
            if folder=='runs':
                call=next((c for c in row['calls'] if c['iteration']==t),None)
                if call:
                    if call['status']=='ACCEPTED':state,hypotheses=p.apply_plan(state,hypotheses,call['plan'],z);enabled=True
                    else:enabled=False
            assert state==event['filter_before'] and hypotheses==event['hypotheses']
            prior=p.forecast(state,hypotheses);assert prior==event['filter_prior']
            weights=prior['weights'] if enabled else {h['id']:0. for h in hypotheses};assert weights==event['trust']
            # Read/copy only new Mixed caches. No old ordinal states are reused.
            base,bmeta=cache.score(prefix);guide,gmeta=cache.knowledge_score(prefix,hypotheses,weights,event['diagnosis']['retained_ids']) if event['diagnosis']['retained_ids'] else (base,bmeta)
            assert bmeta['key']==event['base_score_key'] and gmeta['key']==event['guided_score_key']
            assert bmeta['gp_class']=='MixedSingleTaskGP' and bmeta['cat_dims']==[3]
            ref=p.common.argmax(base,available,ids);budget=p.common.budget_at(4,t);eligible=available&((base[ref]-base)<=budget)
            diagnosis=p.diagnose(base,z,hypotheses,weights,budget,queried,ids)
            assert json.loads(json.dumps(diagnosis.to_dict()))==event['diagnosis']
            selected=p.common.argmax(guide,eligible,ids) if diagnosis.retained_ids else ref;j=ix[event['selected_id']]
            assert selected==j and ids[ref]==event['reference_id'] and event['base_logei_loss']==float(base[ref]-base[j])<=budget
            stats=cache.posterior_stats(prefix);fit,value=cache.model(prefix);assert fit==event['fit_key']
            saved=torch.load(cache.directory/(fit+'.pt'),map_location='cpu',weights_only=False)
            expected_train=e.encode_mixed(pool)[[ix[h['candidate_id']] for h in sorted(prefix,key=lambda h:h['candidate_id'])]]
            assert np.array_equal(saved['train_X'],expected_train) and saved['cat_dims']==[3]
            for module in value['model'].covar_module.modules():
                if isinstance(module,torch.nn.Module) and type(module).__name__=='RBFKernel':assert module.active_dims.tolist()==[0,1,2]
            raw_change=np.zeros(len(ids));feature=np.zeros(5);feature[0]=1.
            for h in hypotheses:
                if h['variable']=='catalyst':m=np.isin([s.old.CATALYSTS.index(point['catalyst']) for point in pool['points']],h['categories'])
                else:
                    k=p.VARIABLES.index(h['variable']);m=(z[:,k]>=h['bounds'][0])&(z[:,k]<=h['bounds'][1])
                assert float(m.mean())==h['pool_fraction'];f=h['effect_pp']/100*(m.astype(float)-m.mean());feature[p.SLOTS.index(h['id'])+1]=f[j]
                if h['id'] in diagnosis.retained_ids:raw_change+=weights[h['id']]*f
            delta=np.clip(raw_change,-.5*stats['sd_y'],.5*stats['sd_y'])
            assert abs(delta[j]-event['mean_change_y'])<1e-12 and abs(float(stats['mu_y'][j])-event['selected_gp_mu_y'])<1e-12
            replay_base=previous.marginal_scores(stats['mu_std'],stats['sd_std'],np.zeros(len(ids)),value['best_std'],bmeta['mc_seed'],cache.z)
            replay_guide=previous.marginal_scores(stats['mu_std'],stats['sd_std'],delta/float(value['std']),value['best_std'],bmeta['mc_seed'],cache.z)
            error=max(float(np.max(np.abs(base-replay_base))),float(np.max(np.abs(guide-replay_guide))));score_error=max(score_error,error);assert error<1e-8
            assert p.common.argmax(replay_base,available,ids)==ref
            assert (p.common.argmax(replay_guide,eligible,ids) if diagnosis.retained_ids else ref)==j
            pm,ps=permutation_posterior(cache,prefix,value['model'],value);pe=max(float(np.max(np.abs(pm-stats['mu_std']))),float(np.max(np.abs(ps-stats['sd_std']))))
            permutation_error=max(permutation_error,pe);assert pe<1e-9;nominal_prefixes+=1
            m=event['measurement'];y=row['history'][t+3]['y'];theta,cov,nm,nv,innovation,variance=previous.independent_filter_update(state,prior,feature,stats['mu_y'][j],stats['sd_predictive_y'][j],y)
            after=event['filter_after'];fe=max(float(np.max(np.abs(theta-after['theta']))),float(np.max(np.abs(cov-after['covariance']))),abs(nm-after['null_mean']),abs(nv-after['null_variance']))
            filter_error=max(filter_error,fe);assert fe<1e-12 and np.linalg.eigvalsh(cov).min()>-1e-12
            assert np.allclose(feature,m['feature'],rtol=0,atol=1e-14) and abs(innovation-m['innovation'])<1e-12 and abs(variance-m['innovation_variance'])<1e-12
            predictions={'GP':float(stats['mu_y'][j]),'GP_offset_null':float(stats['mu_y'][j]+prior['null_mean']),
                         'GP_joint_KF':float(stats['mu_y'][j]+feature@np.array(prior['theta'])),'GP_deployed':float(stats['mu_y'][j]+delta[j])}
            for name,pred in predictions.items():assert abs(pred-m['predictions'][name])<1e-12 and abs((y-pred)**2-m['squared_errors'][name])<1e-12
            state=deepcopy(after);checked+=1;scores_checked+=2*len(ids)
        assert state==row['filter']
    return {'case':case,'choices_and_KF_updates':checked,'independently_recomputed_qLogEI_scores':scores_checked,
            'maximum_independent_qLogEI_error':score_error,'maximum_independent_filter_error':filter_error,
            'category_permutation_prefixes':nominal_prefixes,'maximum_category_permutation_error':permutation_error,
            'audit_cache_counters':cache.counters,'operator_compiles_in_loop':sum(s.JIT.values())}


def report():
    p=s.check_protocol();groups=rows();assert all(len(v)==4 and all(r['status'] in ('COMPLETE','FAILED') for r in v) for v in groups.values())
    tables={};paired={};mechanisms={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        subset={m:[r for r in group if r['case'] in cases] for m,group in groups.items()}
        tables[scope]={m:summary(v) for m,v in subset.items()};paired[scope]={m:metrics.compare(subset[s.METHODS[0]],subset[m]) for m in subset if m!=s.METHODS[0]}
        mechanisms[scope]={m:mechanism(subset[m]) for m in s.METHODS}
    with ProcessPoolExecutor(max_workers=2,initializer=s.worker_init) as executor:audits=list(executor.map(audit_case,(1,2,3,4)))
    new=[r for m in s.METHODS for r in groups[m]];runtime=[s.read(f) for f in (s.ROOT/'runtime').glob('*.json')]
    result={'protocol_sha256':p['protocol_sha256'],'planned_trajectories':12,'completed_trajectories':sum(r['status']=='COMPLETE' for r in new),
            'planned_logical_queries':240,'observed_logical_queries':sum(len(r['history']) for r in new),'tables':tables,'new_minus_comparators':paired,
            'mechanism':mechanisms,'usage':usage(),'runtime':{'new_formal_GP_fits':sum(r['cache']['fits'] for r in runtime),'operator_compiles_in_loop':sum(r['operator_compiles_in_loop'] for r in runtime)},
            'pure_BO_model_change':{scope:metrics.compare([r for r in groups[s.METHODS[2]] if r['case'] in cases],[r for r in groups['Previous_qLogEI_B0'] if r['case'] in cases]) for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4))]},
            'category_permutation':{'verified_prefixes':sum(a['category_permutation_prefixes'] for a in audits),'maximum_posterior_difference':max(a['maximum_category_permutation_error'] for a in audits)},
            'raw_metrics':[{'method':m,'case':r['case'],'seed':0}|metrics.metric(r) for m,group in groups.items() for r in group],
            'limits':p['limits']+['Pure BO model replacement is the matched no-LLM contrast. Mixed framework vs Mixed B0 estimates full knowledge-policy increment on the same GP; old regional/LLINBO references use older knowledge and a different GP.']}
    s.write(s.OUT/'audit_cases.json',audits);s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['scope','method','n','SearchAUC_pp','final_yield_percent'])
        for scope,table in tables.items():
            for m,v in table.items():writer.writerow([scope,m,v['n'],v['SearchAUC_mean'],v['final_yield_mean']])
    figures(result)
    chosen=(*s.METHODS,'Previous_KF_event_B4','Previous_qLogEI_B0')
    return {'complete':result['completed_trajectories'],'primary':{m:{k:round(tables['primary'][m][k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m in chosen},
            'usage':result['usage'],'category_permutation':result['category_permutation'],'runtime':result['runtime']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=(*s.METHODS,'Previous_KF_event_B4','Previous_qLogEI_B0','Previous_LGBO_region_natural_reference')
    labels=('Mixed GP + KF/event/B4','Mixed GP + frozen knowledge','Mixed GP BO','Previous ordinal GP + KF/event/B4','Previous ordinal GP BO','Old regional LGBO reference')
    colors=('#0072B2','#777777','#D55E00','#CC79A7','#000000','#009E73')
    fig,axes=plt.subplots(1,4,figsize=(15,4.1),layout='constrained')
    for c,ax in enumerate(axes,1):
        for m,label,color in zip(chosen,labels,colors):ax.plot(range(5,21),result['tables'][f'Case{c}'][m]['mean_curve'],label=label,color=color,lw=1.7,ls='--' if m.startswith('Previous_') else '-')
        ax.set(title=f'Case {c}, seed 0',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Incumbent yield (%)');axes[-1].legend(fontsize=6.3,loc='lower right')
    fig.suptitle('Nominal Mixed GP replacement: same pool / initialization / 4+16 queries; single seed',fontsize=10.5)
    fig.savefig(s.OUT/'comparison.png',dpi=180);fig.savefig(s.OUT/'comparison.pdf');plt.close(fig)


def verify(config_path='../config.txt'):
    p=s.check_protocol();groups=rows();freeze=s.read(s.OUT/'adaptive_frozen.json');assert freeze['control_states_before_start']==0
    assert s.filehash(s.OUT/'qualification.json')==p['qualification_sha256']
    for path,value in p['input_sha256'].items():assert s.filehash(s.PARENT/'inputs'/path)==value
    for path,value in p['model']['implementation_sha256'].items():assert s.filehash(path)==value
    contexts=0;attempts=0
    for case in (1,2,3,4):
        pool=s.inputs(case);ids=pool['ids'];z=np.array(pool['z']);lookup={k:j for j,k in enumerate(ids)};oracle=np.load(s.PARENT/'inputs'/f'case{case}'/'oracle.npz')['values']
        new=[groups[m][case-1] for m in s.METHODS];adaptive=new[0]
        assert s.filehash(s.ROOT/'runs'/adaptive['run_id']/'state.json')==freeze['states'][adaptive['run_id']]
        for row in new:
            assert row['status']=='COMPLETE' and len(row['history'])==20 and len(row['decisions'])==16 and row['operator_compiles_in_loop']==0
            assert row['protocol_sha256']==p['protocol_sha256'] and len({h['candidate_id'] for h in row['history']})==20
            assert [h['candidate_id'] for h in row['history'][:4]]==pool['initial_ids']['0']
            for h in row['history']:assert h['y']==float(oracle[lookup[h['candidate_id']]])
        assert all(row['history'][:4]==new[0]['history'][:4] for row in new)
        assert not new[1]['calls'] and not new[2]['calls'] and len(adaptive['calls'])<=4
        if adaptive['calls'][0]['status']=='ACCEPTED':
            assert new[1]['frozen_initial_plan']==adaptive['calls'][0]['plan'] and new[1]['decisions'][0]['selected_id']==adaptive['decisions'][0]['selected_id']
        cache=s.make_cache(case,pool);call_rounds=[]
        for call in adaptive['calls']:
            t=call['iteration'];event=adaptive['decisions'][t-1];directory=s.ROOT/'requests'/call['context_sha256'];context=s.read(directory/'context.json')
            raw=deepcopy(context);key=raw.pop('context_sha256');assert digest(raw)==key==call['context_sha256'] and context['history']==adaptive['history'][:t+3]
            prefix=context['history'];base,_=cache.score(prefix);stats=cache.posterior_stats(prefix);prior=s.policy.forecast(event['filter_before_call'],event['hypotheses_before_call'])
            queried=np.array([key in {h['candidate_id'] for h in prefix} for key in ids])
            diag=s.policy.diagnose(base,z,event['hypotheses_before_call'],prior['weights'] if event['enabled_before_call'] else {},s.policy.common.budget_at(4,t),queried,ids)
            reasons=s.policy.event_reasons(t,prefix,adaptive['decisions'][:t-1],event['hypotheses_before_call'],diag,call_rounds)
            assert reasons==call['reasons'];call_rounds.append(t)
            minimal={'task':adaptive['task'],'history':prefix,'decisions':adaptive['decisions'][:t-1],'hypotheses':event['hypotheses_before_call'],'filter':event['filter_before_call']}
            public=s.old.public_request(minimal,pool,stats,base,diag,reasons);assert json.loads(json.dumps(public))==context['public']
            assert len(public['observations'])==t+3 and all('yield_percent' not in r for r in public['gp_top3_per_catalyst'])
            if call['status']=='ACCEPTED':
                valid=s.read(directory/'validated.json');n=valid['accepted_attempt'];assert n<=8 and valid['received_responses']<=3
                response_path=directory/f'attempt{n:02d}'/'response.json';response=s.read(response_path)
                assert s.filehash(response_path)==valid['response_sha256'] and not response['retained_raw_reasoning'] and response['model']=='deepseek-flash'
                assert s.policy.validate_plan(json.loads(response['choices'][0]['message']['content']),event['hypotheses_before_call'],len(prefix),z)==call['plan']==valid['plan']
                for i in range(1,n+1):
                    body=s.read(directory/f'attempt{i:02d}'/'request.json');assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
                    assert body['messages'][:2]==s.old.messages_for(public);attempts+=1
            contexts+=1
    for path,value in freeze['requests'].items():assert s.filehash(path)==value
    protected={};trees={}
    for label,path in s.PROTECTED.items():
        inv=s.read(path)
        for f in inv['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inv['files'])
        for key,expected in inv.items():
            if key.endswith('cache_tree') or key=='score_tree':
                folder='scores' if key=='score_tree' else key.removesuffix('_tree');actual=metrics.tree(Path('data/runs')/f'suzuki_{label}'/folder)
                assert actual=={k:expected[k] for k in actual};trees[label+'_'+folder]=actual
    result=s.read(s.OUT/'results.json');audits=s.read(s.OUT/'audit_cases.json');assert sum(a['choices_and_KF_updates'] for a in audits)==192
    assert sum(a['category_permutation_prefixes'] for a in audits)==192 and all(a['operator_compiles_in_loop']==0 for a in audits)
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4))]:
        for m,group in groups.items():
            selected=[r for r in group if r['case'] in cases];auc=st.mean(sum(max(h['y'] for h in r['history'][:min(n,len(r['history']))])*100 for n in range(5,21))/16 for r in selected)
            assert abs(auc-result['tables'][scope][m]['SearchAUC_mean'])<1e-10
    cfg=s.api.load_config(config_path);scanned=0
    for root in (s.ROOT,s.OUT):
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json','.csv','.xml','.md','.txt','.log') or any(part.startswith('pytest_tmp') for part in path.parts):continue
            assert cfg.api_key not in path.read_text(encoding='utf8'),'credential leaked';scanned+=1
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'logical_queries':240,'independently_replayed_choices_and_KF_updates':192,
             'independently_recomputed_qLogEI_scores':sum(a['independently_recomputed_qLogEI_scores'] for a in audits),
             'maximum_independent_qLogEI_error':max(a['maximum_independent_qLogEI_error'] for a in audits),
             'maximum_independent_filter_error':max(a['maximum_independent_filter_error'] for a in audits),
             'category_permutation_prefixes':192,'maximum_category_permutation_error':max(a['maximum_category_permutation_error'] for a in audits),
             'LLM_contexts_verified':contexts,'API_attempts_verified':attempts,'operator_compiles_in_loop':0,
             'protected_files_unchanged':protected,'protected_trees_unchanged':trees,'additional_protected_files_unchanged':len(p['additional_protected_files']),
             'credential_scan_files':scanned,'credential_leaks':0,'results_sha256':s.filehash(s.OUT/'results.json'),
             'boundary':'Nominal category invariance and implementation integrity; not statistical superiority or a calibrated chemical-knowledge certificate.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();v=s.read(s.OUT/'validation_receipt.json');assert v['integrity_status']=='PASS' and v['results_sha256']==s.filehash(s.OUT/'results.json')
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'));tests={k:sum(int(x.attrib.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']>=34 and tests['failures']==tests['errors']==0
    files=[Path(__file__),Path(s.__file__),Path(e.__file__),Path('chem/benchmarks/suzuki_mixed_kalman.py'),Path('tests/test_suzuki_mixed_kalman.py'),Path('docs/suzuki_mixed_kalman.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(part in ('cache','audit_cache','qualification_cache') for part in f.parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and not any(part.startswith('pytest_tmp') for part in f.parts) and f.name not in ('release_inventory.json','release_receipt.json')]
    cfg=s.api.load_config('../config.txt')
    for f in files:
        if f.suffix in ('.py','.json','.csv','.xml','.md','.txt','.log'):assert cfg.api_key not in f.read_text(encoding='utf8')
    inv={'protocol_sha256':p['protocol_sha256'],'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
         **{folder+'_tree':metrics.tree(s.ROOT/folder) for folder in ('cache','audit_cache','qualification_cache')}}
    inv['file_count']=len(inv['files']);s.write(s.OUT/'release_inventory.json',inv)
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'completed_trajectories':12,'logical_queries':240,'component_tests':tests,
             'usage':usage(),'operator_compiles_in_loop':0,'credential_leaks':0,'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
             'validation_sha256':s.filehash(s.OUT/'validation_receipt.json')}
    s.write(s.OUT/'release_receipt.json',receipt);return receipt
