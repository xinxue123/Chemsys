"""Paired system comparisons and independent audit of all planned Mixed runs."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import torch

from . import mixed_comparison_suzuki_study as s
from . import mixed_comparison_suzuki_engine as e
from . import mixed_kalman_suzuki_report as kreport
from . import kalman_knowledge_suzuki_report as kalgebra
from . import guarded_suzuki_report as original
from . import agent_score_suzuki_report as metrics
from .suzuki_domain import digest


def groups():
    result={r:{m:[] for m in s.METHODS} for r in s.REGIMES}
    for spec in s.read(s.ROOT/'manifest.json'):
        result[spec['regime']][spec['method']].append(s.read(s.ROOT/'runs'/spec['run_id']/'state.json'))
    return result


def references():
    return {m:[s.read(s.previous.ROOT/folder/f'c{case}_s0'/'state.json') for case in (1,2,3,4)]
            for m,folder in zip(s.previous.METHODS,('runs','frozen','base'))}


def summary(group):return kreport.summary(group)


def usage(contexts=None):
    def chosen(f):return contexts is None or f.relative_to(s.ROOT/'requests').parts[0] in contexts
    responses=[s.read(f) for f in (s.ROOT/'requests').glob('*/attempt*/response.json') if chosen(f)]
    return {'logical_proposals':sum(chosen(f) for f in (s.ROOT/'requests').glob('*/validated.json')),
            'received_API_responses':len(responses),
            'HTTP_errors':sum(chosen(f) for f in (s.ROOT/'requests').glob('*/attempt*/transport_error.json')),
            'schema_repairs':sum(chosen(f) for f in (s.ROOT/'requests').glob('*/attempt*/validation_error.json')),
            'models':sorted({r['model'] for r in responses}),
            'fingerprints':sorted({r['system_fingerprint'] for r in responses if r.get('system_fingerprint')}),
            **{key:sum(r['usage'].get(key,0) for r in responses) for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'reasoning_tokens_included_in_completion':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in responses)}


def report():
    p=s.check_protocol();g=groups();refs=references();allrows=[r for methods in g.values() for rows in methods.values() for r in rows]
    assert len(allrows)==80 and all(r['status'] in ('COMPLETE','FAILED') for r in allrows)
    tables={};paired={}
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        tables[scope]={regime:{method:summary([r for r in rows if r['case'] in cases])
                              for method,rows in methods.items()} for regime,methods in g.items()}
        tables[scope]['new_framework_reference']={m:summary([r for r in rows if r['case'] in cases]) for m,rows in refs.items()}
        kf=[r for r in refs['Mixed_KF_event_B4'] if r['case'] in cases]
        paired[scope]={m:metrics.compare(kf,[r for r in rows if r['case'] in cases]) for m,rows in g['assistant_initial'].items()}
    mechanism={regime:{m:{'queries':sum(len(r['decisions']) for r in rows),
        'singleton_budget_rounds':sum(v.get('eligible_candidates')==1 for r in rows for v in r['decisions']),
        'LLINBO_accepts':sum(v.get('accepted',False) for r in rows for v in r['decisions']),
        'LLINBO_rejects':sum(v.get('accepted') is False for r in rows for v in r['decisions']),
        'B4_revoked_clauses':sum(len(v.get('revoked_ids',[])) for r in rows for v in r['decisions']),
        'soft_selected_outside_retained_box':sum(not v.get('selected_satisfies_retained_box',True) for r in rows for v in r['decisions']),
        'budget_violations':sum(v.get('budget_violation',False) is True for r in rows for v in r['decisions'])}
        for m,rows in methods.items()} for regime,methods in g.items()}
    # LLINBO's legacy event schema did not store the eligible-set count.
    # Derive it from its own frozen base scores; absence is not a zero count.
    for regime,methods in g.items():
        singleton=0
        for row in methods['LLINBO_matched']:
            pool=s.inputs(row['case']);ids=pool['ids']
            for t,event in enumerate(row['decisions'],1):
                path=s.ROOT/'cache'/f"c{row['case']}_s0"/(event['base_acquisition']['key']+'.npz')
                with np.load(path,allow_pickle=False) as data:scores=data['scores']
                available=~np.isin(ids,[h['candidate_id'] for h in row['history'][:t+3]])
                singleton+=int(np.sum(available&((scores[available].max()-scores)<=event['budget_logei']))==1)
        mechanism[regime]['LLINBO_matched']['singleton_budget_rounds']=singleton
        mechanism[regime]['LGBO_region']['budget_violations']=None
    old_deltas={}
    for regime,methods in g.items():
        old_deltas[regime]={}
        for method,rows in methods.items():
            old=[]
            for row in rows:
                case=row['case'];key=f'c{case}_s0_{regime}_{method}'
                if method=='LLINBO_matched':path=Path('data/runs/suzuki_llinbo_guarded/runs')/key/'state.json'
                else:
                    if method!='LGBO_region':key+='_r4'
                    path=s.original.ROOT/('development' if case==1 else 'test')/'runs'/key/'state.json'
                old.append(s.read(path))
            old_deltas[regime][method]=metrics.compare([r for r in rows if r['case'] in (2,3,4)],
                                                      [r for r in old if r['case'] in (2,3,4)])
    result={'study':p['study'],'protocol_sha256':p['protocol_sha256'],
            'new_trajectories':80,'completed_new_trajectories':sum(r['status']=='COMPLETE' for r in allrows),
            'new_logical_queries':sum(len(r['history']) for r in allrows),'new_added_queries':sum(len(r['decisions']) for r in allrows),
            'reused_Mixed_reference_trajectories':12,'tables':tables,'KF_minus_natural_baseline':paired,
            'Mixed_minus_previous_ordinal_primary':old_deltas,'mechanism':mechanism,'usage':usage(),
            'LLINBO_usage_by_regime':{regime:usage({ev['response_context_sha256'] for row in methods['LLINBO_matched'] for ev in row['decisions']}) for regime,methods in g.items()},
            'existing_KF_usage_all_four_cases':kreport.usage(),
            'raw_metrics':[{'method':r['method'],'regime':r['regime'],'case':r['case'],'seed':r['seed'],'status':r['status'],**metrics.metric(r)} for r in allrows],
            'limits':['One seed/case; primary n=3; descriptive, no statistical superiority.',
                      'Artificial priors use reference outcomes; KF reference did not receive those priors.',
                      'Natural initial knowledge and LLM cadence differ between baseline systems and KF; common BO is not knowledge-matched causal isolation.',
                      'Nominal category quadrature is a necessary regional-domain adaptation; inherited tilt formulas and selection mechanisms unchanged.',
                      'Old LLINBO provider was current assistant; present provider and backend fingerprint are recorded. Cross-old LLINBO changes are not a pure GP ablation.',
                      'MC512 qLogEI tail sensitivity and shrinking r4 budget retained.']}
    s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','knowledge_condition','method','n','SearchAUC_percent','final_yield_percent'])
        for scope,conditions in tables.items():
            for regime,methods in conditions.items():
                for method,v in methods.items():writer.writerow([scope,regime,method,v['n'],v['SearchAUC_mean'],v['final_yield_mean']])
    figures(result)
    return {'completed':result['completed_new_trajectories'],'primary':tables['primary'],'usage':result['usage']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained')
    styles={'Mixed_KF_event_B4':('KF/event/B4','#0072B2','-'),
            'Mixed_KF_frozen_initial_B4':('Frozen KF/B4','#888888',':'),
            'Mixed_qLogEI_B0':('Mixed pure BO','#000000','--'),
            'LGBO_region':('Regional LGBO','#009E73','-'),
            'LLINBO_matched':('LLINBO Justify','#D55E00','-'),
            'LGBO_gate':('Budget gate','#CC79A7','-'),
            'B4_base':('Hard B4','#E69F00','--'),
            'B4_LGBO':('Soft B4','#56B4E9','--')}
    for case,ax in enumerate(axes,1):
        table=dict(result['tables'][f'Case{case}']['assistant_initial'],**result['tables'][f'Case{case}']['new_framework_reference'])
        for method,(label,color,ls) in styles.items():
            ax.plot(range(5,21),table[method]['mean_curve'],label=label,color=color,ls=ls,lw=1.6)
        ax.set(title=f'Case {case}, seed 0',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Incumbent yield (%)');axes[-1].legend(fontsize=7,loc='lower right')
    fig.suptitle('Common nominal Mixed GP, 4+16 queries: natural-knowledge system comparison',fontsize=11)
    fig.savefig(s.OUT/'natural_comparison.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.8),layout='constrained')
    for regime,ax in zip(s.REGIMES,axes):
        values=[result['tables']['primary'][regime][m]['SearchAUC_mean'] for m in s.METHODS]
        ax.barh(range(5),values,color=[styles[m][1] for m in s.METHODS]);ax.set_yticks(range(5),[styles[m][0] for m in s.METHODS],fontsize=8)
        ax.set(title=regime,xlabel='Case 2-4 mean SearchAUC (%)',xlim=(0,100));ax.invert_yaxis()
        for i,v in enumerate(values):ax.text(v+.5,i,f'{v:.2f}',va='center',fontsize=8)
    fig.suptitle('Four fixed initial knowledge conditions; KF is not a matched artificial-prior control',fontsize=11)
    fig.savefig(s.OUT/'knowledge_conditions.png',dpi=170,bbox_inches='tight');plt.close(fig)


def load_scores(cache,meta):
    path=cache.directory/(meta['key']+'.npz');cache.ensure(path.name)
    with np.load(path,allow_pickle=False) as d:
        assert json.loads(str(d['metadata']))==meta
        return {k:d[k].copy() for k in d.files if k!='metadata'}


def independent_region_shift(model,meta,z):
    # Separate weight construction and numpy linear solve, no vendor tilt call.
    grid=e.NominalRegionalTilt().Xg.clone()
    xyz=grid[:,:3].numpy();lo,hi=np.asarray(meta['box'])
    from scipy.special import expit
    allowed={k for k in range(8) if lo[3]<=(k+.5)/8<=hi[3]}
    weights=(expit((xyz-lo[:3])/(.08+1e-12))*expit((hi[:3]-xyz)/(.08+1e-12))).prod(1)
    weights*=np.isin(np.tile(np.arange(8),64),list(allowed));weights/=max(weights.sum(),1e-12)
    with torch.no_grad():
        post=model.posterior(grid);cov=post.mvn.covariance_matrix.numpy()
        lam=meta['delta']/np.sqrt(max(float(weights@cov@weights),1e-6))
        train=model.train_inputs[0]
        a=model.covar_module(train,train).to_dense().numpy()+float(model.likelihood.noise)*np.eye(len(train))
        ktg=model.covar_module(train,grid).to_dense().numpy()
        u=np.linalg.solve(a,ktg@weights)
        shifts=[]
        for x in z.split(256):
            shifts.append(lam*(model.covar_module(x,grid).to_dense().numpy()@weights-
                               model.covar_module(x,train).to_dense().numpy()@u))
    return np.concatenate(shifts),weights,lam


def audit_case(case,kind='all'):
    pool=s.inputs(case);ids=pool['ids'];ix={k:i for i,k in enumerate(ids)}
    cache=e.MixedRegionalCache(pool,s.ROOT/'audit_cache'/f'c{case}_s0',case,
            sources=[s.ROOT/'cache'/f'c{case}_s0',s.ROOT/'qualification_cache'/f'c{case}_s0',s.previous.ROOT/'cache'/f'c{case}_s0'])
    values=np.load(s.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    checked=0;scores_checked=0;score_error=0.;shift_error=0.;permutation_error=0.;fits_checked=set();score_keys=set();subset_checks=0
    amplified_tables=0;fixed_shift_error=0.;propagation_bound=0.
    specs=[v for v in s.read(s.ROOT/'manifest.json') if v['case']==case and
           (kind=='all' or (v['method']=='LLINBO_matched')==(kind=='llinbo'))]
    rows=[s.read(s.ROOT/'runs'/v['run_id']/'state.json') for v in specs]
    assert all(r['status'] in ('COMPLETE','FAILED') for r in rows)
    for row in rows:
        h=row['history'];assert h[:4]==s.initial(case,pool)
        assert len({v['candidate_id'] for v in h})==len(h)
        assert all(v['y']==float(values[ix[v['candidate_id']]]) for v in h)
        prior=s.advice(case,row['regime']);member=e.regional.clause_membership(pool['z'],prior);hids=[v['id'] for v in prior['clauses']]
        for t,event in enumerate(row['decisions'],1):
            prefix=h[:t+3];base,bmeta=cache.score(prefix);assert bmeta==event['base_acquisition']
            stats=cache.posterior_stats(prefix);fit,value=cache.model(prefix)
            if fit not in fits_checked:
                saved=torch.load(cache.directory/(fit+'.pt'),map_location='cpu',weights_only=False)
                expected=e.mixed.encode_mixed(pool)[[ix[v['candidate_id']] for v in sorted(prefix,key=lambda v:v['candidate_id'])]]
                assert np.array_equal(saved['train_X'],expected) and saved['cat_dims']==[3]
                pm,ps=kreport.permutation_posterior(cache,prefix,value['model'],value)
                err=max(float(np.max(np.abs(pm-stats['mu_std']))),float(np.max(np.abs(ps-stats['sd_std']))))
                permutation_error=max(permutation_error,err);assert err<1e-9;fits_checked.add(fit)
            metas=[bmeta]+list(event.get('guided_acquisitions',{}).values())
            for meta in metas:
                if meta['key'] in score_keys:continue
                stored=load_scores(cache,meta);shift=np.zeros(len(ids));shift_roundoff=0.
                if 'regional_recipe' in meta:
                    shift,weights,lam=independent_region_shift(value['model'],meta,cache.z)
                    shift_roundoff=float(np.max(np.abs(shift-stored['shift_std'])));shift_error=max(shift_error,shift_roundoff)
                    assert shift_roundoff<1e-11 and np.allclose(weights,stored['weights'],rtol=0,atol=1e-14) and abs(lam-meta['lambda'])<1e-9
                replay=kalgebra.marginal_scores(stats['mu_std'],stats['sd_std'],shift,value['best_std'],meta['mc_seed'],cache.z)
                err=float(np.max(np.abs(replay-stored['scores'])));score_error=max(score_error,err)
                if err>=1e-8 and 'regional_recipe' in meta:
                    control=kalgebra.marginal_scores(stats['mu_std'],stats['sd_std'],stored['shift_std'],value['best_std'],meta['mc_seed'],cache.z)
                    ce=float(np.max(np.abs(control-stored['scores'])));fixed_shift_error=max(fixed_shift_error,ce)
                    # q=1 log fatplus is 1/tau_relu-Lipschitz in the mean:
                    # log softplus and log Cauchy each have |derivative|<=1;
                    # positive sums / logmeanexp preserve this bound. Default
                    # tau_relu=1e-6. Check propagated mean roundoff separately,
                    # never relax a decision/gating threshold or alter scores.
                    bound=shift_roundoff/1e-6+ce+1e-10;propagation_bound=max(propagation_bound,bound)
                    assert ce<1e-8 and err<=bound,(case,row['run_id'],t,meta['key'],err,bound)
                    amplified_tables+=1
                else:assert err<1e-8,(case,row['run_id'],t,meta['key'],err)
                score_keys.add(meta['key']);scores_checked+=len(ids)
            available=~np.isin(ids,[v['candidate_id'] for v in prefix]);queried=~available
            ref=min(np.flatnonzero(available),key=lambda i:(-float(base[i]),ids[i]))
            epsilon=e.regional.budget_at(4,t);eligible=available&((base[ref]-base)<=epsilon)
            chosen=ix[event['selected_id']]
            assert event['selected_id']==h[t+3]['candidate_id']
            if row['method']=='LLINBO_matched':
                prop=ix[event['proposed_id']];accepted=not(base[ref]>base[prop]+epsilon)
                expected=prop if accepted else ref
                assert event['accepted']==accepted and event['maximum_base_logei']==float(base[ref])
                assert event['proposal_base_logei']==float(base[prop]) and event['budget_logei']==epsilon
            else:
                kept=tuple(event['retained_ids'])
                if row['method'] in ('B4_base','B4_LGBO'):
                    keep,hard,_=original.independent_subset(base,member,epsilon,queried,ids,hids)
                    assert keep==kept;subset_checks+=1
                else:assert kept==tuple(sorted(hids))
                guided=base
                if kept and row['method']!='B4_base':guided=load_scores(cache,event['guided_acquisitions'][','.join(kept)])['scores']
                mask=available if row['method']=='LGBO_region' else eligible
                expected=hard if row['method']=='B4_base' else min(np.flatnonzero(mask),key=lambda i:(-float(guided[i]),ids[i]))
                assert event['reference_id']==ids[ref]
                assert event['base_logei_loss']==float(base[ref]-base[chosen])
            assert chosen==expected and available[chosen]
            if row['method']!='LGBO_region':assert float(base[ref]-base[chosen])<=epsilon
            checked+=1
        if row['method']=='LLINBO_matched':
            cursor=[]
            def assistant(history,prompt,t):
                event=row['decisions'][t-1];folder=s.ROOT/'requests'/event['response_context_sha256'];context=s.read(folder/'context.json')
                assert context['public']==e.public_point_request(history,prompt,t,pool,prior,row['task'])
                assert history==h[:t+3] and context['history']==history
                raw=deepcopy(context);key=raw.pop('context_sha256');assert digest(raw)==key
                valid=s.read(folder/'validated.json');attempt=folder/f"attempt{valid['accepted_attempt']:02d}"
                if s.read(s.OUT/'protocol.json')['provider']=='deepseek-flash':
                    answer=s.read(attempt/'response.json');parsed=e.parse_point(answer,history)
                    assert all(valid[k]==parsed[k] for k in ('point','reason','support'))
                    assert valid['response_sha256']==s.filehash(attempt/'response.json')
                    base_messages=[{'role':'system','content':e.POINT_SYSTEM},
                        {'role':'user','content':json.dumps(context['public'],ensure_ascii=False,separators=(',',':'))}]
                    received=0
                    for n in range(1,valid['accepted_attempt']+1):
                        at=folder/f'attempt{n:02d}';body=s.read(at/'request.json')
                        assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
                        assert body['messages'][:2]==base_messages
                        assert len(body['messages']) in (2,4)
                        if (at/'response.json').exists():
                            received+=1
                            if n<valid['accepted_attempt']:
                                try:e.parse_point(s.read(at/'response.json'),history)
                                except (ValueError,TypeError,KeyError,IndexError):pass
                                else:raise AssertionError('valid proposal was skipped')
                    assert received==valid['received_responses']
                projection=e.justify.project_proposal(valid['point'],pool,{v['candidate_id'] for v in history})
                assert projection==s.read(folder/'projection.json')==event['projection']
                cursor.append(t);return projection,valid
            def objective(candidate,event):
                assert event==row['decisions'][event['iteration']-1]
                assert candidate==h[event['iteration']+3]['candidate_id']
                return h[event['iteration']+3]['y']
            e.justify.execute(pool=pool,initial=h[:4],advice=prior,case=case,seed=0,ratio=4,
                              cache=cache,assistant=assistant,objective=objective,steps=len(row['decisions']))
            assert cursor==list(range(1,len(row['decisions'])+1))
    return {'case':case,'choices':checked,'B4_independent_subset_checks':subset_checks,
            'category_permuted_unique_prefixes':len(fits_checked),'max_category_permutation_error':permutation_error,
            'verified_fit_keys':sorted(fits_checked),
            'independent_qLogEI_values':scores_checked,'max_qLogEI_error':score_error,
            'max_regional_shift_error':shift_error,'cache_counters':cache.counters,
            'roundoff_amplified_score_tables':amplified_tables,'max_same_shift_qLogEI_error':fixed_shift_error,
            'max_roundoff_propagation_bound':propagation_bound,
            'operator_compiles_in_loop':sum(s.previous.JIT.values())}


def audit_regional_case(case):return audit_case(case,'regional')
def audit_llinbo_case(case):return audit_case(case,'llinbo')


def verify_regional():
    s.check_protocol()
    with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:
        audits=list(executor.map(audit_regional_case,(1,2,3,4)))
    states={str(s.ROOT/'runs'/v['run_id']/'state.json'):s.filehash(s.ROOT/'runs'/v['run_id']/'state.json')
            for v in s.read(s.ROOT/'manifest.json') if v['method']!='LLINBO_matched'}
    result={'integrity_status':'PASS','audit_source_sha256':s.filehash(__file__),'states_sha256':states,'cases':audits}
    s.write(s.OUT/'regional_audit_checkpoint.json',result)
    return {'status':'PASS','choices':sum(a['choices'] for a in audits),
            'max_qLogEI_error':max(a['max_qLogEI_error'] for a in audits)}


def verify():
    p=s.check_protocol();r=s.read(s.OUT/'results.json');assert r['protocol_sha256']==p['protocol_sha256']
    assert s.read(s.OUT/'start_receipt.json')['states_before_start']==0
    assert s.read(s.OUT/'start_receipt.json')['new_framework_results_sha256']==s.filehash(s.previous.OUT/'results.json')
    checkpoint=s.OUT/'regional_audit_checkpoint.json'
    if checkpoint.exists():
        saved=s.read(checkpoint);assert saved['audit_source_sha256']==s.filehash(__file__)
        for f,h in saved['states_sha256'].items():assert s.filehash(f)==h
        with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:
            remaining=list(executor.map(audit_llinbo_case,(1,2,3,4)))
        audits=[]
        for a,b in zip(saved['cases'],remaining):
            row={'case':a['case'],'verified_fit_keys':sorted(set(a['verified_fit_keys'])|set(b['verified_fit_keys']))}
            for k in ('choices','B4_independent_subset_checks','independent_qLogEI_values','operator_compiles_in_loop','roundoff_amplified_score_tables'):row[k]=a[k]+b[k]
            for k in ('max_category_permutation_error','max_qLogEI_error','max_regional_shift_error','max_same_shift_qLogEI_error','max_roundoff_propagation_bound'):row[k]=max(a[k],b[k])
            row['category_permuted_unique_prefixes']=len(row['verified_fit_keys']);audits.append(row)
    else:
        with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:audits=list(executor.map(audit_case,(1,2,3,4)))
    s.write(s.OUT/'audit_cases.json',audits)
    # Recompute each reported metric from histories and retain all planned rows.
    g=groups();assert sum(len(v) for methods in g.values() for v in methods.values())==80
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        for regime,methods in g.items():
            for method,rows in methods.items():assert summary([v for v in rows if v['case'] in cases])==r['tables'][scope][regime][method]
    protected={}
    for label,entry in p['protected_inventories'].items():
        inventory=s.read(entry['path']);count=0
        for f in inventory['files']:
            assert s.filehash(f['path'])==f['sha256'],f['path'];count+=1
        protected[label]=count
    for f,h in p['additional_protected_files'].items():assert s.filehash(f)==h
    # Previous Mixed cache trees are specifically protected as reused references.
    inventory=s.read(s.previous.OUT/'release_inventory.json')
    for folder in ('cache','audit_cache','qualification_cache'):
        assert metrics.tree(s.previous.ROOT/folder)==inventory[folder+'_tree']
    cfg=s.api.load_config('../config.txt');files=[f for f in s.ROOT.rglob('*') if f.is_file() and f.suffix in ('.json','.py','.csv','.txt','.log')]
    for f in files:assert cfg.api_key not in f.read_text(encoding='utf8')
    assert sum(a['choices'] for a in audits)==r['new_added_queries']
    assert all(a['operator_compiles_in_loop']==0 for a in audits)
    assert all(v.get('operator_compiles_in_loop',0)==0 for methods in g.values() for rows in methods.values() for v in rows)
    receipt={'integrity_status':'PASS','new_trajectories':80,'completed':r['completed_new_trajectories'],
             'new_logical_queries':r['new_logical_queries'],'choices_independently_verified':sum(a['choices'] for a in audits),
             'LLINBO_contexts_and_official_branch_replays':sum(len(row['decisions']) for methods in g.values() for row in methods['LLINBO_matched']),
             'B4_independent_subset_checks':sum(a['B4_independent_subset_checks'] for a in audits),
             'category_permuted_unique_prefixes':sum(a['category_permuted_unique_prefixes'] for a in audits),
             'max_category_permutation_error':max(a['max_category_permutation_error'] for a in audits),
             'independent_qLogEI_values':sum(a['independent_qLogEI_values'] for a in audits),
             'max_qLogEI_error':max(a['max_qLogEI_error'] for a in audits),
             'max_regional_shift_error':max(a['max_regional_shift_error'] for a in audits),
             'roundoff_amplified_score_tables':sum(a['roundoff_amplified_score_tables'] for a in audits),
             'max_same_shift_qLogEI_error':max(a['max_same_shift_qLogEI_error'] for a in audits),
             'max_roundoff_propagation_bound':max(a['max_roundoff_propagation_bound'] for a in audits),
             'protected_files_unchanged':protected,'previous_Mixed_cache_trees_unchanged':True,
             'operator_compiles_in_loop':0,'credential_scan_files':len(files),'credential_leaks':0,
             'results_sha256':s.filehash(s.OUT/'results.json'),
             'boundary':'Implementation integrity and categorical representation, not statistical or chemical-knowledge superiority.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();v=s.read(s.OUT/'validation_receipt.json')
    assert v['integrity_status']=='PASS' and v['results_sha256']==s.filehash(s.OUT/'results.json')
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'))
    tests={k:sum(int(x.attrib.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']>=30 and tests['failures']==tests['errors']==0
    files=[Path(__file__),Path(s.__file__),Path(e.__file__),Path('chem/benchmarks/suzuki_mixed_comparison.py'),
           Path('tests/test_suzuki_mixed_comparison.py'),Path('docs/suzuki_mixed_comparison.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(k in ('cache','audit_cache','qualification_cache') for k in f.parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and not any(k.startswith('pytest_tmp') for k in f.parts) and f.name not in ('release_inventory.json','release_receipt.json')]
    cfg=s.api.load_config('../config.txt')
    for f in files:
        if f.suffix in ('.json','.py','.csv','.txt','.md','.xml','.log'):assert cfg.api_key not in f.read_text(encoding='utf8')
    inventory={'protocol_sha256':p['protocol_sha256'],
               'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
               **{folder+'_tree':metrics.tree(s.ROOT/folder) for folder in ('cache','audit_cache','qualification_cache')}}
    s.write(s.OUT/'release_inventory.json',inventory)
    receipt={'integrity_status':'PASS','trajectories':80,'completed':v['completed'],'tests':tests,
             'logical_queries':v['new_logical_queries'],'usage':usage(),'operator_compiles_in_loop':0,
             'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),'validation_sha256':s.filehash(s.OUT/'validation_receipt.json')}
    s.write(s.OUT/'release_receipt.json',receipt);return receipt
