"""Full-trajectory regional ablation metrics and independent audit."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import math
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.special import expit

from . import regional_ablation_suzuki_study as s
from . import regional_ablation_suzuki_engine as e
from . import mixed_kalman_suzuki_report as mixed_report
from . import kalman_knowledge_suzuki_report as algebra
from . import guarded_suzuki_report as subsets
from . import agent_score_suzuki_report as metrics
from .suzuki_domain import digest


def rows():return [s.read(s.ROOT/'runs'/v['run_id']/'state.json') for v in s.read(s.ROOT/'manifest.json')]


def summary(group):return mixed_report.summary(group)


def usage():
    responses=[s.read(f) for f in (s.ROOT/'requests').glob('*/attempt*/response.json')]
    states=[v for v in rows() if v['method']==s.METHODS[3]]
    return {'logical_refreshes':sum(len(v['calls']) for v in states),
            'accepted_refreshes':sum(c['status']=='ACCEPTED' for v in states for c in v['calls']),
            'received_API_responses':len(responses),
            'HTTP_errors':len(list((s.ROOT/'requests').glob('*/attempt*/transport_error.json'))),
            'schema_repairs':len(list((s.ROOT/'requests').glob('*/attempt*/validation_error.json'))),
            'models':sorted({v['model'] for v in responses}),
            'fingerprints':sorted({v['system_fingerprint'] for v in responses if v.get('system_fingerprint')}),
            **{k:sum(v['usage'].get(k,0) for v in responses) for k in ('prompt_tokens','completion_tokens','total_tokens')},
            'reasoning_tokens_included_in_completion':sum(v['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for v in responses)}


def mechanism(group):
    events=[e for r in group for e in r['decisions']]
    calls=[c for r in group for c in r['calls']]
    return {'query_count':len(events),'bank_count':sum(len(r['banks']) for r in group),
            'new_bank_rounds':sum(v['bank_before'] is not None and v['bank_before']['updates']==0 for v in events),
            'amplitude_mean':st.mean(v['amplitude'] for v in events),
            'amplitude_below_0_5_rounds':sum(v['amplitude']<.5 for v in events),
            'amplitude_zero_rounds':sum(v['amplitude']==0 for v in events),
            'B4_revocation_rounds':sum(bool(v['diagnosis']['revoked_ids']) for v in events),
            'B4_applied_revocation_rounds':sum(set(v['retained_ids'])!=set(h['id'] for h in v['knowledge']['clauses']) for v in events),
            'singleton_budget_rounds':sum(v['eligible_candidates']==1 for v in events),
            'refresh_rounds':{str(r['case']):[c['iteration'] for c in r['calls']] for r in group},
            'trigger_counts':{k:sum(k in c['reasons'] for c in calls) for k in sorted({k for c in calls for k in c['reasons']})},
            'deployed_minus_GP_MSE_pp2':st.mean((v['measurement']['squared_errors']['deployed']-v['measurement']['squared_errors']['GP'])*10000 for v in events)}


def report():
    p=s.check_protocol();allrows=rows();assert len(allrows)==64 and all(r['status'] in ('COMPLETE','FAILED') for r in allrows)
    tables={};increments={};overall={}
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        tables[scope]={};increments[scope]={}
        for regime in s.REGIMES:
            g={m:[r for r in allrows if r['case'] in cases and r['regime']==regime and r['method']==m] for m in s.METHODS}
            tables[scope][regime]={m:summary(v) for m,v in g.items()}
            increments[scope][regime]={f'{s.METHODS[i]}_minus_{s.METHODS[i-1]}':metrics.compare(g[s.METHODS[i]],g[s.METHODS[i-1]]) for i in (1,2,3)}
        ordered=lambda m:sorted([r for r in allrows if r['case'] in cases and r['method']==m],key=lambda r:(r['case'],r['regime']))
        overall[scope]={f'{s.METHODS[i]}_minus_{s.METHODS[i-1]}':metrics.compare(ordered(s.METHODS[i]),ordered(s.METHODS[i-1])) for i in (1,2,3)}
    baseline_equal=0
    for r in allrows:
        if r['method']!=s.METHODS[0]:continue
        old=s.read(s.prior.ROOT/'runs'/f"c{r['case']}_s0_{r['regime']}_LGBO_gate"/'state.json')
        assert old['history']==r['history'];baseline_equal+=1
    result={'study':p['study'],'protocol_sha256':p['protocol_sha256'],'planned_trajectories':64,
            'completed':sum(r['status']=='COMPLETE' for r in allrows),'logical_observations':sum(len(r['history']) for r in allrows),
            'added_queries':sum(len(r['decisions']) for r in allrows),'baseline_histories_exactly_reproduced':baseline_equal,
            'tables':tables,'ladder_increments':increments,'overall_ladder_increments':overall,'usage':usage(),
            'mechanism':{g:{m:mechanism([r for r in allrows if r['regime']==g and r['method']==m]) for m in s.METHODS} for g in s.REGIMES},
            'raw_metrics':[{'case':r['case'],'seed':0,'regime':r['regime'],'method':r['method'],'status':r['status'],**metrics.metric(r)} for r in allrows],
            'limits':['One seed per case; three primary reactions repeated under four knowledge conditions. No significance or12-independent-chemistries claim.',
                      'Ladder increments are conditional on addition order, not all2^3 factorial main effects.',
                      'Three artificial initial knowledge conditions are reference-outcome-aware stress tests.',
                      'KF is a new scalar calibration of the unchanged regional field, not the old additive residual-feature filter. A changed geometry has its own calibration bank.',
                      'No0.5SD cap or initial one-SD amplitude discount; initial regional confidence and shape are exactly matched.',
                      'The alias is not an immutable backend snapshot; fresh event prompts and response fingerprints are retained.']}
    s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['scope','knowledge_condition','method','n','SearchAUC_percent','final_yield_percent'])
        for scope,regimes in tables.items():
            for regime,methods in regimes.items():
                for method,x in methods.items():w.writerow([scope,regime,method,x['n'],x['SearchAUC_mean'],x['final_yield_mean']])
    figures(result)
    return {'completed':result['completed'],'primary':{g:{m:round(x['SearchAUC_mean'],5) for m,x in v.items()} for g,v in tables['primary'].items()},
            'overall_primary_ladder':overall['primary'],'usage':result['usage']}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('#111111','#0072B2','#D55E00','#009E73');labels=('Gate','Gate + KF','Gate + KF + B4','Gate + KF + B4 + event')
    fig,axes=plt.subplots(4,4,figsize=(15,10),layout='constrained')
    for i,c in enumerate((1,2,3,4)):
        for j,regime in enumerate(s.REGIMES):
            ax=axes[i,j]
            for m,color,label in zip(s.METHODS,colors,labels):
                ax.plot(range(5,21),result['tables'][f'Case{c}'][regime][m]['mean_curve'],label=label,color=color,lw=1.5)
            ax.set_title(f'Case {c}: {regime}',fontsize=9);ax.set_xticks([5,10,15,20]);ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Best yield (%)')
            if i==3:ax.set_xlabel('Total queries')
    axes[0,0].legend(fontsize=7,loc='lower right');fig.suptitle('Matched regional injection and initial knowledge: complete trajectories, seed 0',fontsize=12)
    fig.savefig(s.OUT/'learning_curves.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');x=np.arange(4)
    for i,(m,label,color) in enumerate(zip(s.METHODS,labels,colors)):
        ax.bar(x+(i-1.5)*.19,[result['tables']['primary'][g][m]['SearchAUC_mean'] for g in s.REGIMES],width=.19,label=label,color=color)
    ax.set_xticks(x,s.REGIMES);ax.set_ylabel('Case 2-4 mean SearchAUC (%)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax.set_title('Same regional field; sequential module additions, n=3 per condition');ax.spines[['top','right']].set_visible(False)
    fig.savefig(s.OUT/'primary_AUC.png',dpi=170,bbox_inches='tight');plt.close(fig)


def independent_shift(model,spec,z):
    xyz=torch.quasirandom.SobolEngine(3).draw(64).double().clamp(1e-6,1-1e-6)
    grid=torch.column_stack([xyz.repeat_interleave(8,0),torch.arange(8,dtype=torch.double).repeat(64)])
    a=np.asarray(spec['bounds']);xx=grid[:,:3].numpy()
    weights=(expit((xx-a[:,0])/(.08+1e-12))*expit((a[:,1]-xx)/(.08+1e-12))).prod(1)
    weights*=np.isin(np.tile(np.arange(8),64),spec['categories']);weights/=max(weights.sum(),1e-12)
    with torch.no_grad():
        cov=model.posterior(grid).mvn.covariance_matrix.numpy()
        # The unchanged vendor uses torch's default-float32 erfinv for this
        # constant. A float64 scipy ppf is a different numerical algorithm.
        confidence=min(max(float(spec['confidence']),1e-12),1-1e-12)
        delta=3*(math.sqrt(2.)*float(torch.erfinv(torch.tensor(2*confidence-1,dtype=torch.float32))))
        lam=float(delta/np.sqrt(max(float(weights@cov@weights),1e-6)))
        train=model.train_inputs[0]
        kt=model.covar_module(train,train).to_dense().numpy()+float(model.likelihood.noise)*np.eye(len(train))
        ktg=model.covar_module(train,grid).to_dense().numpy();u=np.linalg.solve(kt,ktg@weights)
        out=[]
        for x in z.split(256):out.append(lam*(model.covar_module(x,grid).to_dense().numpy()@weights-model.covar_module(x,train).to_dense().numpy()@u))
    return np.concatenate(out)


def audit_case(case,event_only=False,static_only=False):
    p=s.check_protocol();pool=s.inputs(case);ids=pool['ids'];ix={k:i for i,k in enumerate(ids)}
    specs=[v for v in s.read(s.ROOT/'manifest.json') if v['case']==case and
           (not event_only or v['method']==s.METHODS[3]) and (not static_only or v['method']!=s.METHODS[3])]
    cache=e.RegionalAblationCache(pool,s.ROOT/'audit_cache'/f'c{case}_s0',case,sources=[s.ROOT/'cache'/f'c{case}_s0'])
    values=np.load(s.prior.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    choices=0;score_count=0;max_score=0.;max_shift=0.;max_filter=0.;max_perm=0.;calls=0;roundoff=0
    fit_keys=set();score_keys=set();field_memory={};native_control_error=0.;propagation_bound=0.
    for spec in specs:
        actual=s.read(s.ROOT/'runs'/spec['run_id']/'state.json');assert actual['status'] in ('COMPLETE','FAILED')
        assert actual['protocol_sha256']==p['protocol_sha256']
        state={'case':case,'task':spec['task'],'method':spec['method'],'history':s.prior.initial(case,pool),
               'knowledge':s.initial_knowledge(case,spec['regime']),'slots':actual['slots'],'banks':{},'decisions':[],'calls':[]}
        assert actual['initial_knowledge']==state['knowledge'] and actual['history'][:4]==state['history']
        assert len(set(h['candidate_id'] for h in actual['history']))==len(actual['history'])
        for t,ev in enumerate(actual['decisions'],1):
            before=e.setup(cache,state,pool,t)
            call=next((c for c in actual['calls'] if c['iteration']==t),None)
            if spec['method']==s.METHODS[3]:
                reasons=e.event_reasons(state,before,t)
                assert bool(reasons)==bool(call)
                if call:
                    assert call['reasons']==reasons and call['knowledge_before']==state['knowledge']
                    context=s.read(s.ROOT/'requests'/call['context_sha256']/'context.json')
                    expected=s.public_request(state,pool,before,reasons)
                    assert context['public']==json.loads(json.dumps(expected)) and context['history']==state['history']
                    raw=deepcopy(context);key=raw.pop('context_sha256');assert digest(raw)==key==call['context_sha256']
                    if call['status']=='ACCEPTED':
                        folder=s.ROOT/'requests'/key;valid=s.read(folder/'validated.json');at=folder/f"attempt{valid['accepted_attempt']:02d}"
                        answer=s.read(at/'response.json');assert s.filehash(at/'response.json')==valid['response_sha256']
                        assert answer['model']==s.api.MODEL and answer['choices'][0]['finish_reason']=='stop'
                        parsed=e.validate_refresh(json.loads(answer['choices'][0]['message']['content']),state['knowledge'],state['slots'],len(state['history']),pool)
                        assert parsed==valid['plan']==call['plan']
                        public_messages=[{'role':'system','content':e.SYSTEM_PROMPT},{'role':'user','content':json.dumps(context['public'],ensure_ascii=False,separators=(',',':'))}]
                        received=0
                        for n in range(1,valid['accepted_attempt']+1):
                            a=folder/f'attempt{n:02d}';body=s.read(a/'request.json')
                            assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
                            assert body['messages'][:2]==public_messages and len(body['messages']) in (2,4)
                            if (a/'response.json').exists():
                                received+=1
                                if n<valid['accepted_attempt']:
                                    old=s.read(a/'response.json')
                                    try:
                                        assert old['model']==s.api.MODEL and old['choices'][0]['finish_reason']=='stop'
                                        e.validate_refresh(json.loads(old['choices'][0]['message']['content']),state['knowledge'],state['slots'],len(state['history']),pool)
                                    except (ValueError,TypeError,KeyError,IndexError,AssertionError):pass
                                    else:raise AssertionError('A valid refresh response was skipped')
                        assert received==valid['received_responses']
                        state['knowledge']=parsed['knowledge']
                    state['calls'].append(call);calls+=1
            decision=e.setup(cache,state,pool,t)
            assert decision['base_meta']==ev['base_acquisition'] and decision['guided_meta']==ev['guided_acquisition']
            assert state['knowledge']==ev['knowledge'] and decision['bank_key']==ev['bank_key']
            assert decision['bank']==ev['bank_before'] and decision['prior']==ev['bank_prior']
            assert decision['amplitude']==ev['amplitude']
            base=decision['base'];score=decision['score'];available=decision['available'];eligible=decision['eligible']
            retained=decision['retained'];hids=[h['id'] for h in state['knowledge']['clauses']]
            if spec['method'] in s.METHODS[2:]:
                keep,_,_=subsets.independent_subset(base,e.membership(pool,state['knowledge']),decision['budget'],~available,ids,hids)
                assert keep==tuple(retained)
            else:assert retained==sorted(hids)
            j=min(np.flatnonzero(eligible),key=lambda j:(-float(score[j]),ids[j]));assert ids[j]==ev['selected_id']
            assert float(base[decision['reference']]-base[j])==ev['base_logei_loss']<=decision['budget']
            fit,value=cache.model(state['history']);stats=decision['stats']
            if fit not in fit_keys:
                saved=torch.load(cache.directory/(fit+'.pt'),map_location='cpu',weights_only=False)
                expected=e.common.mixed.encode_mixed(pool)[[ix[h['candidate_id']] for h in sorted(state['history'],key=lambda h:h['candidate_id'])]]
                assert np.array_equal(saved['train_X'],expected)
                pm,ps=mixed_report.permutation_posterior(cache,state['history'],value['model'],value)
                er=max(float(np.max(np.abs(pm-stats['mu_std']))),float(np.max(np.abs(ps-stats['sd_std']))));max_perm=max(max_perm,er);assert er<1e-9;fit_keys.add(fit)
            score_key=ev['guided_acquisition']['key']
            field_key=(fit,decision['bank_key'])
            if field_key not in field_memory:
                raw=np.zeros(len(ids)) if not retained else independent_shift(value['model'],e.region(state['knowledge'],retained),cache.z)
                er=float(np.max(np.abs(raw-decision['raw_shift_std'])));max_shift=max(max_shift,er)
                assert er<1e-10,(case,spec['run_id'],t,er,decision['retained'],e.region(state['knowledge'],retained) if retained else None)
                field_memory[field_key]=raw
            raw=field_memory[field_key]
            if score_key not in score_keys:
                applied=raw*decision['amplitude']
                replay=algebra.marginal_scores(stats['mu_std'],stats['sd_std'],applied,value['best_std'],ev['base_acquisition']['mc_seed'],cache.z)
                er=float(np.max(np.abs(replay-score)));max_score=max(max_score,er)
                if er>=1e-8:
                    control=algebra.marginal_scores(stats['mu_std'],stats['sd_std'],decision['shift_std'],value['best_std'],ev['base_acquisition']['mc_seed'],cache.z)
                    ce=float(np.max(np.abs(control-score)));native_control_error=max(native_control_error,ce)
                    bound=float(np.max(np.abs(applied-decision['shift_std'])))/1e-6+ce+1e-10
                    assert ce<1e-8 and er<=bound,(case,spec['run_id'],t,er,bound)
                    roundoff+=1;propagation_bound=max(propagation_bound,bound)
                score_keys.add(score_key);score_count+=len(ids)
            y=float(values[j]);assert y==actual['history'][t+3]['y']
            raw_y=float(decision['raw_shift_std'][j]*float(value['std']));mu=float(stats['mu_y'][j]);sd=float(stats['sd_predictive_y'][j])
            m=ev['measurement'];assert abs(raw_y-ev['raw_shift_y'])<1e-12 and abs(mu-m['prequery_mu_y'])<1e-12
            deployed_shift=float(decision['shift_std'][j]*float(value['std']))
            assert abs(deployed_shift-ev['deployed_shift_y'])<1e-12
            predictions={'GP':mu,'regional_unscaled':mu+raw_y,'deployed':mu+deployed_shift}
            if decision['bank'] is not None:
                pr=decision['prior'];theta=np.asarray(pr['mean']);cov=np.asarray(pr['covariance']);h=np.array([1.,raw_y]);r=sd*sd+.0009
                innovation=y-mu-h@theta;variance=float(h@cov@h+r);gain=cov@h/variance
                updated=theta+gain*innovation;pc=cov-np.outer(cov@h,cov@h)/variance
                after=ev['bank_after'];er=max(float(np.max(np.abs(updated-after['mean']))),float(np.max(np.abs(pc-after['covariance']))))
                assert er<1e-11 and abs(innovation-m['innovation'])<1e-12 and abs(variance-m['innovation_variance'])<1e-12
                assert after['last_round']==t and after['updates']==decision['bank']['updates']+1
                assert np.linalg.eigvalsh(pc).min()>-1e-12;max_filter=max(max_filter,er)
                state['banks'][decision['bank_key']]=deepcopy(after)
                predictions['KF_joint']=mu+pr['mean'][0]+pr['mean'][1]*raw_y
            else:assert ev['bank_after'] is None
            assert set(predictions)==set(m['predictions'])
            for key,pred in predictions.items():
                assert abs(pred-m['predictions'][key])<1e-12 and abs((y-pred)**2-m['squared_errors'][key])<1e-12
            state['decisions'].append(ev);state['history'].append(actual['history'][t+3]);choices+=1
        assert state['history']==actual['history'] and state['banks']==actual['banks']
    return {'case':case,'choices':choices,'LLM_contexts':calls,'fit_keys':sorted(fit_keys),'nominal_prefixes':len(fit_keys),
            'max_label_error':max_perm,'qLogEI_values':score_count,'max_qLogEI_error':max_score,'max_region_shift_error':max_shift,
            'max_filter_error':max_filter,'roundoff_amplified_tables':roundoff,'max_same_shift_error':native_control_error,
            'max_roundoff_bound':propagation_bound,'new_audit_GP_fits':cache.counters['fits'],
            'operator_compiles_in_loop':sum(s.prior.previous.JIT.values())}


def audit_static_case(case):return audit_case(case,static_only=True)
def audit_event_case(case):return audit_case(case,event_only=True)


def verify_static():
    s.check_protocol()
    with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:audits=list(executor.map(audit_static_case,(1,2,3,4)))
    paths=[s.ROOT/'runs'/v['run_id']/'state.json' for v in s.read(s.ROOT/'manifest.json') if v['method']!=s.METHODS[3]]
    checkpoint={'status':'PASS','source_sha256':s.filehash(__file__),'states_sha256':{str(f):s.filehash(f) for f in paths},'cases':audits}
    s.write(s.OUT/'static_audit_checkpoint.json',checkpoint)
    return {'status':'PASS','choices':sum(a['choices'] for a in audits),'max_score_error':max(a['max_qLogEI_error'] for a in audits)}


def verify():
    p=s.check_protocol();r=s.read(s.OUT/'results.json');allrows=rows();assert r['protocol_sha256']==p['protocol_sha256']
    checkpoint=s.OUT/'static_audit_checkpoint.json'
    if checkpoint.exists():
        saved=s.read(checkpoint);assert saved['source_sha256']==s.filehash(__file__)
        for f,h in saved['states_sha256'].items():assert s.filehash(f)==h
        with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:extra=list(executor.map(audit_event_case,(1,2,3,4)))
        audits=saved['cases']+extra
    else:
        with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as executor:audits=list(executor.map(audit_case,(1,2,3,4)))
    s.write(s.OUT/'audit_cases.json',audits)
    for scope,cases in [('primary',(2,3,4)),('all_four',(1,2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        for regime in s.REGIMES:
            groups={m:[v for v in allrows if v['case'] in cases and v['regime']==regime and v['method']==m] for m in s.METHODS}
            for m,vs in groups.items():assert summary(vs)==r['tables'][scope][regime][m]
            for i in (1,2,3):
                key=f'{s.METHODS[i]}_minus_{s.METHODS[i-1]}'
                assert metrics.compare(groups[s.METHODS[i]],groups[s.METHODS[i-1]])==r['ladder_increments'][scope][regime][key]
    for regime in s.REGIMES:
        for case in (1,2,3,4):
            subset=[v for v in allrows if v['case']==case and v['regime']==regime]
            assert len({digest(v['initial_knowledge']) for v in subset})==1
            g0=next(v for v in subset if v['method']==s.METHODS[0]);g1=next(v for v in subset if v['method']==s.METHODS[1])
            assert g0['decisions'][0]['amplitude']==g1['decisions'][0]['amplitude']==1
            assert g0['history'][4]==g1['history'][4]
    frozen=s.read(s.OUT/'static_frozen.json')
    for f,h in frozen['states'].items():assert s.filehash(f)==h
    protected={}
    for name,entry in p['protected_inventories'].items():
        inventory=s.read(entry['path'])
        for f in inventory['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[name]=len(inventory['files'])
    for f,h in p['additional_protected_files'].items():assert s.filehash(f)==h
    cfg=s.api.load_config('../config.txt');files=[f for f in s.ROOT.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.txt','.log','.py')]
    for f in files:assert cfg.api_key not in f.read_text(encoding='utf8')
    assert all(a['new_audit_GP_fits']==0 and a['operator_compiles_in_loop']==0 for a in audits)
    assert sum(a['choices'] for a in audits)==r['added_queries']
    receipt={'integrity_status':'PASS','complete':r['completed'],'planned':64,'logical_observations':r['logical_observations'],
             'actual_choices_verified':sum(a['choices'] for a in audits),'LLM_contexts_verified':sum(a['LLM_contexts'] for a in audits),
             'category_permuted_unique_prefixes':sum(len({key for a in audits if a['case']==c for key in a['fit_keys']}) for c in (1,2,3,4)),
             'max_label_permutation_error':max(a['max_label_error'] for a in audits),'independent_qLogEI_values':sum(a['qLogEI_values'] for a in audits),
             'max_qLogEI_error':max(a['max_qLogEI_error'] for a in audits),'max_region_shift_error':max(a['max_region_shift_error'] for a in audits),
             'max_independent_filter_error':max(a['max_filter_error'] for a in audits),'roundoff_amplified_tables':sum(a['roundoff_amplified_tables'] for a in audits),
             'max_same_shift_qLogEI_error':max(a['max_same_shift_error'] for a in audits),'max_roundoff_propagation_bound':max(a['max_roundoff_bound'] for a in audits),
             'protected_files_unchanged':protected,'credential_scan_files':len(files),'credential_leaks':0,
             'operator_compiles_in_loop':0,'results_sha256':s.filehash(s.OUT/'results.json'),
             'boundary':'Implementation integrity and matched initial/regional controls; not statistical significance or chemical truth.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();v=s.read(s.OUT/'validation_receipt.json');assert v['integrity_status']=='PASS' and v['results_sha256']==s.filehash(s.OUT/'results.json')
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'))
    tests={k:sum(int(a.attrib.get(k,0)) for a in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']>=32 and tests['failures']==tests['errors']==0
    files=[Path(__file__),Path(s.__file__),Path(e.__file__),Path('chem/benchmarks/suzuki_regional_ablation.py'),
           Path('tests/test_suzuki_regional_ablation.py'),Path('docs/suzuki_regional_ablation.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(x in ('cache','audit_cache','qualification_cache') for x in f.parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and not any(x.startswith('pytest_tmp') for x in f.parts) and f.name not in ('release_receipt.json','release_inventory.json')]
    cfg=s.api.load_config('../config.txt')
    for f in files:
        if f.suffix in ('.json','.py','.csv','.md','.txt','.xml','.log'):assert cfg.api_key not in f.read_text(encoding='utf8')
    inventory={'protocol_sha256':p['protocol_sha256'],'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
               **{folder+'_tree':metrics.tree(s.ROOT/folder) for folder in ('cache','audit_cache','qualification_cache')}}
    s.write(s.OUT/'release_inventory.json',inventory)
    receipt={'integrity_status':'PASS','completed':v['complete'],'trajectories':64,'logical_observations':v['logical_observations'],
             'component_tests':tests,'usage':usage(),'operator_compiles_in_loop':0,'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
             'validation_sha256':s.filehash(s.OUT/'validation_receipt.json')}
    s.write(s.OUT/'release_receipt.json',receipt);return receipt
