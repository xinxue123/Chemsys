"""Outcome reporting and independent prequential receipts; never proposes runs."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import statistics as st
import xml.etree.ElementTree as ET

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal

from . import kalman_knowledge_suzuki_study as s
from . import kalman_knowledge_suzuki_engine as e
from . import deepseek_small_suzuki_report as old
from . import agent_score_suzuki_report as metrics
from . import llinbo_guarded_report as regional
from .suzuki_domain import digest


def rows():
    result={m:[s.read(s.ROOT/folder/f'c{c}_s0'/'state.json') for c in (1,2,3,4)] for m,folder in zip(s.METHODS,('runs','frozen'))}
    result.update(old.rows())
    for method in ('LGBO_region','LGBO_gate','B4_base','B4_LGBO'):
        result[method+'_natural_reference']=[regional.baseline(c,0,'assistant_initial',method) for c in (1,2,3,4)]
    result['LLINBO_matched_natural_reference']=[s.read(Path('data/runs/suzuki_llinbo_guarded/runs')/f'c{c}_s0_assistant_initial_LLINBO_matched'/'state.json') for c in (1,2,3,4)]
    return result


def summary(group):
    values=[metrics.metric(r) for r in group]
    return {'n':len(group),'completed':sum(r['status']=='COMPLETE' for r in group),
            'SearchAUC_mean':st.mean(v['AUC'] for v in values),'final_yield_mean':st.mean(v['final'] for v in values),
            'mean_curve':[st.mean(v['curve'][i] for v in values) for i in range(16)]}


def usage():
    answers=[s.read(p) for p in (s.ROOT/'requests').glob('*/attempt*/response.json')]
    return {'logical_calls':sum(len(r['calls']) for r in rows()[s.METHODS[0]]),'received_API_responses':len(answers),
            'HTTP_errors':len(list((s.ROOT/'requests').glob('*/attempt*/transport_error.json'))),
            'schema_failures':len(list((s.ROOT/'requests').glob('*/attempt*/validation_error.json'))),
            'returned_models':sorted({r['model'] for r in answers}),'fingerprints':sorted({r['system_fingerprint'] for r in answers if r.get('system_fingerprint')}),
            **{key:sum(r['usage'].get(key,0) for r in answers) for key in ('prompt_tokens','completion_tokens','total_tokens')},
            'reasoning_tokens_included_in_completion':sum(r['usage'].get('completion_tokens_details',{}).get('reasoning_tokens',0) for r in answers)}


def mechanisms(group):
    events=[v for r in group for v in r['decisions']]
    pred={k:float(np.sqrt(np.mean([v['measurement']['squared_errors'][k] for v in events]))*100) for k in ('GP','GP_offset_null','GP_joint_KF','GP_deployed')}
    calls=[c for r in group for c in r['calls']];ops={op:0 for op in ('add','retain','revise','revoke')}
    for c in calls:
        if c['status']!='ACCEPTED':continue
        for h in c['plan']['hypotheses']:ops[h['operation']]+=1
        ops['revoke']+=len(c['plan']['revocations'])
    return {'queries':len(events),'LLM_calls':len(calls),'accepted_calls':sum(c['status']=='ACCEPTED' for c in calls),'lifecycle_actions':ops,
            'trigger_counts':{key:sum(key in c['reasons'] for c in calls) for key in sorted({key for c in calls for key in c['reasons']})},
            'same_prefix_base_choices_changed':sum(v['selected_id']!=v['reference_id'] for v in events),
            'B4_conflict_rounds':sum(v['diagnosis']['conflict_kind']!='none' for v in events),
            'B4_removed_hypothesis_rounds':sum(bool(v['diagnosis']['revoked_ids']) for v in events),
            'knowledge_disabled_rounds':sum(not v['knowledge_enabled'] for v in events),
            'maximum_base_logei_loss':max(v['base_logei_loss'] for v in events),'guard_violations':sum(v['base_logei_loss']>v['budget_logei'] for v in events),
            'mean_absolute_correction_pp':st.mean(abs(v['mean_change_y'])*100 for v in events),
            'cap_binding_rounds':sum(abs(v['mean_change_y'])>=.5*v['selected_gp_sd_y']-1e-12 and abs(v['mean_change_y'])>0 for v in events),
            'prequential_RMSE_pp':pred,
            'joint_minus_GP_MSE_pp2':st.mean((v['measurement']['squared_errors']['GP_joint_KF']-v['measurement']['squared_errors']['GP'])*10000 for v in events),
            'joint_minus_offset_null_MSE_pp2':st.mean((v['measurement']['squared_errors']['GP_joint_KF']-v['measurement']['squared_errors']['GP_offset_null'])*10000 for v in events),
            'deployed_minus_GP_MSE_pp2':st.mean((v['measurement']['squared_errors']['GP_deployed']-v['measurement']['squared_errors']['GP'])*10000 for v in events),
            'final_trust_by_case':{str(r['case']):e.forecast(r['filter'],r['hypotheses'])['weights'] for r in group}}


def independent_filter_update(before,prior,feature,mu,sd,y):
    """Algebraic replay, separate implementation from the policy update."""
    x=np.array(prior['theta']);p=np.array(prior['covariance']);h=np.array(feature)
    r=sd*sd+.0009;innovation=y-mu-sum(h*x);variance=float(h@p@h+r);gain=p@h/variance
    posterior=x+gain*innovation
    # Algebraically equivalent P - P H H' P / S; compare to Joseph form.
    covariance=p-np.outer(p@h,p@h)/variance
    nullvar=prior['null_variance'];nullgain=nullvar/(nullvar+r)
    nullmean=prior['null_mean']+nullgain*(y-mu-prior['null_mean'])
    return posterior,covariance,nullmean,(1-nullgain)*nullvar,innovation,variance


def load_scores(cache,key):
    with np.load(cache.directory/(key+'.npz'),allow_pickle=False) as d:return d['scores'].copy(),json.loads(str(d['metadata']))


class MarginalReplayModel(Model):
    """Independent q=1 numerical replay from saved latent marginal moments.

    This model contains no knowledge predicates or GP-wrapper implementation.
    Off-diagonal candidate covariances do not enter single-candidate qLogEI.
    """
    num_outputs=1
    batch_shape=torch.Size([])
    def __init__(self):super().__init__();self.mean=None;self.variance=None
    def posterior(self,X,**kwargs):
        assert X.shape[-2]==1 and X.shape[0]==len(self.mean)
        return GPyTorchPosterior(MultivariateNormal(self.mean[:,None],torch.diag_embed(self.variance[:,None])))


def marginal_scores(mu_std,sd_std,change_std,best_std,mc_seed,z):
    model=MarginalReplayModel();acq=qLogExpectedImprovement(model=model,best_f=best_std,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc_seed))
    values=[]
    with torch.no_grad():
        for start in range(0,len(mu_std),256):
            end=start+256;model.mean=torch.tensor(mu_std[start:end]+change_std[start:end],dtype=torch.double)
            model.variance=torch.tensor(sd_std[start:end]**2+1e-6,dtype=torch.double)
            values.append(acq(z[start:end].unsqueeze(1)).reshape(-1))
    return torch.cat(values).numpy()


def audit_case(case):
    pool=s.inputs(case);ids=pool['ids'];z=np.array(pool['z']);ix={k:i for i,k in enumerate(ids)};source=s.make_cache(case,pool)
    # Separate cache for offline alternatives; all formal receipts stay fixed.
    audit=e.KnowledgeCache(pool['z'],ids,s.ROOT/'audit_cache'/f'c{case}_s0',case,0,sources=[source.directory,*source.sources])
    checked=0;diagnostics=[];filter_max_error=0.;score_max_error=0.;numerical_scores=0
    for folder in ('runs','frozen'):
        row=s.read(s.ROOT/folder/f'c{case}_s0'/'state.json');state=e.initial_filter();hs=[];enabled=False
        if folder=='frozen' and row.get('frozen_initial_plan'):
            state,hs=e.apply_plan(state,[],row['frozen_initial_plan'],z);enabled=True
        for t,event in enumerate(row['decisions'],1):
            prefix=row['history'][:t+3];available=np.array([k not in {h['candidate_id'] for h in prefix} for k in ids]);j=ix[event['selected_id']]
            if folder=='runs':
                call=next((c for c in row['calls'] if c['iteration']==t),None)
                if call:
                    if call['status']=='ACCEPTED':state,hs=e.apply_plan(state,hs,call['plan'],z);enabled=True
                    else:enabled=False
            assert hs==event['hypotheses'] and state==event['filter_before']
            prior=e.forecast(state,hs);assert prior==event['filter_prior']
            weights=prior['weights'] if enabled else {h['id']:0. for h in hs};assert weights==event['trust']
            base,bmeta=load_scores(source,event['base_score_key']);guide,gmeta=load_scores(source,event['guided_score_key'])
            ordered=sorted(prefix,key=lambda h:h['candidate_id']);fit=digest({'history':ordered,'case':case,'seed':0,'recipe':source.recipe})
            assert fit==event['fit_key']==bmeta['fit_key'] and gmeta['fit_key']==fit
            assert bmeta['mc_seed']==38102026+case*10000+len(prefix)-3
            reference=e.common.argmax(base,available,ids);budget=float(np.log(4)*(16-t)/15);eligible=available&((base[reference]-base)<=budget)
            diagnosis=e.diagnose(base,z,hs,weights,budget,~available,ids)
            assert json.loads(json.dumps(diagnosis.to_dict()))==event['diagnosis']
            chosen=e.common.argmax(guide,eligible,ids) if diagnosis.retained_ids else reference
            assert chosen==j and ids[reference]==event['reference_id'] and event['base_logei_loss']==float(base[reference]-base[j])<=budget
            if diagnosis.retained_ids:
                spec={'hypotheses':[{**e.canonical(h),'pool_fraction':h['pool_fraction']} for h in hs],'weights':weights,'retained':list(diagnosis.retained_ids)}
                assert gmeta['knowledge']==spec
            stats=audit.posterior_stats(prefix)
            assert abs(float(stats['mu_y'][j])-event['selected_gp_mu_y'])<1e-12 and abs(float(stats['sd_predictive_y'][j])-event['selected_predictive_sd_y'])<1e-12
            # Independent explicit feature construction, not calling features().
            feature=np.zeros(5);feature[0]=1.;raw_change=0.;change_all=np.zeros(len(ids))
            for h in hs:
                if h['variable']=='catalyst':mask=np.isin(np.floor(z[:,3]*8).astype(int),h['categories'])
                else:
                    k=e.VARIABLES.index(h['variable']);mask=(z[:,k]>=h['bounds'][0])&(z[:,k]<=h['bounds'][1])
                assert float(mask.mean())==h['pool_fraction']
                value=h['effect_pp']/100*(float(mask[j])-mask.mean());feature[e.SLOTS.index(h['id'])+1]=value
                if h['id'] in diagnosis.retained_ids:
                    raw_change+=weights[h['id']]*value
                    change_all+=weights[h['id']]*h['effect_pp']/100*(mask.astype(float)-mask.mean())
            delta=float(np.clip(raw_change,-.5*stats['sd_y'][j],.5*stats['sd_y'][j]))
            assert abs(delta-event['mean_change_y'])<1e-12 and np.allclose(feature,event['measurement']['feature'],atol=1e-14)
            change_all=np.clip(change_all,-.5*stats['sd_y'],.5*stats['sd_y'])
            _,fit_value=audit.model(prefix);mc=bmeta['mc_seed']
            replay_base=marginal_scores(stats['mu_std'],stats['sd_std'],np.zeros(len(ids)),fit_value['best_std'],mc,audit.z)
            replay_guide=marginal_scores(stats['mu_std'],stats['sd_std'],change_all/float(fit_value['std']),fit_value['best_std'],mc,audit.z)
            error=max(float(np.max(np.abs(replay_base-base))),float(np.max(np.abs(replay_guide-guide))))
            score_max_error=max(score_max_error,error);assert error<1e-8,(case,t,error)
            assert e.common.argmax(replay_base,available,ids)==reference and (e.common.argmax(replay_guide,eligible,ids) if diagnosis.retained_ids else reference)==j
            numerical_scores+=2*len(ids)
            m=event['measurement'];y=row['history'][t+3]['y'];assert y==m['observed_y']
            theta,cov,nm,nv,innovation,variance=independent_filter_update(state,prior,feature,event['selected_gp_mu_y'],event['selected_predictive_sd_y'],y)
            after=event['filter_after'];err=max(np.max(np.abs(theta-after['theta'])),np.max(np.abs(cov-after['covariance'])),abs(nm-after['null_mean']),abs(nv-after['null_variance']))
            filter_max_error=max(filter_max_error,float(err));assert err<1e-12 and np.linalg.eigvalsh(cov).min()>-1e-12
            preds={'GP':m['prequery_mu_y'],'GP_offset_null':m['prequery_mu_y']+prior['null_mean'],'GP_joint_KF':m['prequery_mu_y']+float(feature@np.array(prior['theta'])),'GP_deployed':m['prequery_mu_y']+delta}
            for k,pred in preds.items():assert abs(pred-m['predictions'][k])<1e-12 and abs((y-pred)**2-m['squared_errors'][k])<1e-12
            assert abs(innovation-m['innovation'])<1e-12 and abs(variance-m['innovation_variance'])<1e-12
            state=deepcopy(after);checked+=1
            if folder=='runs':
                active=[h['id'] for h in hs if weights.get(h['id'],0)>e.PARAMETERS['minimum_trust']]
                if active:ungated,_=audit.knowledge_score(prefix,hs,weights,active)
                else:ungated=base
                no_b4=e.common.argmax(ungated,eligible,ids) if active else reference
                ungated_j=e.common.argmax(ungated,available,ids) if active else reference
                fixed={h['id']:float(1-np.sqrt(.25+.0025)) for h in hs} if enabled else {h['id']:0. for h in hs}
                df=e.diagnose(base,z,hs,fixed,budget,~available,ids)
                fixed_score=audit.knowledge_score(prefix,hs,fixed,df.retained_ids)[0] if df.retained_ids else base
                no_kf=e.common.argmax(fixed_score,eligible,ids) if df.retained_ids else reference
                diagnostics.append({'case':case,'iteration':t,'selected_id':ids[j],'no_B4_id':ids[no_b4],'fixed_initial_trust_id':ids[no_kf],
                    'ungated_id':ids[ungated_j],'ungated_base_logei_loss':float(base[reference]-base[ungated_j]),'budget':budget,
                    'no_B4_changed':bool(no_b4!=j),'no_KF_changed':bool(no_kf!=j),'ungated_guard_violation':bool(base[reference]-base[ungated_j]>budget)})
        assert state==row['filter']
    return {'case':case,'choices_and_filter_updates':checked,'maximum_independent_filter_error':filter_max_error,
            'independently_recomputed_qLogEI_scores':numerical_scores,'maximum_independent_qLogEI_error':score_max_error,
            'diagnostics':diagnostics,'audit_cache':audit.counters,'operator_compiles_in_loop':sum(s.JIT.values())}


def report():
    protocol=s.check_protocol();groups=rows();assert all(len(v)==4 and all(r['status']=='COMPLETE' for r in v) for v in groups.values())
    tables={};paired={};mechanism={}
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4)),*[(f'Case{c}',(c,)) for c in (1,2,3,4)]]:
        subset={m:[r for r in group if r['case'] in cases] for m,group in groups.items()}
        tables[scope]={m:summary(v) for m,v in subset.items()}
        paired[scope]={m:metrics.compare(subset[s.METHODS[0]],subset[m]) for m in groups if m!=s.METHODS[0]}
        mechanism[scope]={m:mechanisms(subset[m]) for m in s.METHODS}
    with ProcessPoolExecutor(max_workers=2,initializer=s.worker_init) as executor:audits=list(executor.map(audit_case,(1,2,3,4)))
    diagnostics=[v for a in audits for v in a['diagnostics']]
    diagnostic_summary={'same_new_prefix_decisions':len(diagnostics),'fixed_initial_trust_changed_choices':sum(v['no_KF_changed'] for v in diagnostics),
                        'no_B4_changed_choices':sum(v['no_B4_changed'] for v in diagnostics),'ungated_suggestions_over_budget':sum(v['ungated_guard_violation'] for v in diagnostics),
                        'boundary':'Offline same-prefix choices only. These are not independently run ablations, not AUC, and not causal effects of individual modules.'}
    result={'protocol_sha256':protocol['protocol_sha256'],'tables':tables,'new_minus_previous':paired,'mechanism':mechanism,'usage':usage(),
            'diagnostic_summary':diagnostic_summary,'planned_trajectories':8,'completed_trajectories':8,'logical_queries':sum(len(r['history']) for m in s.METHODS for r in groups[m]),
            'raw_metrics':[{'method':m,'case':r['case'],'seed':0}|metrics.metric(r) for m,group in groups.items() for r in group],
            'limits':protocol['limits']+['Natural-prior regional/LLINBO references used older current-assistant knowledge; new DeepSeek receives GP summaries and has different cadence. System comparison, not a model-only or knowledge-matched algorithm comparison.',
                                      'The frozen-initial control retains Kalman filtering and event-independent B4; it removes only subsequent LLM knowledge refresh.']}
    s.write(s.OUT/'audit_cases.json',audits);s.write(s.OUT/'results.json',result)
    with (s.OUT/'metrics.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','method','n','SearchAUC_pp','final_yield_percent'])
        for scope,table in tables.items():
            for method,v in table.items():writer.writerow([scope,method,v['n'],v['SearchAUC_mean'],v['final_yield_mean']])
    figures(result)
    return {'primary':{m:{k:round(v[k],4) for k in ('SearchAUC_mean','final_yield_mean')} for m,v in tables['primary'].items()},'usage':result['usage'],'diagnostic_summary':diagnostic_summary}


def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=(s.METHODS[0],s.METHODS[1],'qLogEI_B0','DeepSeek_score_adaptive','LGBO_region_natural_reference','B4_LGBO_natural_reference','LLINBO_matched_natural_reference')
    labels=('KF + event + B4','KF + frozen initial knowledge','qLogEI BO','DeepSeek direct score','Regional LGBO (old knowledge)','Soft B4 (old knowledge)','LLINBO matched (old knowledge)')
    colors=('#0072B2','#777777','#000000','#D55E00','#009E73','#CC79A7','#E69F00')
    fig,axes=plt.subplots(1,4,figsize=(15,4.1),layout='constrained')
    for c,ax in enumerate(axes,1):
        for m,label,color in zip(chosen,labels,colors):ax.plot(range(5,21),result['tables'][f'Case{c}'][m]['mean_curve'],label=label,color=color,lw=1.6,ls='--' if 'reference' in m else '-')
        ax.set(title=f'Case {c}, seed 0',xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Incumbent yield (%)');axes[-1].legend(fontsize=6,loc='lower right');fig.suptitle('Fixed pool / initialization / 4+16 queries: single-seed exploratory system comparison',fontsize=10)
    fig.savefig(s.OUT/'comparison.png',dpi=180);fig.savefig(s.OUT/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,3.4),layout='constrained')
    for c,ax in enumerate(axes,1):
        row=s.read(s.ROOT/'runs'/f'c{c}_s0'/'state.json')
        for k,key in enumerate(e.SLOTS):
            versions=sorted({h['version'] for v in row['decisions'] for h in v['hypotheses'] if h['id']==key})
            for i,version in enumerate(versions):
                events=[v for v in row['decisions'] if any(h['id']==key and h['version']==version for h in v['hypotheses'])]
                ax.plot([v['iteration'] for v in events],[v['trust'][key] for v in events],marker='.',ms=4,color=f'C{k}',label=key if i==0 else None)
        for call in row['calls']:ax.axvline(call['iteration'],color='#555555',alpha=.25,lw=1)
        ax.set(title=f'Case {c}',xlabel='Added query',ylim=(-.03,1.03),xticks=[1,4,8,12,16]);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('One-SD lower coefficient (clipped)');axes[-1].legend(fontsize=7);fig.suptitle('Adaptive knowledge trust; vertical lines mark LLM calls; revised ids reset their coefficient',fontsize=10)
    fig.savefig(s.OUT/'trust.png',dpi=170);plt.close(fig)


def verify(config_path='../config.txt'):
    p=s.check_protocol();groups=rows();freeze=s.read(s.OUT/'adaptive_frozen.json');assert freeze['frozen_control_states_before_start']==0
    assert s.filehash(s.OUT/'qualification.json')==p['qualification_sha256']
    assert sum(len(r['history']) for m in s.METHODS for r in groups[m])==160
    for key,value in p['input_sha256'].items():assert s.filehash(s.PARENT/'inputs'/key)==value
    contexts=0;requests=0
    for adaptive,control in zip(groups[s.METHODS[0]],groups[s.METHODS[1]]):
        case=adaptive['case'];pool=s.inputs(case);z=np.array(pool['z']);values=np.load(s.PARENT/'inputs'/f'case{case}'/'oracle.npz')['values'];ix={k:i for i,k in enumerate(pool['ids'])}
        assert s.filehash(s.ROOT/'runs'/adaptive['run_id']/'state.json')==freeze['states'][adaptive['run_id']]
        for row in (adaptive,control):
            assert row['seed']==0 and row['status']=='COMPLETE' and len(row['history'])==20 and len(row['decisions'])==16
            assert len({h['candidate_id'] for h in row['history']})==20 and row['operator_compiles_in_loop']==0
            assert row['protocol_sha256']==p['protocol_sha256'] and [h['candidate_id'] for h in row['history'][:4]]==pool['initial_ids']['0']
            for h in row['history']:assert h['y']==float(values[ix[h['candidate_id']]])
        assert adaptive['history'][:4]==control['history'][:4] and not control['calls']
        if adaptive['calls'][0]['status']=='ACCEPTED':assert control['frozen_initial_plan']==adaptive['calls'][0]['plan'] and control['decisions'][0]['selected_id']==adaptive['decisions'][0]['selected_id']
        assert len(adaptive['calls'])<=4
        call_rounds=[]
        for call in adaptive['calls']:
            t=call['iteration'];event=adaptive['decisions'][t-1];directory=s.ROOT/'requests'/call['context_sha256'];context=s.read(directory/'context.json');raw=deepcopy(context);key=raw.pop('context_sha256')
            assert digest(raw)==key==call['context_sha256'] and context['history']==adaptive['history'][:t+3]
            prior=e.forecast(event['filter_before_call'],event['hypotheses_before_call']);source=s.make_cache(case,pool)
            base,_=load_scores(source,event['base_score_key']);stats=source.posterior_stats(context['history']);queried=np.array([k in {h['candidate_id'] for h in context['history']} for k in pool['ids']])
            # enabled before this call is the previous round's state; the first
            # call has no active knowledge regardless of this flag.
            enabled=(adaptive['decisions'][t-2]['knowledge_enabled'] if t>1 else False)
            diag=e.diagnose(base,z,event['hypotheses_before_call'],prior['weights'] if enabled else {},e.common.budget_at(4,t),queried,pool['ids'])
            reasons=e.event_reasons(t,context['history'],adaptive['decisions'][:t-1],event['hypotheses_before_call'],diag,call_rounds)
            assert reasons==call['reasons'];call_rounds.append(t)
            minimal={'task':adaptive['task'],'history':context['history'],'decisions':adaptive['decisions'][:t-1],'hypotheses':event['hypotheses_before_call'],'filter':event['filter_before_call']}
            public=s.public_request(minimal,pool,stats,base,diag,reasons)
            assert json.loads(json.dumps(public))==context['public']
            assert len(public['observations'])==t+3 and all('yield_percent' not in v for v in public['gp_top3_per_catalyst'])
            if call['status']=='ACCEPTED':
                valid=s.read(directory/'validated.json');attempt=valid['accepted_attempt'];assert attempt<=8 and valid['received_responses']<=3
                response=directory/f'attempt{attempt:02d}'/'response.json';answer=s.read(response)
                assert s.filehash(response)==valid['response_sha256'] and answer['model']=='deepseek-flash' and not answer['retained_raw_reasoning']
                parsed=e.validate_plan(json.loads(answer['choices'][0]['message']['content']),event['hypotheses_before_call'],len(context['history']),z)
                assert parsed==valid['plan']==call['plan']
                for n in range(1,attempt+1):
                    body=s.read(directory/f'attempt{n:02d}'/'request.json');assert {k:v for k,v in body.items() if k!='messages'}==s.PARAMETERS
                    assert body['messages'][:2]==s.messages_for(public);requests+=1
            contexts+=1
    for path,value in freeze['requests'].items():assert s.filehash(path)==value
    audits=s.read(s.OUT/'audit_cases.json');assert sum(a['choices_and_filter_updates'] for a in audits)==128 and all(a['operator_compiles_in_loop']==0 for a in audits)
    protected={};trees={}
    for label,path in s.PROTECTED.items():
        inv=s.read(path)
        for f in inv['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inv['files'])
        for key,expected in inv.items():
            if key.endswith('cache_tree') or key=='score_tree':
                folder='scores' if key=='score_tree' else key.removesuffix('_tree');actual=metrics.tree(Path('data/runs')/f'suzuki_{label}'/folder)
                assert actual=={k:expected[k] for k in actual},label+' '+folder;trees[label+'_'+folder]=actual
    result=s.read(s.OUT/'results.json')
    for scope,cases in [('all_four',(1,2,3,4)),('primary',(2,3,4))]:
        for method,group in groups.items():
            chosen=[r for r in group if r['case'] in cases];auc=st.mean(sum(max(h['y'] for h in r['history'][:n])*100 for n in range(5,21))/16 for r in chosen)
            assert abs(auc-result['tables'][scope][method]['SearchAUC_mean'])<1e-10
            for i,row in enumerate(group):assert row['history'][:4]==groups[s.METHODS[0]][i]['history'][:4]
    cfg=s.api.load_config(config_path);scanned=0
    for root in (s.ROOT,s.OUT):
        for path in root.rglob('*'):
            if not path.is_file() or path.suffix not in ('.json','.csv','.log','.xml','.md') or any(part.startswith('pytest_tmp') for part in path.parts):continue
            assert cfg.api_key not in path.read_text(encoding='utf8'),'credential leaked';scanned+=1
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'logical_queries':160,'independently_replayed_choices_and_KF_updates':128,
             'maximum_independent_filter_error':max(a['maximum_independent_filter_error'] for a in audits),'LLM_contexts_verified':contexts,'API_attempts_verified':requests,
             'independently_recomputed_qLogEI_scores':sum(a['independently_recomputed_qLogEI_scores'] for a in audits),
             'maximum_independent_qLogEI_error':max(a['maximum_independent_qLogEI_error'] for a in audits),
             'operator_compiles_in_loop':0,'protected_files_unchanged':protected,'protected_trees_unchanged':trees,'credential_scan_files':scanned,'credential_leaks':0,
             'results_sha256':s.filehash(s.OUT/'results.json'),'boundary':'Integrity and implementation qualification, not replicated superiority or a chemical-knowledge truth certificate.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def release():
    p=s.check_protocol();validation=s.read(s.OUT/'validation_receipt.json');assert validation['integrity_status']=='PASS'
    assert s.filehash(s.OUT/'results.json')==validation['results_sha256']
    suites=list(ET.parse(s.OUT/'component_tests.xml').getroot().iter('testsuite'));tests={k:sum(int(x.attrib.get(k,0)) for x in suites) for k in ('tests','failures','errors','skipped')}
    assert tests['tests']>=20 and tests['failures']==tests['errors']==0
    files=[Path(__file__),Path(s.__file__),Path(e.__file__),Path('chem/benchmarks/suzuki_kalman_knowledge.py'),Path('tests/test_suzuki_kalman_knowledge.py'),Path('docs/suzuki_kalman_knowledge.md')]
    files += [f for f in s.ROOT.rglob('*') if f.is_file() and not any(part in ('cache','audit_cache','qualification_cache') for part in f.parts)]
    files += [f for f in s.OUT.rglob('*') if f.is_file() and not any(part.startswith('pytest_tmp') for part in f.parts) and f.name not in ('release_inventory.json','release_receipt.json')]
    cfg=s.api.load_config('../config.txt');scanned=0
    for path in set(files):
        if path.suffix in ('.json','.xml','.csv','.md','.py','.log'):
            assert cfg.api_key not in path.read_text(encoding='utf8'),'credential leaked into release';scanned+=1
    inv={'protocol_sha256':p['protocol_sha256'],'files':[{'path':f.as_posix(),'bytes':f.stat().st_size,'sha256':s.filehash(f)} for f in sorted(set(files))],
         'cache_tree':metrics.tree(s.ROOT/'cache'),'audit_cache_tree':metrics.tree(s.ROOT/'audit_cache'),'qualification_cache_tree':metrics.tree(s.ROOT/'qualification_cache')}
    inv['file_count']=len(inv['files']);s.write(s.OUT/'release_inventory.json',inv)
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'component_tests':tests,'completed_trajectories':8,'logical_queries':160,
             'operator_compiles_in_loop':0,'usage':usage(),'credential_scan_files':scanned,'credential_leaks':0,
             'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),'validation_sha256':s.filehash(s.OUT/'validation_receipt.json')}
    s.write(s.OUT/'release_receipt.json',receipt);return receipt
