"""V10 audit without API calls, oracle access, or model fitting.

Independent numerical checks: Brier/KF/lifecycle, ranking, GP probabilities and
direct quadrature of saved Gaussian fantasy maxima. Proposal/public-payload and
full-state replay additionally reuse production validators (explicit boundary).
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from math import erfc,exp,pi,sqrt
from pathlib import Path
import argparse,json,sys,warnings
import numpy as np
from scipy.integrate import quad,IntegrationWarning

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from chem.benchmarks.knowledge_llm import digest
from knowledge_information_v10 import study,engine,llm,proposals
from knowledge_rank_v9 import kf
from audit_knowledge_rank_v9 import independent_kf,rank,gp_hash,event_allowed,close,require,scope


def independent_KG(mu,cov,variance):
    if variance==0:return 0.,0.
    beta=cov/sqrt(variance);base=float(np.max(mu));peak=int(np.argmax(mu));beta=beta-beta[peak]
    shifted=mu-base
    # Independent divide-and-conquer discovery of active kink locations. A
    # grid-only quadrature can miss a narrow interval of a third active line
    # beside the crossing of two sampled winners, despite a small error estimate.
    # This algorithm never sorts slopes or uses the production monotone hull.
    cuts=[];pending=[(int(np.argmax(shifted-12*beta)),int(np.argmax(shifted+12*beta)),-12.,12.)]
    steps=0;envelope_tolerance=0.
    while pending:
        a,b,left,right=pending.pop();steps+=1
        if steps>4*len(mu)+10:raise ArithmeticError('Independent envelope subdivision did not terminate')
        if a==b:continue
        if beta[a]==beta[b]:raise ArithmeticError('Inconsistent parallel envelope endpoints')
        z=float((shifted[b]-shifted[a])/(beta[a]-beta[b]))
        if z<left-1e-10 or z>right+1e-10:raise ArithmeticError('Envelope crossing outside interval')
        z=min(right,max(left,z));values=shifted+beta*z;c=int(np.argmax(values))
        tolerance=16*np.finfo(float).eps*max(1.,abs(float(values[c])))
        envelope_tolerance=max(envelope_tolerance,tolerance)
        if c in (a,b) or values[c]-max(values[a],values[b])<=tolerance:
            if -12.<z<12.:cuts.append(z)
        else:
            pending.append((c,b,z,right));pending.append((a,c,left,z))
    def integrand(z):return float(np.max(shifted+beta*z))*exp(-z*z/2)/sqrt(2*pi)
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        value,error=quad(integrand,-12.,12.,points=sorted(set(cuts)),epsabs=2e-13,epsrel=2e-11,limit=max(500,len(cuts)+50))
    # The omitted standard-normal tail beyond12 has probability<4e-33.
    tail_bound=4e-31*max(1.,float(np.max(abs(mu))),float(np.max(abs(beta))))
    return max(0.,value),error+tail_bound+envelope_tolerance


def audit_task(root,task):
    root=Path(root);protocol=study.check(root);base=study.directory(root)
    path=base/'runs'/f'{task}_information';state=study.old.read(path/'state.json')
    pool=study.gpmod.load_task(root,task,seed=0)
    require(state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16,'complete4+16')
    require(state['protocol_sha256']==protocol['protocol_sha256'] and state['seed']==0 and state['arm']=='information','registered task state')
    require(len({r['candidate_id'] for r in state['history']})==20,'unique observations')
    control=study.old.read(study.v9.directory(root)/'runs'/f'{task}_bo/state.json')
    require(state['history'][:4]==control['history'][:4],'matched initial4')
    init=study.old.read(path/'initial_commit.json')
    require(init['commit_sha256']==digest({k:v for k,v in init.items() if k!='commit_sha256'})==state['initial_commit_sha256'],'initial hash')
    require(init['initial_ids']==list(pool.initial_ids),'input-only initial IDs')
    counts=dict(task=task,rounds=0,forecasts=0,KG_columns=0,KG_max_abs_error=0.,information=0,dissent=0,outside_top5=0,
        KG_boundary_checks=0)
    previous=None
    for t,d in enumerate(state['decisions'],1):
        c=study.old.read(path/'rounds'/f'{t:02d}_commit.json');pre=c['prestate'];obs=state['history'][t+3]
        require(c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})==d['commit_sha256'],'commit hash')
        require(c['protocol_sha256']==protocol['protocol_sha256'] and c['query']==t,'registered round')
        require(c['prestate_sha256']==digest(pre) and c['history_sha256']==digest(pre['history']),'prequery hashes')
        require(pre['history']==state['history'][:t+3] and len(pre['decisions'])==t-1,'causal own prefix')
        if previous is not None:require(digest({k:v for k,v in previous.items() if k!='cache_counters'})==
            digest({k:v for k,v in pre.items() if k!='cache_counters'}),'feedback to next prestate')
        with np.load(path/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        require(gp_hash(a)==c['GP_sha256']==c['GP_sha256_after_ranking'],'unchanged GP arrays')
        available=np.array([key not in {r['candidate_id'] for r in pre['history']} for key in pool.ids])
        np.testing.assert_array_equal(a['available'],available)
        ties=study.gpmod.semantic_priorities(pool,t,0);np.testing.assert_array_equal(a['tie_priority'],ties)
        candidates=np.flatnonzero(available);ordered=candidates[np.lexsort((ties[candidates],-a['base_logei'][candidates]))][:5]
        require(c['ordered_ids']==[pool.ids[j] for j in ordered] and c['BO_reference_id']==pool.ids[ordered[0]],'native top5 and BO')
        context=dict(stats={k:a[k] for k in ('mu_y','sd_y','sd_predictive_y')},base_logei=a['base_logei'],available=available,
            tie_priority=ties,eligible=a['eligible_V8'],epsilon=c['V8_epsilon'])
        refresh=event_allowed(pool,pre,ordered,t);require(refresh==c['refresh'],'independent edit trigger')
        hs=deepcopy(pre['hypotheses'])
        if refresh['allowed']:
            key=c['proposal_context_sha256'];req=study.old.read(base/'requests'/key/'context.json')
            validated=study.old.read(base/'requests'/key/'validated.json')
            public=llm.proposal_public(pool,pre,t,refresh,ties)
            require(req==dict(system=llm.PROPOSAL_SYSTEM,public=public,model=protocol['model'],reasoning_effort=protocol['reasoning_effort']),
                'proposal input-only exact public context')
            require(digest(req)==key==validated['context_sha256'] and validated['raw']==c['proposal_raw'],'proposal request custody')
            result=llm.validate_proposal(c['proposal_raw'],pool,pre,t,available,ties)
            require(result==validated['plan'] and result['reviews']==c['reviews'],'proposal validation consistency')
            hs=result['hypotheses']
        else:require(c['proposal_raw'] is None and not c['reviews'],'no non-event knowledge edits')
        require(hs==c['hypotheses_at_commit'],'immutable proposal state')
        for h in hs:proposals.validate(h)
        _,display,mapped=engine.menu(pool,pre,hs,context,t);ids=[pool.ids[j] for j in display]
        require(ids==c['display_ids'],'full-pool projection/menu replay')
        key=c['LLM_context_sha256'];req=study.old.read(base/'requests'/key/'context.json');valid=study.old.read(base/'requests'/key/'validated.json')
        public=llm.forecast_public(pool,pre,hs,ids,t)
        require(req==dict(system=llm.FORECAST_SYSTEM,public=public,model=protocol['model'],reasoning_effort=protocol['reasoning_effort']),
            'forecast uses own prefix, neutral menu and fixed known references')
        require(digest(req)==key==valid['context_sha256'] and valid['raw']==c['LLM_raw'],'forecast request custody')
        probabilities=llm.validate_forecast(c['LLM_raw'],hs,ids)['probabilities'];require(probabilities==c['forecast_probabilities'],'committed probabilities')
        fallback,ranks=rank(pool,ordered,ties,hs,probabilities)
        require(pool.ids[fallback]==c['fallback_id'],'independent V9 fallback selection')
        close(ranks['combined_rank_scores'],c['ranking']['combined_rank_scores'],'unchanged fallback rank formula')
        kg={};errors={};kgfile=path/'rounds'/f'{t:02d}_KG.npz'
        require(kgfile.exists()==(t<=8 and pre['information_used']<3 and bool(mapped)),'expected KG calculation only')
        if kgfile.exists():
            with np.load(kgfile,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['mu_y'],a['mu_y'])
                for k,j in enumerate(z['query_indices']):
                    np.testing.assert_allclose(z['observation_variance'][k],a['sd_predictive_y'][j]**2,atol=1e-14,rtol=1e-12)
                    value,error=independent_KG(z['mu_y'],z['cross_covariance'][:,k],float(z['observation_variance'][k]))
                    recorded=c['KG_values'][pool.ids[j]];delta=abs(recorded-value)
                    require(delta<=max(4e-10,5*error),'independent saved-covariance KG quadrature')
                    kg[int(j)]=recorded;errors[int(j)]=max(delta,error)
                    counts['KG_max_abs_error']=max(counts['KG_max_abs_error'],delta);counts['KG_columns']+=1
        require(set(c['KG_values'])=={pool.ids[j] for j in kg},'all saved KG columns audited')
        rows=engine.probe_rows(pool,pre,hs,context,t,probabilities,fallback,mapped,kg)
        require(digest(rows)==digest(c['probe_reviews']),'all probe gate/priority consistency replay')
        # Independent gate and winner calculation on recorded GP values.
        choices=[];pilots=[]
        for r in rows:
            j=r['index'];h=next(h for h in hs if (h['id'],h['version'])==(r['id'],r['version']))
            w=h['kf']['weight'] if h['kf']['status']=='active' else 0.
            eligible=t<=8 and pre['information_used']<3 and w>0 and h['probe_signature'] not in pre['tested_probe_signatures']
            eligible=eligible and j!=fallback and a['base_logei'][ordered[0]]-a['base_logei'][j]<=np.log(4)+1e-12
            sd=float(a['sd_predictive_y'][j]);g=.5*erfc(-(a['mu_y'][j]-h['reference_y'])/(sqrt(2)*sd)) if sd else float(a['mu_y'][j]>h['reference_y'])
            p=probabilities[h['id']][pool.ids[j]];close(g,r['p_GP'],'independent probe GP probability')
            if eligible and j in kg:
                delta=kg[j]-kg[fallback];tol=max(1e-12,1e-10*max(abs(kg[j]),abs(kg[fallback])))
                if delta>tol:choices.append((w*delta,j,r['probe_signature']))
                if delta>tol and delta-tol<=errors[j]+errors[fallback]:counts['KG_boundary_checks']+=1
            if eligible and not pre['dissent_used'] and abs(p-g)>=.2:pilots.append((w*abs(p-g),j,r['probe_signature']))
        mode=None;winner=fallback
        if choices or pilots:
            choices,mode=(choices,'information_KG') if choices else (pilots,'information_disagreement')
            _,winner,_=min(choices,key=lambda z:(-z[0],-a['base_logei'][z[1]],int(ties[z[1]]),z[2]))
        require(c['selected_id']==pool.ids[winner]==obs['candidate_id']==d['selected_id'],'independent final selection')
        require(c['information_override']==(mode is not None),'override flag')
        if mode:require(c['mode']==mode,'information decision route')
        require(available[winner],'only unqueried point revealed')
        expected_forecasts={(h['id'],h['version']) for h in hs if h['kf']['status']!='retracted'
            and scope(pool,h,winner)}
        require(expected_forecasts=={(s['prediction']['id'],s['prediction']['version']) for s in c['predictions']},'exact prospective scope coverage')
        for sh in c['predictions']:
            pred=sh['prediction'];require(digest(pred)==sh['prediction_sha256'],'prospective prediction hash')
            h=next(h for h in hs if (h['id'],h['version'])==(pred['id'],pred['version']))
            require(pred['p_LLM']==probabilities[h['id']][pool.ids[winner]],'prequery probability scored')
            require(pred['point_id']==obs['candidate_id'] and pred['history_sha256']==digest(pre['history']),'causal scored forecast')
            ref=next(r for r in pre['history'] if r['candidate_id']==h['reference_id']);require(ref['y']==pred['reference_y'],'known reference outcome')
            sd=a['sd_predictive_y'][winner];g=.5*erfc(-(a['mu_y'][winner]-ref['y'])/(sqrt(2)*sd)) if sd else float(a['mu_y'][winner]>ref['y'])
            close(g,pred['p_GP'],'independent selected-point GP probability')
            post,measure=independent_kf(h['kf'],pred['p_LLM'],g,bool(obs['y']>ref['y']),t)
            event=next(e for e in d['KF_feedback'] if (e['id'],e['version'])==(h['id'],h['version']))
            require(event['before']==h['kf'] and event['event_id']==digest({k:v for k,v in event.items() if k!='event_id'}),'KF prior/hash')
            for k,v in post.items():
                if isinstance(v,float):close(event['after'][k],v,'independent KF posterior')
                else:require(event['after'][k]==v,'independent lifecycle')
            for k,v in measure.items():
                if isinstance(v,float):close(event[k],v,'independent KF measurement')
                else:require(event[k]==v,'independent KF reason')
            counts['forecasts']+=1
        expected=deepcopy(pre);engine.feedback(expected,c,obs)
        require(expected['decisions'][-1]==d,'full feedback receipt replay')
        previous=expected;counts['rounds']+=1;counts['information']+=int(c['information_override'])
        counts['dissent']+=int(c['mode']=='information_disagreement');counts['outside_top5']+=int(winner not in ordered)
    require(digest({k:v for k,v in previous.items() if k!='cache_counters'})==digest({k:v for k,v in state.items() if k!='cache_counters'}),'final full state')
    require(counts['information']<=3 and counts['dissent']<=1,'query quotas')
    require(state['cache_counters']['operator_compiles_in_loop']==0,'no operator compilation inside loop')
    return counts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    p=study.check(ROOT);failures=[];rows=[]
    for name,sha in p['protected_V9_sha256'].items():
        if study.old.sha(ROOT/name)!=sha:failures.append('Frozen V9/control drift: '+name)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs={ex.submit(audit_task,str(ROOT),t):t for t in p['tasks']}
        for f,t in fs.items():
            try:rows.append(f.result())
            except Exception as error:failures.append(t+': '+type(error).__name__+': '+str(error))
    result=dict(status='PASS_LISTED_CHECKS' if not failures and len(rows)==8 else 'FAIL',rows=rows,failures=failures,
        rounds=sum(r['rounds'] for r in rows),forecasts=sum(r['forecasts'] for r in rows),
        KG_columns=sum(r['KG_columns'] for r in rows),max_KG_error=max((r['KG_max_abs_error'] for r in rows),default=None),
        boundary='Independent equations/rank/gates/KG quadrature; proposal, projection and whole-state replay reuse production functions. No independent GP refit or external timestamp attestation.',
        protected_control_files=len(p['protected_V9_sha256']),new_oracle_queries=0,new_API_calls=0,new_GP_fits=0)
    study.write(ROOT/'experiments/reports/knowledge_loop_20261101/v10_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
