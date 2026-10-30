"""Frozen development/test orchestration for matched B4 regional guidance."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from chem.decision.replay import write_json_atomic
from . import guarded_suzuki_engine as e
from .official_suzuki_adapter import encode_point,decode_z,modules,PARAMETERS,check_sources
from .suzuki_domain import FrozenSuzukiOracle,CATALYSTS,NUMERIC,LOW,HIGH,digest

ROOT=Path('data/runs/suzuki_guarded')
OUT=Path('research/suzuki_guarded')
NATIVE=Path('data/runs/suzuki_official')
RATIOS=(2,4,16)
SEEDS=tuple(range(5))
JIT={'script':0,'trace':0}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):write_json_atomic(path,value)
def scientific_hashes():return {p.name:filehash(p) for p in (Path(e.__file__),Path(__file__))}
def pointkey(p):return tuple([p['catalyst'],*[p[n] for n in NUMERIC]])


def _advice(bounds,confidence,assignment):
    clauses=[{'id':f'H{assignment[j]+1}','coordinate':j,'interval':[float(bounds[0][j]),float(bounds[1][j])]} for j in range(4)]
    return {'confidence':float(confidence),'clauses':sorted(clauses,key=lambda c:c['id'])}


def prepare_inputs():
    """Reference outcomes construct labeled stress priors, never policy scores."""
    OUT.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    core,prompt=modules('LGBO');space=core.ContinuousSpace.from_parameters(PARAMETERS)
    qualities=[];sources=[]
    for case in (1,2,3,4):
        folder=ROOT/'inputs'/f'case{case}';folder.mkdir(parents=True,exist_ok=True)
        task=read(Path('data/runs/chapter5/summit_cache')/f'case_{case}'/'task.json')
        points=list(task['candidates']);seen={pointkey(p) for p in points}
        for k,cat in enumerate(CATALYSTS):
            unit=qmc.Sobol(3,scramble=True,seed=27102026+case*10000+k).random_base2(9)
            corners=np.array([[i,j,l] for i in (0,1) for j in (0,1) for l in (0,1)])
            for u in np.concatenate([unit,corners]):
                point={'catalyst':cat,**dict(zip(NUMERIC,map(float,LOW+u*(HIGH-LOW))))}
                if pointkey(point) not in seen:points.append(point);seen.add(pointkey(point))
        initials={};parent_states={}
        for seed in SEEDS:
            s=read(NATIVE/'runs'/f'c{case}_s{seed}_LGBO'/'state.json');parent_states[seed]=s
            initials[str(seed)]=[h['point'] for h in s['history'][:4]]
            for p in initials[str(seed)]:
                if pointkey(p) not in seen:points.append(p);seen.add(pointkey(p))
        ids=[f'p{i:05d}' for i in range(len(points))];lookup={pointkey(p):i for i,p in enumerate(points)}
        z=np.array([encode_point(p) for p in points])
        # No outcome was used to generate the common candidate coordinates.
        y=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',case).predict(points)
        for seed,s in parent_states.items():
            for h in s['history'][:4]:
                i=lookup[pointkey(h['point'])];assert abs(y[i]-h['y'])*100<1e-4;y[i]=h['y']
        best=int(np.argmax(y));reference=z[best];lo=np.maximum(0.,reference-.30);hi=np.minimum(1.,reference+.30)
        cat=CATALYSTS.index(points[best]['catalyst']);lo[3]=cat/8;hi[3]=(cat+1)/8
        good=(lo.tolist(),hi.tolist())
        alternatives=[]
        for k in range(8):
            if k==cat:continue
            lb=lo.copy();ub=hi.copy();lb[3]=k/8;ub[3]=(k+1)/8
            mask=((z>=lb)&(z<=ub)).all(1)
            alternatives.append((float(y[mask].max()) if mask.any() else 2.,k,lb,ub))
        _,badcat,badlo,badhi=min(alternatives,key=lambda a:(a[0],a[1]))
        anchors=(z[:,3]>=lo[3])&(z[:,3]<=hi[3])&(z[:,1]>=lo[1])&(z[:,1]<=hi[1])
        anchors&=(abs(z[:,0]-reference[0])>.15)&(abs(z[:,2]-reference[2])>.15)
        indices=sorted(np.flatnonzero(anchors),key=lambda i:(float(y[i]),ids[i]))[:64]
        candidates=[]
        for i in indices:
            lb=lo.copy();ub=hi.copy()
            for j in (0,2):lb[j]=max(0,z[i,j]-.15);ub[j]=min(1,z[i,j]+.15)
            mask=((z>=lb)&(z<=ub)).all(1)
            candidates.append((float(y[mask].max()),ids[i],lb,ub))
        _,_,mixlo,mixhi=min(candidates,key=lambda a:(a[0],a[1]))
        priors={};init_ids={}
        for seed,s in parent_states.items():
            assignment=np.random.default_rng(29102026+case*10000+seed*100).permutation(4)
            answer=s['responses'][0]
            request=read(NATIVE/'requests'/answer['context_sha256']/'request.json')
            assert request['history']==s['history'][:4] and request['iteration']==1
            parsed=prompt.parse_assistant_response(answer['upstream_reply']);assert parsed['mode']=='region'
            actual_box=space.normalize_region(parsed['lb'],parsed['ub'])
            priors[str(seed)]={
                'correct':_advice(good,.7,assignment),
                'wrong_category':_advice((badlo,badhi),.7,assignment),
                'mixed':_advice((mixlo,mixhi),.7,assignment),
                'assistant_initial':_advice(actual_box,parsed['confidence'],assignment)}
            init_ids[str(seed)]=[ids[lookup[pointkey(p)]] for p in initials[str(seed)]]
            for regime,advice in priors[str(seed)].items():
                member=e.clause_membership(z,advice);mask=member.all(0)
                assert mask.any()
                qualities.append({'case':case,'seed':seed,'regime':regime,'region_candidates':int(mask.sum()),
                    'pool_optimum_pct':float(y.max()*100),'region_optimum_pct':float(y[mask].max()*100),
                    'region_mean_pct':float(y[mask].mean()*100),
                    'region_optimum_gap_pct':float((y.max()-y[mask].max())*100),
                    'individual_clause_gaps_pct':{h['id']:float((y.max()-y[member[j]].max())*100) for j,h in enumerate(advice['clauses'])}})
            sources.append({'case':case,'seed':seed,'context_sha256':answer['context_sha256'],
                'actual_reply_sha256':digest(answer),'usage':'Only first original reply, whose four observations match current shared initialization; held fixed for all16 decisions.'})
        write(folder/'pool.json',{'case':case,'ids':ids,'points':points,'z':z.tolist(),'initial_ids':init_ids})
        np.savez_compressed(folder/'oracle.npz',values=y)
        write(folder/'advice.json',priors)
    write(OUT/'prior_quality_evaluation_only.json',qualities)
    write(OUT/'actual_advice_sources.json',sources)
    return qualities


def prepare():
    path=OUT/'protocol.json'
    if path.exists():return check_protocol()
    qualities=prepare_inputs()
    from .guarded_suzuki_report import qualify
    qualification=qualify();assert qualification['status']=='PASS'
    protocol={'study':'matched_B4_regional_guidance_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'question':'Can B4 prior admission preserve regional LGBO gains and reduce losses from wrong knowledge beyond simpler guards?',
        'methods':list(e.METHODS),'regimes':list(e.REGIMES),'development_cases':[1],'test_cases':[2,3,4],
        'seeds':list(SEEDS),'initial_queries':4,'additional_queries':16,'repeat_queries_allowed':False,
        'common_pool':'Old512 + 512 Sobol numeric triples per catalyst +64 corners + all20 existing LHS initial points per case; generated without outcomes.',
        'common_model':'Original LGBO BaseBOSampler.fit_model -> SingleTaskGP, 4D interval category code, source Y standardization and default RBF GP fit.',
        'common_acquisition':'BoTorch qLogExpectedImprovement q1 MC512, identical sampler seed and covariance jitter1e-6 for base and guided scores.',
        'regional_component':'Original LinearExponentialRegionalMeanTiltPlugAndPlay grid512, fixed smooth0.08, confidence mapping3*Phi^-1(c). Regional-only matched variant; no point fallback.',
        'comparison_boundary':'New finite-pool matched component study; not the original official whole LGBO algorithm. Official13 files and all prior experiments remain frozen.',
        'B4_LGBO_semantics':'Use unchanged B4 subset solver on unshifted logEI to admit a largest compatible clause subset; lift its intersection; maximize lifted logEI over budget-eligible candidates. Retained clauses are soft prior preferences, not query hard constraints.',
        'B4_base_semantics':'Unchanged B4 subset solver and original hard retained-clause execution, now with the common logEI input.',
        'comparators':{'B3_gate':'Validate the one full-guidance proposal, else base argmax.',
            'LGBO_gate':'Maximize full-guidance scores on the same eligible set, without clause revocation.',
            'A2_LGBO':'Admit all four clauses if their intersection has an eligible witness, else revoke all; same guided maximization on eligible set.'},
        'budgets':{'development_ratios':list(RATIOS),'schedule':'log(ratio)*(16-t)/15; eligibility uses exact base_max-base_score<=budget, t1..16.',
            'meaning':'Bound on common MC logEI scoring loss, not a true-yield or laboratory safety guarantee.'},
        'development_selection':{'good_margin_pct':1.,'retained_gain_fraction_minimum':.8,'apply_fraction_only_if_LGBO_gain_pct_at_least':1.,
            'ranking':'Among correct-prior-qualified ratios, maximize min(mean wrong_category AUC,mean mixed AUC), then average of those, then prefer smaller ratio.',
            'if_none_qualified':'Use predefined fallback ratio16 for negative-evidence testing; do not claim development qualification.'},
        'empirical_test_success':{'correct_noninferiority_margin_pct':1.,'positive_gain_retention_minimum':.8,
            'wrong_mixed_AUC_improvement_vs_LGBO_pct':2.,'incremental_AUC_over_each_simple_guard_on_wrong_mixed':'strictly positive mean',
            'statistical_boundary':'Descriptive gates only; three reaction scenarios, not 15 independent chemistries.'},
        'knowledge':{'synthetic':'Reference-outcome-aware stress priors; correct includes reference optimum, wrong_category replaces its catalyst clause, mixed corrupts residence/loading. These are not LLM-generated claims.',
            'truth_is_not_policy_input':True,'confidence':.7,'hypothesis_ids':'Random fixed coordinate-to-ID permutation per case/seed, independent of truth; same permutation across regimes.',
            'natural':'First saved real assistant ROI only; exact same initial four observations; no new LLM inference, no adaptive refresh.'},
        'development_unique_runs':325,'test_unique_runs':375,'B0_reused_across_regimes':True,'ungated_LGBO_reused_across_budget_choices':True,
        'failure_policy':'Retain all planned rows, carry last incumbent through remaining budget; infrastructure errors recorded separately.',
        'data_exposure':'Case1 used for selection; previous chat exposed all Suzuki scenarios, so new study is exploratory and not blind.',
        'resources':{'workers':5,'threads_per_worker':1,'device':'cpu','GP_and_score_cache':'Exact history/configuration keys; source model fit once per distinct history, no compiled operators in loops.'},
        'scientific_sha256':scientific_hashes(),'official_sources':check_sources(),
        'qualification_sha256':filehash(OUT/'qualification.json'),
        'input_sha256':{str(f):filehash(f) for f in (ROOT/'inputs').rglob('*') if f.is_file()},
        'packages':{k:metadata.version(k) for k in ('numpy','scipy','torch','botorch','gpytorch')},
        'preserved_primary_inventory_sha256':filehash('research/suzuki_official/release_inventory.json')}
    protocol['protocol_sha256']=digest(protocol);write(path,protocol);write(ROOT/'protocol.json',protocol)
    return protocol


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h
    assert p['scientific_sha256']==scientific_hashes()
    check_sources()
    for f,h in p['input_sha256'].items():assert filehash(f)==h,f
    return p


def worker_init():
    threadpool_limits(1);torch.set_num_threads(1);modules('LGBO')
    torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    for name in ('script','trace'):
        original=getattr(torch.jit,name)
        def counted(*args,_name=name,_original=original,**kwargs):
            JIT[_name]+=1;return _original(*args,**kwargs)
        setattr(torch.jit,name,counted)


def run_group(job):
    case,seed,stage,ratios=job;p=check_protocol()
    pool=read(ROOT/'inputs'/f'case{case}'/'pool.json');priors=read(ROOT/'inputs'/f'case{case}'/'advice.json')[str(seed)]
    values=np.load(ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    ids=pool['ids'];indices={key:i for i,key in enumerate(ids)};z=np.array(pool['z'])
    initial=[{'candidate_id':key,'y':float(values[indices[key]])} for key in pool['initial_ids'][str(seed)]]
    states=[]
    specs=[('B0','none',None)]+[('LGBO_region',g,None) for g in e.REGIMES]
    specs += [(m,g,r) for g in e.REGIMES for r in ratios for m in e.GATED]
    for method,regime,ratio in specs:
        key=f'c{case}_s{seed}_{regime}_{method}'+(f'_r{ratio}' if ratio else '')
        file=ROOT/stage/'runs'/key/'state.json'
        state=read(file) if file.exists() else {'run_id':key,'case':case,'seed':seed,'stage':stage,
            'method':method,'regime':regime,'ratio':ratio,'status':'RUNNING','history':deepcopy(initial),'decisions':[],
            'protocol_sha256':p['protocol_sha256'],'active_seconds':0.,'operator_compiles_in_loop':0}
        assert state['protocol_sha256']==p['protocol_sha256'];states.append((state,file))
    cache=e.ScoreCache(z,ids,ROOT/'cache'/f'c{case}_s{seed}',case,seed)
    empty={'confidence':.7,'clauses':[]}
    failures=[]
    for t in range(1,17):
        for state,file in states:
            if state['status'] in ('COMPLETE','FAILED') or len(state['history'])!=t+3:continue
            started=time.perf_counter();before_jit=sum(JIT.values())
            try:
                advice=priors[state['regime']] if state['regime']!='none' else empty
                membership=e.clause_membership(z,advice)
                history=state['history'];base,base_meta=cache.score(history)
                guidance_meta={}
                def guided(retained):
                    if not retained:return base
                    score,meta=cache.score(history,e.region_for(advice,retained),advice['confidence'])
                    guidance_meta[','.join(retained)]=meta;return score
                queried=np.array([key in {h['candidate_id'] for h in history} for key in ids],dtype=bool)
                budget=e.budget_at(state['ratio'],t) if state['ratio'] else 0.
                selected,event=e.choose_policy(state['method'],base,advice,membership,budget,queried,ids,guided)
                assert not queried[selected]
                if state['method'] in e.GATED:assert not event['budget_violation']
                else:event['budget_logei']=None;event['budget_violation']=None
                if state['method']=='B4_base':assert event['selected_satisfies_retained_box']
                state['decisions'].append({'iteration':t,**event,'base_acquisition':base_meta,'guided_acquisitions':guidance_meta})
                # The hidden full response table is used only after selection.
                state['history'].append({'candidate_id':ids[selected],'y':float(values[selected])})
                if len(state['history'])==20:state['status']='COMPLETE'
            except Exception as exc:
                import traceback
                state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());failures.append(state['run_id'])
            state['active_seconds']+=time.perf_counter()-started
            state['operator_compiles_in_loop']+=sum(JIT.values())-before_jit
            write(file,state)
    receipt={'case':case,'seed':seed,'stage':stage,'planned':len(states),'complete':sum(s['status']=='COMPLETE' for s,f in states),
        'failed':failures,'cache_counters':cache.counters,'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s,f in states)}
    write(ROOT/stage/'groups'/f'c{case}_s{seed}.json',receipt)
    return receipt


def run_stage(stage):
    p=check_protocol()
    if stage=='development':cases=[1];ratios=RATIOS
    else:
        selection=read(OUT/'selected_budget.json');assert selection['protocol_sha256']==p['protocol_sha256']
        cases=[2,3,4];ratios=(selection['ratio'],)
    jobs=[(c,s,stage,ratios) for c in cases for s in SEEDS]
    with ProcessPoolExecutor(max_workers=5,initializer=worker_init) as pool:
        for f in as_completed([pool.submit(run_group,j) for j in jobs]):
            value=f.result();print(json.dumps(value,separators=(',',':')),flush=True)
    return {'stage':stage,'groups':len(jobs)}


def state_metrics(state):
    y=[h['y'] for h in state['history']];curve=np.maximum.accumulate(y).tolist()
    curve.extend([curve[-1]]*(20-len(curve)))
    return {'run_id':state['run_id'],'case':state['case'],'seed':state['seed'],'method':state['method'],
        'regime':state['regime'],'ratio':state['ratio'],'status':state['status'],'queries':len(y),
        'auc':float(np.mean(curve[4:]))*100,'final':curve[-1]*100,'initial':curve[3]*100,'curve':[v*100 for v in curve]}


def select_budget():
    p=check_protocol();path=OUT/'selected_budget.json'
    if path.exists():return read(path)
    if list((ROOT/'test').rglob('state.json')):raise ValueError('test data exists before development selection')
    states=[read(f) for f in (ROOT/'development/runs').glob('*/state.json')]
    assert len(states)==p['development_unique_runs'] and all(s['status']=='COMPLETE' for s in states)
    rows=[state_metrics(s) for s in states]
    mean=lambda method,regime=None,ratio=None:float(np.mean([r['auc'] for r in rows if r['method']==method and (regime is None or r['regime']==regime) and (ratio is None or r['ratio']==ratio)]))
    base=mean('B0');guided=mean('LGBO_region','correct');gain=guided-base;candidates=[]
    for ratio in RATIOS:
        correct=mean('B4_LGBO','correct',ratio);retention=(correct-base)/gain if gain>=1 else None
        wrong=mean('B4_LGBO','wrong_category',ratio);mixed=mean('B4_LGBO','mixed',ratio)
        candidates.append({'ratio':ratio,'correct_auc':correct,'LGBO_correct_auc':guided,'B0_auc':base,
            'gain_retention':retention,'qualified':correct>=guided-1 and (retention is None or retention>=.8),
            'wrong_auc':wrong,'mixed_auc':mixed,'worst_harmful_auc':min(wrong,mixed),'mean_harmful_auc':(wrong+mixed)/2})
    valid=[r for r in candidates if r['qualified']]
    chosen=min(valid,key=lambda r:(-r['worst_harmful_auc'],-r['mean_harmful_auc'],r['ratio'])) if valid else next(r for r in candidates if r['ratio']==16)
    result={'protocol_sha256':p['protocol_sha256'],'ratio':chosen['ratio'],'development_qualified':bool(valid),
        'selection_status':'QUALIFIED_ON_CASE1' if valid else 'NO_DEVELOPMENT_CONFIGURATION_PRESERVES_GUIDANCE_PREDEFINED_FALLBACK16',
        'candidates':candidates,'development_results_sha256':digest(rows),'test_results_seen_before_selection':False}
    write(path,result);return result
