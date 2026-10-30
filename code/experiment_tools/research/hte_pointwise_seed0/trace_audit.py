"""Independent observed-prefix, KF-state continuity and top5 transport audit."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
ROOT=BASE/'data/runs/hte_pointwise_seed0'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(x):return sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf8')).hexdigest()


def main():
    states=[read(ROOT/'runs'/r['run_id']/'state.json') for r in read(ROOT/'manifest.json')]
    assert len(states)==16 and all(s['status']=='COMPLETE' for s in states)
    prefix_count=0;kf_count=0;top5_count=0
    for path in (ROOT/'requests').glob('*/context.json'):
        ctx=read(path);assert (path.parent/'validated.json').exists()
        if ctx['method']=='shared_initial_knowledge':
            expected=read(ROOT/'initialization'/ctx['task']/'observed.json')
        else:
            s=next(s for s in states if s['task']==ctx['task'] and s['method']==ctx['method'])
            expected=s['history'][:ctx['public']['query_round']+3]
        assert ctx['observed_prefix']==expected
        assert ctx['public']['observation_count']==len(expected);prefix_count+=1
    for s in states:
        pool=read(BASE/'data/benchmarks/reasoning_bo'/s['task']/'pool.json');banks={}
        for d in s['decisions']:
            t=d['iteration'];folder=ROOT/'runs'/s['run_id']/'rounds';c=read(folder/f'{t:02d}_committed.json')['decision']
            with np.load(folder/f'{t:02d}_candidates.npz') as a:
                if s['method']=='ReasoningBO_point':
                    order=sorted(np.flatnonzero(a['available']),key=lambda j:(-float(a['base_qLogEI'][j]),pool['ids'][j]))[:5]
                    assert d['extra']['shortlist_ids']==[pool['ids'][j] for j in order];top5_count+=1
                if d['measurement']:
                    key=c['bank_key'];bank=banks.get(key,{'mean':[0.,1.],'covariance':[[.0025,0],[0,.25]],'last_round':t-1,'updates':0})
                    assert c['bank']==bank
                    np.testing.assert_allclose(c['prior']['mean'],bank['mean'],atol=1e-12,rtol=0)
                    expected=np.asarray(bank['covariance'])+np.diag([.000025,.0025])*(t-bank['last_round'])
                    np.testing.assert_allclose(c['prior']['covariance'],expected,atol=1e-12,rtol=0)
                    j=pool['ids'].index(d['selected_id']);m=d['measurement']
                    assert m['prequery_mu_y']==a['mu_y'][j]
                    assert m['prequery_predictive_sd_y']==a['sd_observation_y'][j]
                    assert m['observed_y']==d['y'] and m['residual']==d['y']-a['mu_y'][j]
                    banks[key]=d['calibration_after'];kf_count+=1
        if s['method']=='New_R2':assert banks==s['banks']
    pairs=[]
    for task in ('Perera_Suzuki','Doyle_Buchwald_Hartwig'):
        subset=[s for s in states if s['task']==task]
        for i,a in enumerate(subset):
            for b in subset[i+1:]:
                x=[d['selected_id'] for d in a['decisions']];y=[d['selected_id'] for d in b['decisions']]
                cx=[d['incumbent_after'] for d in a['decisions']];cy=[d['incumbent_after'] for d in b['decisions']]
                pairs.append({'task':task,'left':a['method'],'right':b['method'],
                    'same_ordered_choices':sum(i==j for i,j in zip(x,y)),'same_whole_trajectory':x==y,
                    'same_incumbent_curve':cx==cy})
    result={'status':'PASS','true_own_prefix_requests':prefix_count,'KF_state_chain_updates':kf_count,
        'exact_base_top5_transports':top5_count,'trajectory_pairs':pairs,
        'scope':'Checks observation isolation and chronological state transport. No new experimental selections.'}
    (OUT/'trace_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='trajectory_pairs'}))


if __name__=='__main__':main()
