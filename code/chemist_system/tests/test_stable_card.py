from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from chem.benchmarks import stable_card_policy as p,stable_card_requests as requests
from chem.benchmarks import stable_agent_policy as legacy,reasoning_bo_data as data
from chem.benchmarks import stable_card_study as study,stable_card_report


def doy(round,seed=0):
    pool=data.load_pool('Doyle_Buchwald_Hartwig');path=next(Path(f'data/runs/stable_r2_20261029/followup_t{seed}/runs').glob('Doyle*/state.json'));saved=data.read(path)
    state={**saved,'variant':'card_t0','history':saved['history'][:round+3],'decisions':saved['decisions'][:round-1],
        'knowledge':saved['decisions'][round-1]['knowledge'],'validations':[v for v in saved['validations'] if v['query']<round],
        'fit_recovery_seen':False,'tie_calls':[]}
    with np.load(path.parent/'rounds'/f'{round:02d}_candidates.npz') as z:
        d={'base':z['base_qLogEI'].copy(),'score':z['guided_qLogEI'].copy(),'available':z['available'].copy(),
            'eligible':z['eligible'].copy(),'stats':{k:z[k].copy() for k in ('mu_y','sd_y')}}
    d['reference']=pool['ids'].index(saved['decisions'][round-1]['reference_id'])
    d['selected']=pool['ids'].index(saved['decisions'][round-1]['normal_R2_proposal_id'])
    return pool,state,d


def test_expected_values_and_explicit_chemical_preference_not_ids_or_row_order():
    prepared={'candidates':[{'candidate_id':k} for k in ('p999','p001','p555')]}
    raw={'ratings':[{'candidate_id':k,'expected_yield_percent':80.,'reason':'same value, different chemistry'} for k in ('p999','p001','p555')],
        'preference_order':['p555','p999','p001'],'support':[1],'summary':'prefer chemistry555'}
    assert p.ratings(raw,prepared,[{}])['candidate_id']=='p555'
    reversed_raw=deepcopy(raw);reversed_raw['ratings'].reverse();prepared['candidates'].reverse()
    assert p.ratings(reversed_raw,prepared,[{}])['candidate_id']=='p555'
    reversed_raw['ratings'][0]['expected_yield_percent']=90
    assert p.ratings(reversed_raw,prepared,[{}])['candidate_id']==reversed_raw['ratings'][0]['candidate_id']
    invalid=deepcopy(raw);invalid['preference_order']=['p555']*3
    with pytest.raises(ValueError):p.ratings(invalid,prepared,[{}])


def test_full_counter_pool_and_complete_exact_tie_class():
    pool,state,d=doy(6,1);old=legacy.build_candidates(pool,state,d);new=p.build_candidates(pool,state,d)
    assert len(old['candidates'])==5 and len(new['candidates'])>5
    pool,state,d=doy(3);tie=p.tie_candidates(pool,state,d,3);top=d['score'][d['eligible']].max()
    expected={pool['ids'][j] for j in np.flatnonzero(d['eligible']&(d['score']==top))}
    assert len(expected)==3 and {r['candidate_id'] for r in tie['candidates']}==expected
    state['fit_recovery_seen']=True;assert not p.tie_candidates(pool,state,d,3)['candidates']


def test_grouped_summary_has_no_tie_cut_and_is_catalog_order_invariant():
    pool,state,d=doy(3)
    c=SimpleNamespace(pool=pool,score=lambda h:(d['base'],None,None),stats=lambda h:d['stats'])
    result=p.grouped_summary(c,state);assert sum(r['candidate_count'] for r in result)>=16
    available=d['available']
    for row in result:assert row['candidate_count']==int(np.sum(available&(d['base']==row['base_qLogEI'])))
    order=np.random.default_rng(19).permutation(len(pool['ids']));other={**pool,'ids':[pool['ids'][j] for j in order],'points':[pool['points'][j] for j in order]}
    c=SimpleNamespace(pool=other,score=lambda h:(d['base'][order],None,None),stats=lambda h:{k:v[order] for k,v in d['stats'].items()})
    assert p.grouped_summary(c,{**state,'tie_seed':2})==result


def test_shared_card_key(tmp_path,monkeypatch):
    calls=[]
    def get(client,ctx,folder,system,validator):
        path=folder/'validated.json'
        if path.exists():return data.read(path)
        calls.append(ctx);raw={'v':len(calls)};answer={'raw':raw,'plan':validator(raw),'context_sha256':ctx['context_sha256']}
        data.write(path,answer);return answer
    monkeypatch.setattr(requests.llm.nominal,'get',get)
    fake=SimpleNamespace(ROOT=tmp_path/'r',__file__=study.__file__,folders=lambda v:(tmp_path/v,tmp_path/'out'/v))
    state={'task':'Doyle_Buchwald_Hartwig','history':[{'candidate_id':'own1','y':.3}],
        'protocol_sha256':'own_protocol','fit_recovery_seen':False}
    public={'query_round':1,'observation_count':1,'candidates':['chemistry'],'normal_BO_reference_id':'arbitrary1'}
    a=requests.request_for(fake,'card_test0',None);b=requests.request_for(fake,'card_test1',None)
    x,k=a(None,public,state,'systemA',lambda r:r,'tie_review')
    y,j=b(None,{**public,'normal_BO_reference_id':'arbitrary2'},state,'systemA',lambda r:r,'tie_review')
    assert x['raw']==y['raw'] and len(calls)==1 and k!=j
    assert data.read(tmp_path/'card_test0/requests'/k/'advisor_source.json')['new_api']
    assert not data.read(tmp_path/'card_test1/requests'/j/'advisor_source.json')['new_api']
    changed={**state,'history':state['history']+[{'candidate_id':'own2','y':.7}]}
    b(None,public,changed,'systemA',lambda r:r,'tie_review');assert len(calls)==2
    b(None,public,state,'systemB',lambda r:r,'tie_review');assert len(calls)==3
