from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from chem.benchmarks import stable_feedback_requests as r,stable_feedback_bridge as b
from chem.benchmarks import stable_followup_bridge as previous,reasoning_bo_data as data
from chem.benchmarks import stable_feedback_study as study,stable_feedback_report


def test_normalization_preserves_observation_indices_and_json_types():
    pool={'ids':['low_id','high_id'],'axes':['a'],'points':[{'a':'Z'},{'a':'A'}]}
    public={'B4_diagnosis':{'retained_ids':('H1',),'subset_checks':({'retained_ids':('H1',)},)},
        'observations':[{'index':1,'candidate_id':'low_id'},{'index':2,'candidate_id':'high_id'}],
        'pool':{'candidates':[{'candidate_id':'low_id'},{'candidate_id':'high_id'}]}}
    out=r.normalize(public,pool)
    assert out['B4_diagnosis']['retained_ids']==['H1']
    assert out['B4_diagnosis']['subset_checks']==[{'retained_ids':['H1']}]
    assert out['observations']==public['observations']
    assert [x['candidate_id'] for x in out['pool']['candidates']]==['high_id','low_id']
    changed=deepcopy(public);changed['pool']['candidates'].reverse()
    assert r.normalize(changed,pool)==out


def test_raw_cache_revalidates_own_plan(tmp_path,monkeypatch):
    calls=[]
    def get(client,ctx,folder,system,validator):
        calls.append(ctx);raw={'value':7};answer={'raw':raw,'plan':validator(raw),'context_sha256':ctx['context_sha256']}
        data.write(folder/'validated.json',answer);return answer
    monkeypatch.setattr(r.llm.nominal,'get',get)
    fake=SimpleNamespace(ROOT=tmp_path/'r',__file__=study.__file__,policy=study.policy,bridge=study.bridge,engine=study.engine,
        config=lambda v:{'id':v,'tie_seed':0},folders=lambda v:(tmp_path/v,tmp_path/'out'/v))
    pool={'ids':['p1'],'axes':['a'],'points':[{'a':'chem'}]};cache=SimpleNamespace(pool=pool)
    state={'task':'T','history':[{'candidate_id':'own1','y':.3}],'protocol_sha256':'p','fit_recovery_seen':True}
    public={'query_round':1,'B4_diagnosis':{'retained_ids':('H1',)},'pool':{'candidates':[{'candidate_id':'p1'}]}}
    a=r.request_for(fake,'feedback_test0',cache);c=r.request_for(fake,'feedback_test1',cache)
    x,k=a(None,public,state,'system',lambda raw:{'chosen':'own0','raw':raw},'model_challenge')
    y,j=c(None,public,state,'system',lambda raw:{'chosen':'own1','raw':raw},'model_challenge')
    assert len(calls)==1 and x['raw']==y['raw'] and x['plan']!=y['plan'] and k!=j
    c(None,public,{**state,'history':state['history']+[{'candidate_id':'own2','y':.4}]},'system',lambda raw:raw,'model_challenge')
    assert len(calls)==2


def test_own_category_improvement_keeps_the_next_context_test_on_that_category():
    root=Path('data/runs/stable_r2_20261029/complete_t2/runs');path=next(root.glob('Perera*/state.json'));s=data.read(path)
    pool=data.load_pool('Perera_Suzuki');ix={k:i for i,k in enumerate(pool['ids'])}
    state={**s,'history':s['history'][:5],'decisions':s['decisions'][:1],'recombination_calls':s['recombination_calls'][:1],
        'knowledge':s['initial_knowledge']}
    commit=data.read(path.parent/'rounds/02_committed.json')
    with np.load(path.parent/'rounds/02_candidates.npz') as z:d={'available':z['available'].copy(),'model_identifiability':commit['decision']['model_identifiability']}
    old=previous.candidates(pool,state,d,2);new=b.candidates(pool,state,d,2);axis=new['critical_axis']
    level=pool['points'][ix[state['recombination_calls'][0]['selected_id']]][axis]
    assert len({pool['points'][ix[c['candidate_id']]][axis] for c in old['candidates']})==2
    assert new['candidates'] and all(pool['points'][ix[c['candidate_id']]][axis]==level for c in new['candidates'])
    assert new['feedback_focus']['previous_own_best_percent']==17.26
    assert new['feedback_focus']['observed_improvement_pp']>=b.CATEGORY_IMPROVEMENT_PP
    bad=deepcopy(state);bad['recombination_calls'][0]['observed_percent']=20.
    assert 'feedback_focus' not in b.candidates(pool,bad,d,2)


def test_archive_uses_earliest_identical_input_not_results(tmp_path):
    context={'task':'T','phase':'P','system_sha256':'s','observed_prefix':[{'y':.3}],'public':{'x':1}}
    for name,time,future in [('older',1.,0.),('newer',2.,100.)]:
        folder=tmp_path/'card_shared'/name
        data.write(folder/'context.json',{**context,'later_result_not_used':future})
        data.write(folder/'validated.json',{'raw':{'candidate_id':name},'accepted_attempt':1})
        data.write(folder/'attempt01/response.json',{'created':time})
    assert r.find_archive(tmp_path,context,Path('absent')).parent.name=='older'
    assert r.find_archive(tmp_path,{**context,'public':{'x':2}},Path('absent')) is None
