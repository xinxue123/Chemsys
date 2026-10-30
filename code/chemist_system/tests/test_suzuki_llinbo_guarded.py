"""Qualify upstream execution, provider pause replay, and matched transport."""
import json
import numpy as np
import pytest
from chem.benchmarks import llinbo_guarded_engine as e
from chem.benchmarks.official_suzuki_adapter import AwaitingAssistant


def fixture():
    z=[[.05+i*.025,.5,.5,.0625] for i in range(32)]
    points=[{'catalyst':'P1-L1','t_res':60+540*x[0],'temperature':70.,'catalyst_loading':1.5} for x in z]
    pool={'ids':[f'p{i:02d}' for i in range(32)],'z':z,'points':points}
    initial=[{'candidate_id':pool['ids'][i],'y':.1+i*.01} for i in range(4)]
    class Cache:
        def score(self,h):return np.arange(32,dtype=float)*.2,{'fixture':True,'history':h}
    return pool,initial,Cache()


def run(proposals,steps=2,pause=None):
    pool,initial,cache=fixture();records=[];prompts=[]
    def assistant(history,prompt,t):
        prompts.append(prompt)
        if pause==t:raise AwaitingAssistant('fixture')
        i=proposals[t-1]
        projection={'candidate_id':pool['ids'][i],'z':pool['z'][i],'point':pool['points'][i],
                    'normalized_numeric_distance':0.,'raw_point':pool['points'][i],'raw_z':pool['z'][i]}
        return projection,{'context_sha256':f'fixture-{t}'}
    def objective(candidate,event):records.append(event);return .5
    result=e.execute(pool=pool,initial=initial,advice={'clauses':[]},case=1,seed=0,ratio=4,
        cache=cache,assistant=assistant,objective=objective,steps=steps)
    return result,records,prompts


def test_projection_preserves_category_and_excludes_previously_queried_point():
    pool,_,_=fixture();point=pool['points'][10]
    # A different catalyst has exactly the same numeric coordinates: it must
    # never win the projection merely because its numeric distance is zero.
    pool['ids'].append('other-category')
    pool['z'].append([*pool['z'][10][:3],.1875])
    pool['points'].append({**point,'catalyst':'P2-L1'})
    projection=e.project_proposal(point,pool,{pool['ids'][10]})
    assert projection['candidate_id']!=pool['ids'][10]
    assert projection['point']['catalyst']==point['catalyst']
    assert projection['normalized_numeric_distance']==pytest.approx(.025)


def test_actual_source_branch_accepts_and_rejects_shared_scores():
    result,events,_=run([30,4])
    assert result['source_entrypoint']=='LLMIBO_BFO.run -> unchanged _run_justify'
    assert [v['accepted'] for v in events]==[True,False]
    assert [v['selected_id'] for v in events]==['p30','p31']
    assert all(v['selected_logei_loss']<=v['budget_logei'] for v in events)


def test_upstream_body_and_prompt_are_stable_under_restart():
    with pytest.raises(AwaitingAssistant):run([30,4],pause=2)
    a=run([30,4]);b=run([30,4])
    assert a==b
    assert 'The following are past evaluations' in a[2][0]
    assert 'recommend the next point' in a[2][0]


def test_matched_schedule_closes_at_last_query():
    schedule=e.MatchedThreshold(4)
    assert schedule/1==np.log(4) and schedule/16==0.


def test_source_files_remain_unchanged():
    from chem.benchmarks.official_suzuki_adapter import check_sources
    assert len(check_sources()['files'])==13
