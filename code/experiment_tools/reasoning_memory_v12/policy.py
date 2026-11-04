"""Immutable notes; historical facts survive edits, old KF measurements do not."""
from copy import deepcopy
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import policy as base, kf

scope_indices=base.scope_indices
validate_existing=base.validate_existing
next_ids=base.next_ids


def incumbent(pool,history):
    return min(history,key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))


def negative_facts(hs):
    facts={}
    for h in hs:
        for e in h['events']:
            if e['Brier_gain'] < -1e-12 or (not e['success'] and e['p_LLM']>=.5):
                facts[e['event_id']]=dict(event_id=e['event_id'],id=h['id'],version=h['version'],
                    query=e['query'],point_id=e['point_id'],reference_id=e['reference_id'],
                    reference_y=e['reference_y'],observed_y=e['observed_y'],
                    p_LLM=e['p_LLM'],p_GP=e['p_GP'],Brier_gain=e['Brier_gain'],
                    success=e['success'],claim=h['claim'],scope=deepcopy(h['scope']))
    return sorted(facts.values(),key=lambda e:(e['query'],e['id'],e['version']))


def positive_count(h):
    # Correctly anticipating a failure is calibration evidence, not support for
    # an improvement direction. Two successful experiments are still observational.
    return sum(e['success'] and e['Brier_gain']>1e-12 for e in h['events'])


def support_eligible(h):
    return h['kf']['status']=='active' and h['kf']['weight']>0 and positive_count(h)>=2


def apply(pool,hs,actions,history,t):
    if not isinstance(actions,list) or len(actions)>3:raise ValueError('At most three knowledge actions')
    result=deepcopy(hs);reviews=[];touched=set();allowed=set(next_ids(hs))
    ref=incumbent(pool,history);facts=negative_facts(hs)
    for h in result:validate_existing(h)
    for pos,a in enumerate(actions):
        review=dict(action=pos,accepted=False)
        try:
            if not isinstance(a,dict) or set(a)!=base.ACTION_KEYS|{'evidence_queries'}:raise ValueError('Incorrect knowledge action schema')
            op,hid=a['op'],a['id'];review.update(op=op,id=hid)
            if op not in ('add','retain','revise','retract') or type(hid) is not str or hid in touched:raise ValueError('Invalid or repeated action')
            if type(a['reason']) is not str or not 1<=len(a['reason'])<=800:raise ValueError('Reason required')
            evidence=a['evidence_queries']
            if not isinstance(evidence,list) or any(type(q) is not int or not 1<=q<t for q in evidence):raise ValueError('Evidence must name completed own query numbers')
            old=next((h for h in reversed(result) if h['id']==hid),None)
            if op=='add' and (old is not None or hid not in allowed):raise ValueError('Use supplied unused ID')
            if op!='add' and old is None:raise ValueError('Unknown existing knowledge')
            if op in ('retain','retract'):
                if any(a[k] is not None for k in ('claim','effect','reference_id')) or a['focus']!=[] or a['scope']!={}:raise ValueError('Retain/retract cannot rewrite semantics')
                if old['kf']['status']=='retracted':raise ValueError('Retired note requires revision')
                if op=='retract':
                    negative_queries={f['query'] for f in facts if f['id']==hid}
                    if not set(evidence)&negative_queries:raise ValueError('Retraction needs a recorded adverse forecast; lack of menu coverage is dormancy')
                    old['kf'].update(status='retracted',weight=0.,retracted_round=t)
                    old['retirement_reason']='LLM: '+a['reason']
                touched.add(hid);review.update(accepted=True,outcome=op,version=old['version']);reviews.append(review);continue
            if (type(a['claim']) is not str or not 1<=len(a['claim'])<=900 or a['effect'] not in ('promote','avoid')
                or not isinstance(a['focus'],list) or not 1<=len(a['focus'])<=2 or len(set(a['focus']))!=len(a['focus'])
                or set(a['focus'])-set(pool.axes) or a['reference_id']!=ref['candidate_id']):raise ValueError('Invalid claim, focus or current incumbent birth reference')
            new=dict(id=hid,version=old['version']+1 if old else 1,claim=a['claim'],focus=sorted(a['focus']),effect=a['effect'],
                scope=base.normalize_scope(pool,a['scope']),reference_id=ref['candidate_id'],reference_y=float(ref['y']),
                created_round=t,events=[],inherited_negative=[],carried_fact_ids=[],birth_history_sha256=digest(history),
                kf=kf.initial(),reason=a['reason'])
            if len(scope_indices(pool,new,range(len(pool.ids))))<2:raise ValueError('Scope needs at least two pool inputs; current-menu overlap is not required')
            new['semantic_sha256']=base.semantics(new);new['structure_sha256']=base.structural(new)
            if any(h['id']!=hid and base.structural(h)==new['structure_sha256'] for h in result):raise ValueError('Structural duplicate: revise its existing ID')
            if old is not None and old['semantic_sha256']==new['semantic_sha256']:
                review.update(accepted=True,outcome='unchanged_version_preserved',version=old['version']);reviews.append(review);touched.add(hid);continue
            live=sum(h['kf']['status']!='retracted' for h in result)
            if live-int(old is not None and old['kf']['status']!='retracted')>=4:raise ValueError('At most four live notes')
            # Facts refer to actual measured conditions; changing focus or note ID
            # cannot remove a matching point's adverse evidence. No Brier replay.
            new['carried_fact_ids']=[f['event_id'] for f in facts if f['id']==hid or len(scope_indices(pool,new,[pool.ix[f['point_id']]]))]
            if old is not None:
                old['kf'].update(status='retracted',weight=0.,retracted_round=t)
                old['retirement_reason']='superseded by explicit revision'
            result.append(new);touched.add(hid)
            review.update(accepted=True,outcome=op,version=new['version'],carried_fact_count=len(new['carried_fact_ids']),numeric_prior_replays=0)
        except (ValueError,TypeError,KeyError) as error:review['reason']=str(error)
        reviews.append(review)
    return result,reviews
