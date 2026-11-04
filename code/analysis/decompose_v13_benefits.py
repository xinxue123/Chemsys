"""Exact, post-hoc benefit accounting; no policy execution or new queries.

Run with runtime/environments/.venv-official/Scripts/python.exe. Evaluation
labels never enter a decision worker. Local one-step comparisons keep the
actual prefix and must not be interpreted as counterfactual trajectories.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper/v13_late/data'
OUT = ROOT / 'experiments/reports/v13_benefit_decomposition_20261103'
TASKS = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
LABELS = dict(c1='Suzuki 1', c2='Suzuki 2', c3='Suzuki 3', c4='Suzuki 4',
              perera='Perera', doyle='Doyle', snar='SnAr', hplc='HPLC')
METHODS = ('bo', 'reasoning_bo', 'v13')
READS: dict[str, str] = {}
EPS = 1e-8


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tracked(path):
    p = Path(path).resolve()
    b = p.read_bytes()
    READS[p.relative_to(ROOT).as_posix()] = sha(b)
    return b


def read(path):
    return json.loads(tracked(path).decode('utf-8-sig'))


def rows(path):
    return list(csv.DictReader(tracked(path).decode('utf-8-sig').splitlines()))


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8'))


def close(a, b):
    if not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f'Numerical mismatch: {a} != {b}')


def sign(x):
    return 'win' if x > EPS else 'loss' if x < -EPS else 'tie'


def wtl(values):
    c = Counter(map(sign, values))
    return dict(wins=c['win'], ties=c['tie'], losses=c['loss'])


def csv_write(name, values):
    values = list(values)
    if not values:
        raise ValueError(f'Unexpected empty output: {name}')
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        for r in values:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})


def phase(values, floor, cutoff):
    B = len(values)
    best = np.maximum.accumulate(values)
    increments = np.diff(np.r_[floor, best])
    weighted = np.arange(B, 0, -1) / B * increments
    early_origin = float(weighted[:cutoff].sum())
    late_origin = float(weighted[cutoff:].sum())
    auc = float(best.mean())
    close(floor + early_origin + late_origin, auc)
    early_area = float(best[:cutoff].sum() / B)
    carry = float((B-cutoff) / B * best[cutoff-1])
    close(early_area + carry + late_origin, auc)
    return dict(SearchAUC=auc, first_objective=float(values[0]),
                M_cutoff=float(best[cutoff-1]), final_best=float(best[-1]),
                early_window_contribution=early_area,
                inherited_incumbent_contribution=carry,
                early_origin_contribution=early_origin,
                late_origin_contribution=late_origin,
                late_best_gain=float(best[-1]-best[cutoff-1]),
                first_final_best_query=int(np.flatnonzero(best >= best[-1]-EPS)[0])+1)


def load_commit(spec):
    key, q, p = spec
    b = p.read_bytes()
    return key, q, p.resolve().relative_to(ROOT).as_posix(), sha(b), json.loads(b)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [ROOT / p for p in (
        'paper/v13_source.zip', 'paper/v13/main.pdf', 'paper/v13/supplement.pdf',
        'paper/v13_late_source.zip', 'paper/v13_late/main.pdf',
        'paper/v13_late/supplement.pdf')]
    protected = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in protected_paths}
    tracked(ROOT / 'docs/v13_benefit_decomposition_20261103.txt')
    metrics = [r for r in rows(DATA / 'metrics.csv') if r['regime']=='cold']
    assert len(metrics)==96
    assert Counter(r['method'] for r in metrics)==Counter({m:32 for m in METHODS})
    exported_rounds=defaultdict(list)
    for r in rows(DATA/'rounds.csv'):
        if r['regime']=='cold':exported_rounds[r['task'],int(r['seed']),r['method']].append(r)
    coverage = { (r['job'], int(r['query'])):r for r in rows(DATA/'candidate_coverage_queries.csv')
                 if r['cohort']=='cold' }
    note_coverage = { (r['job'], int(r['query']), r['note_id'], int(r['version'])):r
                     for r in rows(DATA/'candidate_coverage_notes.csv') if r['cohort']=='cold' }
    conditional = rows(DATA/'conditional_effects.csv')
    truth = {}; floors = {}; spans = {}
    for r in rows(DATA/'datasets.csv'):
        p = read(ROOT/r['pool_path'])
        tracked(ROOT/r['oracle_path'])
        with np.load(ROOT/r['oracle_path'], allow_pickle=False) as z:
            raw = z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*100
        assert len(p['ids'])==len(raw) and len(set(p['ids']))==len(raw)
        truth[r['task']] = dict(zip(p['ids'], map(float, raw)))
        floors[r['task']] = min(0., float(raw.min()))
        spans[r['task']] = float(raw.max()-raw.min())
        assert spans[r['task']]>0

    states = {}; specs=[]; by_key={}
    for m in metrics:
        key=(m['task'], int(m['seed']), m['method'])
        assert key not in states and int(m['initial'])==0 and int(m['queries'])==20
        sf=ROOT/m['state_path']; s=read(sf)
        assert sha(sf.read_bytes())==m['state_sha256'] and s['status']=='COMPLETE'
        assert len(s['history'])==len(s['decisions'])==20
        assert len({h['candidate_id'] for h in s['history']})==20
        er=sorted(exported_rounds[key],key=lambda r:int(r['query']))
        assert len(er)==20
        for q,(h,d) in enumerate(zip(s['history'],s['decisions']),1):
            close(h['raw_value'],truth[m['task']][h['candidate_id']])
            assert d['query']==q and d['selected_id']==h['candidate_id']
            assert er[q-1]['candidate_id']==h['candidate_id']
            close(er[q-1]['objective'],h['raw_value'])
            close(er[q-1]['cumulative_best'],max(x['raw_value'] for x in s['history'][:q]))
            specs.append((key,q,sf.parent/'rounds'/f'{q:02}_commit.json'))
        close(np.maximum.accumulate([h['raw_value'] for h in s['history']]).mean(),m['SearchAUC'])
        close(max(h['raw_value'] for h in s['history']),m['final_best'])
        states[key]=s; by_key[key]=m
    commits={}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for key,q,path,hs,c in ex.map(load_commit,specs):
            READS[path]=hs
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['commit_sha256']==states[key]['decisions'][q-1]['commit_sha256']
            assert c['selected_id']==states[key]['history'][q-1]['candidate_id']
            assert c['BO_reference_id']==c['ordered_ids'][0]
            assert len(c['display_ids'])==5 and set(c['display_ids'])==set(c['ordered_ids'])
            assert c['selected_id'] in c['display_ids']
            assert not set(c['display_ids']) & {h['candidate_id'] for h in states[key]['history'][:q-1]}
            if key[2]=='bo':assert c['selected_id']==c['BO_reference_id']
            commits[key,q]=c

    trajectories=[]; phases={}
    for key,s in states.items():
        task,seed,method=key
        for cutoff in (5,10,15):
            v=phase([h['raw_value'] for h in s['history']],floors[task],cutoff)
            phases[key,cutoff]=v
            trajectories.append(dict(task=task,seed=seed,method=method,cutoff=cutoff,
                                     **v,state_path=by_key[key]['state_path']))
    pairs=[]; divergence=[]; before_feedback=[]
    for task in TASKS:
        for seed in range(4):
            vk=(task,seed,'v13');vs=states[vk]
            first_feedback=next((d['query'] for d in vs['decisions'] if d['feedback']),None)
            for comparator in ('bo','reasoning_bo'):
                ck=(task,seed,comparator);cs=states[ck]
                # The forecast is scored after this choice: a first update at
                # q cannot explain any record created at or before query q.
                cutoff=first_feedback or 20
                av=np.maximum.accumulate([h['raw_value'] for h in vs['history']])
                bv=np.maximum.accumulate([h['raw_value'] for h in cs['history']])
                ai=np.diff(np.r_[floors[task],av]);bi=np.diff(np.r_[floors[task],bv])
                weighted_gap=np.arange(20,0,-1)/20*(ai-bi)
                pre=float(weighted_gap[:cutoff].sum());post=float(weighted_gap[cutoff:].sum())
                close(pre+post,float(av.mean()-bv.mean()))
                before_feedback.append(dict(task=task,seed=seed,comparator=comparator,
                    first_KF_feedback_query='' if first_feedback is None else first_feedback,
                    no_KF_feedback_entire_trajectory=int(first_feedback is None),
                    delta_SearchAUC=float(av.mean()-bv.mean()),
                    delta_record_credit_before_or_at_first_KF_feedback=pre,
                    delta_record_credit_after_first_KF_feedback=post))
                for cutoff in (5,10,15):
                    v=phases[vk,cutoff];b=phases[ck,cutoff]
                    delta={f'delta_{k}':v[k]-b[k] for k in v if k!='first_final_best_query'}
                    close(delta['delta_SearchAUC'],delta['delta_early_origin_contribution']+delta['delta_late_origin_contribution'])
                    close(delta['delta_SearchAUC'],delta['delta_early_window_contribution']+
                          delta['delta_inherited_incumbent_contribution']+delta['delta_late_origin_contribution'])
                    close(delta['delta_final_best'],delta['delta_M_cutoff']+delta['delta_late_best_gain'])
                    # Separate derivation from the archived per-round export:
                    # early area + carried incumbent + later excess area.
                    ae=sorted(exported_rounds[vk],key=lambda r:int(r['query']))
                    be=sorted(exported_rounds[ck],key=lambda r:int(r['query']))
                    ae=[float(r['cumulative_best']) for r in ae]
                    be=[float(r['cumulative_best']) for r in be]
                    direct_early=sum(ae[:cutoff])/20-sum(be[:cutoff])/20
                    direct_carry=(20-cutoff)/20*(ae[cutoff-1]-be[cutoff-1])
                    direct_late=sum(x-ae[cutoff-1] for x in ae[cutoff:])/20-sum(x-be[cutoff-1] for x in be[cutoff:])/20
                    close(delta['delta_early_origin_contribution'],direct_early+direct_carry)
                    close(delta['delta_late_origin_contribution'],direct_late)
                    pairs.append(dict(task=task,seed=seed,comparator=comparator,cutoff=cutoff,**delta))
                first=None
                for q in range(1,21):
                    a=commits[vk,q];b=commits[ck,q]
                    if first is None:
                        assert a['GP_sha256']==b['GP_sha256'] and a['ordered_ids']==b['ordered_ids'] and a['display_ids']==b['display_ids']
                    if a['selected_id']!=b['selected_id']:
                        first=q;break
                divergence.append(dict(task=task,seed=seed,comparator=comparator,
                    first_different_choice='' if first is None else first,
                    all_twenty_choices_identical=int(first is None),
                    V13_first_id=vs['history'][0]['candidate_id'],comparator_first_id=cs['history'][0]['candidate_id'],
                    first_menu_equal=int(commits[vk,1]['display_ids']==commits[ck,1]['display_ids']),
                    first_choice_objective_gap=vs['history'][0]['raw_value']-cs['history'][0]['raw_value'],
                    first_divergence_immediate_gap=0 if first is None else vs['history'][first-1]['raw_value']-cs['history'][first-1]['raw_value']))

    selections=[]; records=[]; feedback=[]
    for key,s in states.items():
        task,seed,method=key
        if method=='bo':continue
        job=f'historical_cold_{task}{seed}'
        before=floors[task];prior_updates=0
        for q,(h,d) in enumerate(zip(s['history'],s['decisions']),1):
            c=commits[key,q];gp_id=c['BO_reference_id'];gp_y=truth[task][gp_id];y=h['raw_value']
            own_gain=max(0.,y-before);gp_gain=max(0.,gp_y-before)
            close(y,truth[task][c['selected_id']])
            cov=coverage[job,q] if method=='v13' else None
            notes={(n['id'],n['version']):n for n in c.get('edited_notes',[])}
            uses=d.get('knowledge_uses',[])
            used_with_prior=[u for u in uses if notes.get((u['id'],u['version']),{}).get('last_feedback_query',0)>0]
            if method=='v13':
                assert int(cov['selected_forecast_count'])==len(d['feedback'])
            selected_with_prior=any(n.get('last_feedback_query',0)>0 and
                int(note_coverage[job,q,n['id'],n['version']]['selected_in_scope'])
                for n in notes.values() if n['recommendation']['permission']!='withdrawn') if method=='v13' else False
            menu_with_prior=any(n.get('last_feedback_query',0)>0 and
                int(note_coverage[job,q,n['id'],n['version']]['candidates_covered'])>0
                for n in notes.values() if n['recommendation']['permission']!='withdrawn') if method=='v13' else False
            assert all(n.get('last_feedback_query',0)<q for n in notes.values())
            row=dict(task=task,seed=seed,method=method,query=q,phase='early' if q<=10 else 'late',
                GP_defined=int(q>1),selected_id=h['candidate_id'],own_GP_reference_id=gp_id,
                changed_from_reference=int(h['candidate_id']!=gp_id),objective=y,reference_objective_offline=gp_y,
                acquisition_comparison='no_GP_prior_menu' if q==1 else 'same_point' if h['candidate_id']==gp_id
                    else 'tied_acquisition' if c['opportunity_gap']==0 else 'lower_acquisition',
                qLogEI_gap='' if q==1 else c['opportunity_gap'],
                incumbent_before='' if q==1 else before,own_record_gain=own_gain,reference_record_gain=gp_gain,
                local_record_advantage=own_gain-gp_gain,
                local_record_advantage_fraction_pool_range=(own_gain-gp_gain)/spans[task],
                immediate_objective_gap=y-gp_y,selected_record=int(own_gain>EPS),
                GP_reference_record=int(gp_gain>EPS),
                creates_record_reference_would_not=int(own_gain>EPS and gp_gain<=EPS),
                misses_record_reference_would_make=int(own_gain<=EPS and gp_gain>EPS),
                menu_best_objective_offline=max(truth[task][cid] for cid in c['display_ids']),
                editor_this_query=int(c['memory'] is not None),prior_structured_forecast_updates=prior_updates,
                selected_forecast_count=len(d.get('feedback',[])),
                live_notes='' if cov is None else int(cov['live_notes']),
                any_menu_coverage='' if cov is None else int(cov['any_menu_coverage']),
                any_selective_coverage='' if cov is None else int(cov['any_selective_coverage']),
                prior_feedback_note_covers_menu=int(menu_with_prior),
                prior_feedback_note_covers_selected=int(selected_with_prior),
                declared_note_uses=uses,declared_note_with_prior_feedback=int(bool(used_with_prior)),
                record_SearchAUC_contribution=(21-q)/20*own_gain,
                raw_insight_comment='' if not c['reasoning'] else
                    (c['reasoning']['raw'].get('insight',c['reasoning']['raw'])).get('comment',''),
                commit_path=(ROOT/by_key[key]['state_path']).parent.joinpath('rounds',f'{q:02}_commit.json').as_posix())
            selections.append(row)
            if own_gain>EPS:
                if q==1:category='first_query_no_GP_no_structured_feedback'
                elif h['candidate_id']==gp_id:category='same_as_own_GP_reference'
                elif used_with_prior:category='different_GP_declared_note_with_prior_feedback'
                elif d.get('feedback'):category='different_GP_selected_forecast_no_declared_prior_feedback'
                else:category='different_GP_no_selected_forecast'
                records.append(dict(**{k:row[k] for k in ('task','seed','method','query','phase','objective',
                    'own_record_gain','local_record_advantage','record_SearchAUC_contribution',
                    'declared_note_uses','prior_structured_forecast_updates','commit_path')},category=category))
            if method=='v13':
                packets={p['commit_sha256']:p for p in c['decision']['forecasts']}
                for e in d['feedback']:
                    f=packets[e['commit_sha256']];cal=e['calibration'];b=cal['before'];a=cal['after']
                    z=(int(e['improved'])-f['p_GP'])**2-(int(e['improved'])-f['p_LLM'])**2
                    close(z,cal['Brier_gain'])
                    changed=abs(a['reliability']-b['reliability'])>1e-12
                    next_note=None;next_cov=None;next_use=[]
                    if q<20:
                        nc=commits[key,q+1]
                        next_note=next((n for n in nc['edited_notes'] if (n['id'],n['version'])==(f['id'],f['version'])
                                       and n['recommendation']['permission']!='withdrawn'),None)
                        next_cov=note_coverage.get((job,q+1,f['id'],f['version']))
                        next_use=[u for u in states[key]['decisions'][q]['knowledge_uses']
                                  if (u['id'],u['version'])==(f['id'],f['version'])]
                    carried=next_note is not None and next_note['calibration']['reliability']==a['reliability']
                    feedback.append(dict(task=task,seed=seed,query=q,note_id=f['id'],version=f['version'],use=f['use'],
                        p_LLM=f['p_LLM'],p_GP=f['p_GP'],improved=int(e['improved']),Brier_gain=z,
                        kalman_gain=cal['kalman_gain'],updates_before=b['updates'],updates_after=a['updates'],
                        reliability_before=b['reliability'],reliability_after=a['reliability'],
                        reliability_changed=int(changed),permission_before=e['recommendation_before'],
                        permission_after=e['recommendation_after'],has_next_query=int(q<20),
                        same_version_live_next_query=int(next_note is not None),reliability_carried_next_query=int(carried),
                        same_note_next_menu_coverage=0 if next_cov is None else int(next_cov['candidates_covered']),
                        same_note_next_selected_in_scope=0 if next_cov is None else int(next_cov['selected_in_scope']),
                        same_note_next_declared_use=int(bool(next_use)),
                        next_choice_differs_GP=int(q<20 and commits[key,q+1]['selected_id']!=commits[key,q+1]['BO_reference_id'])))
            prior_updates+=len(d.get('feedback',[]));before=max(before,y)
        close(floors[task]+sum(r['record_SearchAUC_contribution'] for r in records
                              if (r['task'],r['seed'],r['method'])==key),phases[key,10]['SearchAUC'])

    task_summary=[]
    for comparator in ('bo','reasoning_bo'):
        for task in TASKS:
            ps=[p for p in pairs if p['task']==task and p['comparator']==comparator and p['cutoff']==10]
            assert len(ps)==4
            keys=[k for k in ps[0] if k.startswith('delta_')]
            task_summary.append(dict(task=task,comparator=comparator,n=4,
                **{k:mean(p[k] for p in ps) for k in keys},
                SearchAUC_wins=sum(p['delta_SearchAUC']>EPS for p in ps),
                SearchAUC_ties=sum(abs(p['delta_SearchAUC'])<=EPS for p in ps),
                SearchAUC_losses=sum(p['delta_SearchAUC'] < -EPS for p in ps)))
    selection_summary=[]
    for method in ('v13','reasoning_bo'):
        for task in (*TASKS,'all_tasks'):
            for period in ('all','first_query','early','late'):
                ss=[r for r in selections if r['method']==method and (task=='all_tasks' or r['task']==task)
                    and (period=='all' or period=='first_query' and r['query']==1 or r['phase']==period)]
                changed=[r for r in ss if r['changed_from_reference']]
                selection_summary.append(dict(task=task,method=method,period=period,choices=len(ss),
                    changed_choices=len(changed),record_events=sum(r['selected_record'] for r in ss),
                    **{f'local_record_advantage_{k}':v for k,v in wtl(r['local_record_advantage'] for r in changed).items()},
                    creates_record_reference_would_not=sum(r['creates_record_reference_would_not'] for r in ss),
                    misses_record_reference_would_make=sum(r['misses_record_reference_would_make'] for r in ss),
                    normalized_local_record_advantage_mean=mean(r['local_record_advantage_fraction_pool_range'] for r in ss),
                    selected_forecast_choices=sum(r['selected_forecast_count']>0 for r in ss),
                    prior_feedback_note_menu_choices=sum(r['prior_feedback_note_covers_menu'] for r in ss),
                    declared_prior_feedback_note_choices=sum(r['declared_note_with_prior_feedback'] for r in ss)))
    source_summary=[]
    for method in ('v13','reasoning_bo'):
        for task in TASKS:
            for category in sorted({r['category'] for r in records}):
                rs=[r for r in records if r['method']==method and r['task']==task and r['category']==category]
                source_summary.append(dict(task=task,method=method,category=category,record_events=len(rs),
                    mean_SearchAUC_record_credit=sum(r['record_SearchAUC_contribution'] for r in rs)/4,
                    early_record_events=sum(r['query']<=10 for r in rs),late_record_events=sum(r['query']>10 for r in rs)))

    facts={}
    for comparator in ('bo','reasoning_bo'):
        primary=[p for p in pairs if p['comparator']==comparator and p['cutoff']==10]
        positive=[p for p in primary if p['delta_SearchAUC']>EPS]
        facts[comparator]=dict(total_SearchAUC=wtl(p['delta_SearchAUC'] for p in primary),
            early_origin=wtl(p['delta_early_origin_contribution'] for p in primary),
            late_origin=wtl(p['delta_late_origin_contribution'] for p in primary),
            positive_total_comparisons=len(positive),
            positive_total_early_only=sum(p['delta_early_origin_contribution']>EPS and p['delta_late_origin_contribution']<=EPS for p in positive),
            positive_total_late_only=sum(p['delta_late_origin_contribution']>EPS and p['delta_early_origin_contribution']<=EPS for p in positive),
            positive_total_both=sum(p['delta_early_origin_contribution']>EPS and p['delta_late_origin_contribution']>EPS for p in positive),
            identical_full_choice_sequences=sum(d['all_twenty_choices_identical'] for d in divergence if d['comparator']==comparator),
            first_different_choice_counts=dict(Counter(str(d['first_different_choice']) for d in divergence if d['comparator']==comparator)))
        bf=[p for p in before_feedback if p['comparator']==comparator]
        positive_bf=[p for p in bf if p['delta_SearchAUC']>EPS]
        facts[comparator]['positive_total_before_first_KF_positive']=sum(
            p['delta_record_credit_before_or_at_first_KF_feedback']>EPS for p in positive_bf)
        facts[comparator]['positive_total_only_before_first_KF_positive']=sum(
            p['delta_record_credit_before_or_at_first_KF_feedback']>EPS and
            p['delta_record_credit_after_first_KF_feedback']<=EPS for p in positive_bf)
        facts[comparator]['no_KF_feedback_runs']=[dict(task=p['task'],seed=p['seed'],delta_SearchAUC=p['delta_SearchAUC'])
            for p in bf if p['no_KF_feedback_entire_trajectory']]
        facts[comparator]['cutoff_sensitivity']=[]
        for cutoff in (5,10,15):
            cs=[p for p in pairs if p['comparator']==comparator and p['cutoff']==cutoff]
            positive_cs=[p for p in cs if p['delta_SearchAUC']>EPS]
            facts[comparator]['cutoff_sensitivity'].append(dict(cutoff=cutoff,
                early_origin=wtl(p['delta_early_origin_contribution'] for p in cs),
                late_origin=wtl(p['delta_late_origin_contribution'] for p in cs),
                positive_total_with_positive_early_origin=sum(p['delta_early_origin_contribution']>EPS for p in positive_cs)))
    vsel=[r for r in selections if r['method']=='v13']
    changed_next=[f for f in feedback if f['reliability_changed'] and f['has_next_query']]
    facts['feedback_chain']=dict(choices=len(vsel),live_notes=sum(bool(r['live_notes']) for r in vsel),
        menu_coverage=sum(r['any_menu_coverage'] for r in vsel),selective_menu_coverage=sum(r['any_selective_coverage'] for r in vsel),
        selected_forecast_choices=sum(r['selected_forecast_count']>0 for r in vsel),forecast_events=len(feedback),
        prior_feedback_note_menu_choices=sum(r['prior_feedback_note_covers_menu'] for r in vsel),
        prior_feedback_note_selected_choices=sum(r['prior_feedback_note_covers_selected'] for r in vsel),
        declared_prior_feedback_note_choices=sum(r['declared_note_with_prior_feedback'] for r in vsel),
        changed_reliability_with_next_query=len(changed_next),
        changed_reliability_carried=sum(r['reliability_carried_next_query'] for r in changed_next),
        changed_reliability_carried_next_menu_covered=sum(r['reliability_carried_next_query'] and r['same_note_next_menu_coverage']>0 for r in changed_next),
        changed_reliability_carried_next_selected_covered=sum(r['reliability_carried_next_query'] and r['same_note_next_selected_in_scope'] for r in changed_next),
        changed_reliability_carried_next_declared_use=sum(r['reliability_carried_next_query'] and r['same_note_next_declared_use'] for r in changed_next))
    facts['V13_record_categories']=dict(Counter(r['category'] for r in records if r['method']=='v13'))
    scored_versions=defaultdict(list)
    for f in feedback:scored_versions[f['task'],f['seed'],f['note_id'],f['version']].append(f)
    facts['KF_data_density']=dict(scored_note_versions=len(scored_versions),
        versions_with_only_one_scored_observation=sum(len(fs)==1 for fs in scored_versions.values()),
        median_forecasts_per_scored_version=median(len(fs) for fs in scored_versions.values()),
        median_Kalman_gain=median(f['kalman_gain'] for f in feedback),
        median_first_update_Kalman_gain=median(fs[0]['kalman_gain'] for fs in scored_versions.values()))
    mechanism_groups=[]
    for method in ('v13','reasoning_bo'):
        ss=[r for r in selections if r['method']==method and r['query']>1]
        for source in ('same_point','tied_acquisition','lower_acquisition'):
            rs=[r for r in ss if r['acquisition_comparison']==source]
            mechanism_groups.append(dict(method=method,dimension='acquisition',group=source,choices=len(rs),
                changed_choices=sum(r['changed_from_reference'] for r in rs),
                **{f'local_record_advantage_{k}':v for k,v in wtl(r['local_record_advantage'] for r in rs).items()},
                record_events=sum(r['selected_record'] for r in rs)))
    for group in ('no_selected_forecast','new_forecast','declared_prior_feedback'):
        rs=[r for r in vsel if r['query']>1 and (
            group=='no_selected_forecast' and not r['selected_forecast_count'] or
            group=='new_forecast' and r['selected_forecast_count'] and not r['declared_note_with_prior_feedback'] or
            group=='declared_prior_feedback' and r['declared_note_with_prior_feedback'])]
        changed=[r for r in rs if r['changed_from_reference']]
        mechanism_groups.append(dict(method='v13',dimension='feedback_context',group=group,choices=len(rs),
            changed_choices=len(changed),
            **{f'local_record_advantage_{k}':v for k,v in wtl(r['local_record_advantage'] for r in changed).items()},
            record_events=sum(r['selected_record'] for r in rs)))
    facts['observed_mechanism_groups']=mechanism_groups
    facts['first_KF_feedback_query_counts']=dict(Counter(str(r['first_KF_feedback_query']) for r in before_feedback if r['comparator']=='bo'))
    facts['conditional_after10']={comp:wtl(float(r['delta_continuation_SearchAUC']) for r in conditional if r['comparator']==comp)
                                  for comp in sorted({r['comparator'] for r in conditional})}
    assert facts['bo']['total_SearchAUC']==dict(wins=22,ties=1,losses=9)
    assert facts['reasoning_bo']['total_SearchAUC']==dict(wins=13,ties=5,losses=14)
    assert facts['feedback_chain']['selected_forecast_choices']==117
    for name,values in [('trajectory_decomposition.csv',trajectories),('paired_decomposition.csv',pairs),
        ('task_summary.csv',task_summary),('selection_diagnostics.csv',selections),('selection_summary.csv',selection_summary),
        ('record_sources.csv',records),('record_source_summary.csv',source_summary),
        ('first_divergence.csv',divergence),('feedback_chain.csv',feedback),
        ('before_first_KF_decomposition.csv',before_feedback),('observed_mechanism_groups.csv',mechanism_groups)]:csv_write(name,values)
    (OUT/'facts.json').write_text(json.dumps(facts,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    draw(task_summary)
    report(task_summary,facts,selection_summary)
    for p,hs in protected.items():assert sha((ROOT/p).read_bytes())==hs
    for p,hs in READS.items():assert sha((ROOT/p).read_bytes())==hs
    receipt=dict(status='PASS_READ_ONLY_EXACT_DECOMPOSITION',cold_trajectories=96,
        historical_selected_outcomes_verified=1920,native_commits_verified=len(commits),
        paired_rows=len(pairs),primary_comparisons=64,source_files_verified=len(READS),
        exported_rounds_cross_checked=1920,independent_area_and_record_increment_formulas_agree=True,
        cutoffs=[5,10,15],primary_cutoff=10,GP_fits=0,LLM_calls=0,new_experiment_queries=0,
        offline_oracle_access='post-hoc frozen-label evaluation of logged menus, chosen points and own-prefix GP references; pool ranges for evaluation only; no decision worker',
        causal_module_effect_identified=False,old_releases_unchanged=True,
        protected_artifacts=protected,analysis_source_sha256=sha(Path(__file__).read_bytes()))
    (OUT/'source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(receipt=receipt,facts=facts,task_summary=task_summary),ensure_ascii=False))


def draw(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(4,2,figsize=(11.6,12.4))
    for ax,task in zip(axes.flat,TASKS):
        rs=[next(r for r in summary if r['task']==task and r['comparator']==c) for c in ('bo','reasoning_bo')]
        for i,r in enumerate(rs):
            early=r['delta_early_origin_contribution'];late=r['delta_late_origin_contribution'];total=r['delta_SearchAUC']
            ax.bar(i,early,width=.42,color='#386FAD',edgecolor='#25405B',linewidth=.6)
            bottom=early if early*late>=0 else 0
            ax.bar(i,late,bottom=bottom,width=.42,color='#E7B350',hatch='///',edgecolor='#6D5221',linewidth=.6)
            ax.plot(i,total,'D',color='#232B35',markersize=5)
            ax.annotate(f'{total:+.2f}',(i,total),xytext=(20,0),textcoords='offset points',
                        va='center',fontsize=9,fontweight='bold')
        ax.axhline(0,color='#56616B',linewidth=.8)
        ax.set_xticks([0,1],['V13 − BO','V13 − RBO']);ax.set_xlim(-.6,1.85)
        ax.set_title(LABELS[task],loc='left',fontweight='bold')
        ax.set_ylabel('SearchAUC difference\n(native objective units)')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.margins(y=.22)
    fig.legend(handles=[Patch(facecolor='#386FAD',label='Record gains originating at queries 1–10'),
        Patch(facecolor='#E7B350',hatch='///',label='New record gains at queries 11–20'),
        Line2D([0],[0],marker='D',linestyle='',color='#232B35',label='Total SearchAUC difference')],
        loc='upper center',bbox_to_anchor=(.5,.965),ncol=1,frameon=False)
    fig.suptitle('Frozen V13 cold-start benefit decomposition',y=.994,fontweight='bold',fontsize=15)
    fig.text(.5,.015,'Four starts per task; all 20 queries retained. Exact arithmetic contributions, not causal module effects.\n'
             'Independent vertical scales; units: yield points, SnAr productivity, or HPLC peak area. RBO: local adapted Reasoning BO.',
             ha='center',fontsize=9)
    fig.subplots_adjust(top=.865,bottom=.08,hspace=.62,wspace=.32)
    fig.savefig(OUT/'benefit_decomposition.png',dpi=180,facecolor='white')
    fig.savefig(OUT/'benefit_decomposition.pdf',facecolor='white');plt.close(fig)


def report(summary,facts,selection_summary):
    def fwl(x):return f"{x['wins']}胜/{x['ties']}平/{x['losses']}负"
    by_task={r['task']:r for r in summary if r['comparator']=='bo'}
    wins=facts['bo']
    initial=next(r for r in selection_summary if r['method']=='v13' and r['task']=='all_tasks' and r['period']=='first_query')
    chain=facts['feedback_chain'];density=facts['KF_data_density']
    doyleno=sorted((r for r in facts['bo']['no_KF_feedback_runs'] if r['task']=='doyle'),key=lambda r:r['seed'])
    lines=['V13零初始化收益来源拆解（冻结结果，只读分析）','',
      '范围：8任务×4起始编号×3方法，共96条0＋20轨迹。未重跑LLM、GP或实验。',
      '主分界为第10次查询；另提供第5、15次分界的敏感性表，不据结果选择分界。','',
      '核心发现',
      f"1. 对BO的{wins['positive_total_comparisons']}组SearchAUC胜出中，{wins['positive_total_early_only']}组只有前10轮来源差为正，"
      f"{wins['positive_total_both']}组前后均正，{wins['positive_total_late_only']}组仅后半程差为正。优势多数已在前半程形成。",
      f"2. 32组中有{initial['changed_choices']}组在第1次选择就偏离BO；这时尚无GP或结构化知识反馈。"
      f"这些偏离在同一先验菜单内得到{initial['local_record_advantage_wins']}个更高、{initial['local_record_advantage_losses']}个更低的产出。",
      '3. Doyle以下起始轨迹全程没有结构化知识预测或KF反馈：'+
      '；'.join(f"起始{r['seed']}对BO的SearchAUC差{r['delta_SearchAUC']:+.4f}" for r in doyleno)+
      '。至少这些收益不依赖动态KF。',
      f"4. Perera后10轮来源差为{by_task['perera']['delta_late_origin_contribution']:+.4f}，但前10轮来源差"
      f"{by_task['perera']['delta_early_origin_contribution']:+.4f}；HPLC对应为"
      f"{by_task['hplc']['delta_late_origin_contribution']:+.4f}和{by_task['hplc']['delta_early_origin_contribution']:+.4f}。"
      '两者后期确有追赶、最终最佳均更高，但平均SearchAUC仍低于BO，不能只讲成功的一面。',
      '5. V13和Reasoning BO都能在本轮已有历史上改善GP首选的即时纪录，两者观察计数接近；该共同收益尚不能当作V13独有机制的证据。',
      f"6. {chain['choices']}次选择中，仅{chain['declared_prior_feedback_note_choices']}次声明使用已获得过反馈的知识；"
      f"已评分的{density['scored_note_versions']}个知识版本中{density['versions_with_only_one_scored_observation']}个只有一个评分。动态反馈作用范围和有效样本都有限。",
      '7. 现有第11轮消融只能判断继续更新的边际价值，不能判断前10轮模块是否必要。早期KF/刷新净效应仍未被因果识别。','',
      '一、精确指标拆分',
      'M0=b（各方法共用的目标下界）；ΔMj=Mj−M(j−1)。',
      'SearchAUC = b + Σj[(21−j)/20]ΔMj。',
      '前10轮来源贡献包括这些纪录在后10轮延续的面积；后10轮贡献只包含新纪录增量。',
      '配对差值=前10轮来源差值+后10轮新纪录来源差值；此等式不等于模块因果归因。','',
      '二、任务均值（每任务4个起始编号；数值单位与各任务原指标相同）',
      '任务 | 对照 | 总SearchAUC差 | 前10轮来源差 | 后10轮新纪录差 | 最终最佳差']
    for r in summary:
        lines.append(f"{LABELS[r['task']]} | {r['comparator']} | {r['delta_SearchAUC']:+.4f} | "
            f"{r['delta_early_origin_contribution']:+.4f} | {r['delta_late_origin_contribution']:+.4f} | {r['delta_final_best']:+.4f}")
    lines+=['','三、全部配对结果，不跨任务平均原始数值']
    for c in ('bo','reasoning_bo'):
        x=facts[c]
        lines.append(f"对{c}：总SearchAUC {fwl(x['total_SearchAUC'])}；前10轮来源 {fwl(x['early_origin'])}；后10轮来源 {fwl(x['late_origin'])}。")
        lines.append(f"总SearchAUC为正的{x['positive_total_comparisons']}组中：只有前10轮来源为正{x['positive_total_early_only']}组；"
                     f"只有后10轮来源为正{x['positive_total_late_only']}组；两部分均正{x['positive_total_both']}组。")
        lines.append(f"与{c}完整20次选点完全一致：{x['identical_full_choice_sequences']}组。")
        lines.append(f"总SearchAUC胜出的{x['positive_total_comparisons']}组中，首个KF反馈更新发生之前（含该次选点）的纪录来源差已为正："
            f"{x['positive_total_before_first_KF_positive']}组；仅此部分为正：{x['positive_total_only_before_first_KF_positive']}组。")
        for item in x['cutoff_sensitivity']:
            lines.append(f"敏感性分界q{item['cutoff']}：前段来源{fwl(item['early_origin'])}；后段来源{fwl(item['late_origin'])}；"
                f"总胜出组中前段来源为正{item['positive_total_with_positive_early_origin']}组。")
    lines+=['','四、在V13实际历史上，与其自身GP参考点做单步比较',
      '比较固定当轮已有历史，只用冻结标签计算两点各自能否提高incumbent。',
      '这不是重新跑一条BO轨迹，也不能把单步差值之和解释成长期方法收益。',
      '第1轮尚无GP，必须单独解释为相同先验菜单中的选择。']
    for r in selection_summary:
        if r['task']=='all_tasks' and r['period'] in ('all','first_query','early','late'):
            lines.append(f"{r['method']} {r['period']}：{r['choices']}次选择，{r['changed_choices']}次偏离参考点；"
                f"偏离中的即时纪录优势{r['local_record_advantage_wins']}正/{r['local_record_advantage_ties']}零/{r['local_record_advantage_losses']}负；"
                f"额外创纪录{r['creates_record_reference_would_not']}次，错过参考点可创纪录{r['misses_record_reference_would_make']}次。")
    chain=facts['feedback_chain']
    lines+=['','五、结构化反馈的参与机会（与性能收益不可等同）']
    lines.extend(f'{k}: {v}' for k,v in chain.items())
    lines+=['','以下六条V13轨迹全程没有结构化知识预测，因此也没有KF反馈更新：']
    lines.extend(f"{r['task']} 起始{r['seed']}，对BO的SearchAUC差{r['delta_SearchAUC']:+.4f}。"
                 for r in facts['bo']['no_KF_feedback_runs'])
    lines+=['','已发生评分的知识版本，其时间序列仍很短：']
    lines.extend(f'{k}: {v}' for k,v in facts['KF_data_density'].items())
    lines+=['','单步观察性分组（不同分组的候选难度、历史和适用范围不同，禁止当作模块效果）：']
    for r in facts['observed_mechanism_groups']:
        lines.append(f"{r['method']} {r['dimension']} {r['group']}：{r['choices']}次选择，"
            f"局部纪录优势{r['local_record_advantage_wins']}正/{r['local_record_advantage_ties']}零/{r['local_record_advantage_losses']}负，"
            f"其中实际创纪录{r['record_events']}次。")
    lines+=['','六、现有第11轮分叉消融']
    lines.extend(f"完整V13相对{c}：{fwl(v)}。" for c,v in facts['conditional_after10'].items())
    lines+=['固定KF仍保留事实反馈、推荐权限和编辑；停止刷新仍保留旧知识和KF更新。',
      '两个控制都继承第1～10轮的知识及状态，因此不能回答从第1轮不设模块是否更好。',
      '各分支使用独立LLM响应，每条件两次；差值也包含响应差异。','',
      '七、解释边界',
      '1. 原完整V13相对BO的收益不会被后期消融推翻；两者估计的问题不同。',
      '2. 早期来源包含初始LLM选点、早期Note/KF/刷新和GP路径改变，现有日志不能把它们因果拆开。',
      '3. 选点与自身GP一致，表示当时GP也会选到该点；该GP可能已受更早LLM选点的数据路径影响。',
      '4. 无选中点预测不代表没有自由文本Insight、警告或历史记忆的间接影响。',
      '5. 不能把按声明知识分组的产出差解释为知识效果；适用范围和实验难度不同。',
      '6. 需要从第1轮开始的匹配消融，才能进一步判断KF/刷新早期是否必要。',
      '7. 各自历史的后期增量有不同提升空间：早期已找到高值的策略，后期新增贡献可能更小。这不等于后期模块造成损害。',
      '8. 第1轮LLM选点的收益还不能区分化学先验、一般启发式和响应随机性；不能仅凭时序宣称化学知识得到验证。',
      '详细逐轮证据：selection_diagnostics.csv、record_sources.csv、feedback_chain.csv。',
      '全部来源和保护校验：source_hashes.json、receipt.json。']
    (OUT/'收益来源拆解.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')


if __name__=='__main__':
    main()
