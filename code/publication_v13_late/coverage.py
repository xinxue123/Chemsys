"""Post-hoc candidate coverage and next-step visibility, not causal attribution."""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study
from cold_start_v13 import adapter,control

P=ROOT/'paper/v13_late'
FULL={s:t for t,s in study.SHORT.items()}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,values):
    with (P/'data'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def main(fresh=False):
    specs=[]
    for r in rows(P/'data/metrics.csv'):
        if r['method']!='v13':continue
        cohort='cold' if r['regime']=='cold' else 'warm_development' if r['seed']=='0' else 'warm_primary'
        specs.append(dict(job=f"historical_{r['regime']}_{r['task']}{r['seed']}",cohort=cohort,task=r['task'],seed=r['seed'],
            start=1,state_path=r['state_path']))
    if fresh:
        for r in rows(P/'data/final_metrics.csv'):
            if r['arm'] in ('bo','rbo'):continue
            specs.append(dict(job=r['job'],cohort='new_'+r['kind']+'_'+r['arm'],task=r['task'],seed=r['prefix_seed'],
                start=11 if r['kind']=='prefix' else 1,state_path=r['state_path']))
    pools={};query_rows=[];note_rows=[];carry_rows=[];first_packets=defaultdict(list)
    for spec in specs:
        if spec['task'] not in pools:pools[spec['task']]=study.pool_for(FULL[spec['task']])
        pool=pools[spec['task']]
        sf=ROOT/spec['state_path'];state=read(sf)
        commits={d['query']:read(sf.parent/'rounds'/f"{d['query']:02}_commit.json") for d in state['decisions'] if d['query']>=spec['start']}
        for d in state['decisions']:
            q=d['query']
            if q not in commits:continue
            c=commits[q];live=[n for n in c['edited_notes'] if n['recommendation']['permission']!='withdrawn']
            expected={(n['id'],n['version']) for n in live if control.in_scope(n['scope'],adapter.conditions(pool,c['selected_id']))}
            actual={(n['id'],n['version']) for n in c['decision']['forecasts']}
            if expected!=actual:raise ValueError('Coverage reconstruction disagrees with prospective forecasts')
            counts=[]
            for n in live:
                covered=[cid for cid in c['display_ids'] if control.in_scope(n['scope'],adapter.conditions(pool,cid))]
                counts.append(len(covered))
                selected_in=c['selected_id'] in covered;GP_in=c['BO_reference_id'] in covered
                note_rows.append(dict(job=spec['job'],cohort=spec['cohort'],task=spec['task'],seed=spec['seed'],query=q,
                    note_id=n['id'],version=n['version'],effect=n['effect'],permission=n['recommendation']['permission'],
                    candidates_covered=len(covered),selected_in_scope=int(selected_in),GP_top_in_scope=int(GP_in),
                    selected_is_GP_top=int(c['selected_id']==c['BO_reference_id']),
                    promote_choice_in_scope_GP_top_out=int(n['effect']=='promote' and selected_in and not GP_in),
                    avoid_choice_outside_GP_top_in=int(n['effect']=='avoid' and not selected_in and GP_in)))
            query_rows.append(dict(job=spec['job'],cohort=spec['cohort'],task=spec['task'],seed=spec['seed'],query=q,
                live_notes=len(live),any_menu_coverage=int(any(counts)),any_selective_coverage=int(any(0<n<5 for n in counts)),
                any_all_five_coverage=int(5 in counts),selected_forecast_count=len(d['feedback']),
                selected_id=c['selected_id'],GP_top_id=c['BO_reference_id']))
            if spec['cohort'].startswith('new_prefix') and q==11 and spec['cohort']!='new_prefix_coupled':
                first_packets[spec['task'],spec['seed'],c['reasoning']['context_sha256']].append(dict(job=spec['job'],selected_id=c['selected_id']))
            if q+1 in commits:
                nc=commits[q+1];by_key={(n['id'],n['version']):n for n in nc['edited_notes']}
                for event in d['feedback']:
                    packet=next(f for f in c['decision']['forecasts'] if f['commit_sha256']==event['commit_sha256'])
                    n=by_key.get((packet['id'],packet['version']));after=event['calibration']['after']
                    retained=n is not None and n['recommendation']['permission']!='withdrawn'
                    carried=retained and n['calibration']['reliability']==after['reliability']
                    count=sum(control.in_scope(n['scope'],adapter.conditions(pool,cid)) for cid in nc['display_ids']) if retained else 0
                    carry_rows.append(dict(job=spec['job'],cohort=spec['cohort'],task=spec['task'],query=q,note_id=packet['id'],version=packet['version'],
                        actual_reliability_changed=int(abs(after['reliability']-event['calibration']['before']['reliability'])>1e-12),
                        version_live_next_query=int(retained),reliability_retained_next_query=int(carried),next_menu_candidates_covered=count))
    summary=[]
    for cohort in sorted({r['cohort'] for r in query_rows}):
        qs=[r for r in query_rows if r['cohort']==cohort];ns=[r for r in note_rows if r['cohort']==cohort];cs=[r for r in carry_rows if r['cohort']==cohort and r['actual_reliability_changed']]
        summary.append(dict(cohort=cohort,queries=len(qs),queries_with_any_live_note=sum(bool(r['live_notes']) for r in qs),
            queries_with_any_menu_coverage=sum(r['any_menu_coverage'] for r in qs),queries_with_selective_coverage=sum(r['any_selective_coverage'] for r in qs),
            queries_with_all_five_note=sum(r['any_all_five_coverage'] for r in qs),
            queries_with_selected_forecast=sum(bool(r['selected_forecast_count']) for r in qs),
            live_note_menu_pairs=len(ns),zero_coverage_pairs=sum(r['candidates_covered']==0 for r in ns),
            selective_coverage_pairs=sum(0<r['candidates_covered']<5 for r in ns),all_five_coverage_pairs=sum(r['candidates_covered']==5 for r in ns),
            promote_selected_GP_outside_associations=sum(r['promote_choice_in_scope_GP_top_out'] for r in ns),
            avoid_selected_outside_GP_inside_associations=sum(r['avoid_choice_outside_GP_top_in'] for r in ns),
            changed_reliability_events_with_next_query=len(cs),changed_reliability_live_and_visible_next_query=sum(r['reliability_retained_next_query'] for r in cs),
            changed_reliability_with_next_menu_coverage=sum(r['reliability_retained_next_query'] and r['next_menu_candidates_covered']>0 for r in cs)))
    write('candidate_coverage_queries.csv',query_rows);write('candidate_coverage_notes.csv',note_rows);write('candidate_coverage_summary.csv',summary)
    write('next_query_reliability_visibility.csv',carry_rows)
    packet_groups=[dict(task=k[0],prefix_seed=k[1],public_context_sha256=k[2],members=v,
        distinct_selected_points=len({x['selected_id'] for x in v})) for k,v in first_packets.items() if len(v)>1]
    result=dict(status='COMPLETE_POST_HOC_COVERAGE',fresh_included=fresh,queries=len(query_rows),
        repeated_identical_first_query_packet_groups=len(packet_groups),
        identical_packet_groups_with_different_selected_point=sum(r['distinct_selected_points']>1 for r in packet_groups),
        first_query_packet_groups=packet_groups,
        limitations=['Scope is a region, not a causal variable contrast','A note covering all five can still provide free-form context',
            'Avoidance can select outside a Note scope without a selected-point forecast or valid in-scope declared reference',
            'Post-hoc associations and visible state do not identify performance effects'],new_queries=0,new_GP_fits=0,new_LLM_calls=0)
    (P/'data/candidate_coverage_receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='first_query_packet_groups'}))


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--fresh',action='store_true');args=a.parse_args();main(args.fresh)
