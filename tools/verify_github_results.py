"""Check published metric arithmetic and derived maxima, using no private logs."""
from collections import defaultdict,Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-7),(a,b)


def cohort(folder,keys,n,metric='SearchAUC'):
    rows=read(folder+'/metrics.csv') if (ROOT/folder/'metrics.csv').exists() else read(folder+'/per_seed_results.csv')
    curves=defaultdict(list)
    for r in read(folder+'/running_best.csv'):curves[tuple(r[k] for k in keys)].append(r)
    assert len(curves)==len(rows)
    result={}
    for m in rows:
        key=tuple(m[k] for k in keys);points=sorted(curves[key],key=lambda r:int(r['query']))
        assert [int(r['query']) for r in points]==list(range(1,n+1))
        values=[float(r['cumulative_best']) for r in points]
        assert all(math.isfinite(x) for x in values) and all(a<=b+1e-10 for a,b in zip(values,values[1:]))
        close(math.fsum(values)/n,m[metric]);close(values[-1],m['final_best'])
        if n==16:assert values[0]>=float(m['initial_best'])-1e-9
        result[key]=values
    return rows,result


def bucket(value):return 'win' if value>1e-9 else 'loss' if value< -1e-9 else 'tie'


def verify_results():
    historical={};counts={}
    for regime,folder,n in [('warm','warm_4plus16',16),('cold','v13_cold_0plus20',20)]:
        rows,curves=cohort('results/cohorts/'+folder,('task','seed','method'),n)
        assert len(rows)==(131 if regime=='warm' else 96)
        lookup={(r['task'],r['seed'],r['method']):r for r in rows};historical[regime]=(rows,curves)
        for reference in ('bo','reasoning_bo'):
            c=Counter(bucket(float(r['SearchAUC'])-float(lookup[t,s,reference]['SearchAUC']))
                for (t,s,m),r in lookup.items() if m=='v13')
            assert sum(c.values())==32;counts[regime+'_V13_vs_'+reference]=dict(c)
    assert counts['warm_V13_vs_bo']=={'win':17,'loss':15}
    assert counts['warm_V13_vs_reasoning_bo']=={'win':11,'tie':7,'loss':14}
    assert counts['cold_V13_vs_bo']=={'win':22,'tie':1,'loss':9}
    assert counts['cold_V13_vs_reasoning_bo']=={'win':13,'tie':5,'loss':14}
    current,curves=cohort('results/grounded_baselines_20261105',('task','seed','method'),20)
    assert len(current)==160
    lookup={(r['task'],r['seed'],r['method']):r for r in current}
    for reference in ('bo','first'):
        c=Counter(bucket(float(r['SearchAUC'])-float(lookup[t,s,reference]['SearchAUC']))
            for (t,s,m),r in lookup.items() if m=='hyp')
        counts['current_hyp_vs_'+reference]=dict(c)
    assert counts['current_hyp_vs_bo']=={'win':23,'loss':9}
    assert counts['current_hyp_vs_first']=={'win':8,'tie':16,'loss':8}
    backend=Counter(r['backend'] for r in read('results/grounded_baselines_20261105/LLM_backend_by_round.csv'))
    assert backend=={'independent_collaboration_subagent':552,'openai_responses_api':664}
    for name,keys,size in [('kf_weight',('task','seed','method'),10),('kf_confirmation',('cell','arm'),15)]:
        rows,_=cohort('results/'+name,keys,20);assert len(rows)==size
    folder='results/cohorts/v13_late_and_transfer'
    remaining=defaultdict(list)
    for r in read(folder+'/continuation_running_best.csv'):remaining[r['job']].append(r)
    late=read(folder+'/final_metrics.csv');assert len(late)==75
    for m in late:
        rs=sorted(remaining[m['job']],key=lambda r:int(r['query']))
        n=int(m['new_queries']);prefix=int(m['prefix_queries'])
        assert len(rs)==n and [int(r['query']) for r in rs]==list(range(prefix+1,21))
        values=[float(r['cumulative_best']) for r in rs]
        prefix_values=historical['cold'][1][m['task'],m['prefix_seed'],'v13'][:prefix] if prefix else []
        close(max(prefix_values) if prefix_values else 0,m['prefix_best'])
        close(math.fsum(values)/n,m['continuation_SearchAUC'])
        close(math.fsum(prefix_values+values)/20,m['SearchAUC20']);close(values[-1],m['final_best'])
    effects=read(folder+'/conditional_effects.csv')
    for comparator in ('kf_fixed','no_refresh'):
        subset=[r for r in effects if r['comparator']==comparator]
        assert len(subset)==6 and all(float(r['delta_continuation_SearchAUC'])<=1e-9 for r in subset)
    master=read('results/combined_primary_metrics.csv')
    assert len(master)==487 and all(r['state_path'] and r['state_sha256'] and r['objective_unit'] for r in master)
    return dict(status='PASS_PUBLISHED_SUMMARY_ARITHMETIC',comparison_records=487,
        warm_4plus16_records=131,original_cold_records=96,continuation_and_transfer_records=75,
        KF_comparison_records=25,latest_records=160,SearchAUC_WTL=counts,
        scope='Arithmetic from published derived cumulative maxima; private raw outcomes, GP arrays and state chains are not replayed.',
        raw_native_audit_performed_in_this_checkout=False,new_experiments=0,new_GP_fits=0,new_LLM_calls=0)


def integrity():
    manifest=json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text(encoding='utf-8'))
    def check(row):
        path,expected=row
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,'Published file changed: '+path
        return 1
    with ThreadPoolExecutor(max_workers=8) as pool:n=sum(pool.map(check,manifest['files_sha256'].items()))
    return dict(status='PASS_PUBLISHED_FILE_INTEGRITY',managed_files=n,
        full_local_archive_required=False,raw_logs_or_checkpoints_uploaded=False)
