"""Stdlib-only independent arithmetic; does not import experiment code."""
import json
from pathlib import Path
from statistics import mean,stdev
from math import isclose
from hashlib import sha256


def main():
    out=Path('research/suzuki_official')
    result=json.loads((out/'results.json').read_text(encoding='utf8'))
    actual=[];maximum_error=0.
    for f in sorted(Path('data/runs/suzuki_official/runs').glob('*/state.json')):
        s=json.loads(f.read_text(encoding='utf8'))
        observations=[h['y']*100 for h in s['history']]
        curve=[max(observations[:min(b,len(observations))],default=0.) for b in range(1,21)]
        independent={'run_id':s['run_id'],'case':s['case'],'method':s['method'],
            'search_auc_pct':mean(curve[4:]),'final_best_pct':curve[-1],
            'initial_best_pct':curve[3],'queries':len(observations),'status':s['status']}
        stored=next(r for r in result['runs'] if r['run_id']==s['run_id'])
        for name in ('search_auc_pct','final_best_pct','initial_best_pct'):
            err=abs(independent[name]-stored[name]);maximum_error=max(maximum_error,err)
            assert isclose(independent[name],stored[name],rel_tol=0,abs_tol=1e-12)
        assert curve==stored['best_so_far_pct']
        actual.append(independent)
    assert len(actual)==60
    checks=0
    for case in (None,1,2,3,4):
        for method in ('LLINBO','LGBO','HEBO'):
            subset=[r for r in actual if r['method']==method and (case is None or r['case']==case)]
            recorded=result['overall'][method] if case is None else result['by_case'][str(case)][method]
            assert len(subset)==recorded['n']
            for key in ('search_auc_pct','final_best_pct','initial_best_pct'):
                for label,value in [('mean',mean(r[key] for r in subset)),('sd',stdev(r[key] for r in subset))]:
                    maximum_error=max(maximum_error,abs(value-recorded[key][label]))
                    assert abs(value-recorded[key][label])<1e-12
                    checks+=1
    receipt={'status':'PASS','implementation':'stdlib only; max over raw history at every charged budget position',
        'raw_trajectories':len(actual),'budget_positions':1200,'additional_auc_positions':960,
        'aggregate_mean_sd_checks':checks,'max_error_percentage_points':maximum_error,
        'results_sha256':sha256((out/'results.json').read_bytes()).hexdigest()}
    (out/'independent_metrics.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
