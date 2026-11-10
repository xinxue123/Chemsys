"""Rebuild paper-related saved-evidence analyses; no API calls, GP fits or queries."""
from contextlib import redirect_stdout
import argparse
import importlib
import io
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/runs/hypothesis_scope_v2_20261105'
sys.path.insert(0,str(ROOT/'code/analysis'))

def run_native(module,args):
    previous=sys.argv
    sys.argv=[module,*map(str,args)]
    try:importlib.import_module(module).main()
    finally:sys.argv=previous

def development():
    run_native('analyze_hypothesis_scope_method',[])

def heldout():
    for module,folder in [
        ('analyze_hypothesis_scope_v2_heldout','hypothesis_scope_v2_heldout_20261105'),
        ('analyze_hypothesis_scope_v2_lgbo','hypothesis_scope_v2_lgbo_heldout_20261106'),
        ('analyze_hypothesis_scope_v2_heldout_named','hypothesis_scope_v2_heldout_named_20261107'),
        ('analyze_hypothesis_scope_v2_first_point_streams','hypothesis_scope_v2_first_point_streams_20261107'),
        ('analyze_hypothesis_scope_v2_shared_start','hypothesis_scope_v2_shared_start_20261107'),
        ('analyze_hypothesis_scope_v2_cpa','hypothesis_scope_v2_cpa_20261107'),
        ('analyze_hypothesis_scope_v2_paper_bo','hypothesis_scope_v2_paper_bo_20261107'),
    ]:run_native(module,[BASE,ROOT/'results'/folder])

def screen():
    run_native('analyze_hypothesis_scope_v2_screen',[BASE,ROOT/'results/hypothesis_scope_v2_screen_20261107'])

def first_points():
    # Test F is retained in the condensed paper; unreported S/G sweeps are not run.
    m=importlib.import_module('analyze_hypothesis_scope_v2_gp_followups')
    out=ROOT/'results/hypothesis_scope_v2_gp_followups_20261106';out.mkdir(parents=True,exist_ok=True)
    rows,result=m.test_f(BASE,m.tops(),random.Random(m.SEED))
    m.write_rows(out/'first_point_runs.csv',rows)
    (out/'gp_followups.json').write_text(json.dumps({'test_F':result},indent=2)+'\n',encoding='utf-8')
    print('Test F:',{k:v.get('gap_outside_minus_inside') for k,v in result.items() if isinstance(v,dict)})

def knowledge():
    # Call the unchanged claim, translation and forecast analysis functions.
    # Unreported controller-variant sweeps stay outside this clean project.
    m=importlib.import_module('analyze_hypothesis_scope_v2_mechanism')
    out=m.FOLDER;out.mkdir(parents=True,exist_ok=True)
    coding=m.read(out/'claim_coding.json')['hypotheses']
    pools=m.load_pools(BASE);runs=m.histories(BASE)
    claims,contrasts=m.audit(pools,runs,coding)
    m.write_rows(out/'claims.csv',claims);m.write_rows(out/'contrasts.csv',contrasts)
    rows,translated=m.translation(pools,coding,claims);m.write_rows(out/'translation.csv',rows)
    rows,forecasts=m.forecast_quality(pools,runs,random.Random(m.SEED));m.write_rows(out/'forecasts_by_task.csv',rows)
    rows,ceiling=m.ceilings(pools,BASE);m.write_rows(out/'ceilings.csv',rows)
    summary=dict(status='EXPLORATORY_SAVED_EVIDENCE',new_model_calls=0,new_GP_fits=0,new_oracle_queries=0,
                 claims=m.summarize_claims(claims,contrasts,coding),translation=translated,
                 forecasts=forecasts,ceilings=ceiling,
                 variance={t:m.variance(pools[t]) for t in ('aryl','doyle','perera')})
    (out/'mechanism.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Knowledge audit:',len(claims),'hypotheses;',len(contrasts),'contrasts')

JOBS=dict(development=development,heldout=heldout,screen=screen,first_points=first_points,knowledge=knowledge)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=[*JOBS,'all'],default='all',nargs='?')
    args=parser.parse_args()
    if not BASE.exists() or not (ROOT/'archives/run__hypothesis_scope_loop_20261105.zip').exists():
        raise SystemExit('This native saved-evidence rebuild needs local records. See LOCAL_DATA_REQUIREMENTS.md.')
    for name in JOBS if args.part=='all' else [args.part]:
        capture=io.StringIO()
        with redirect_stdout(capture):JOBS[name]()
        log=ROOT/'output/rebuild_logs'/f'{name}.txt';log.parent.mkdir(parents=True,exist_ok=True)
        log.write_text(capture.getvalue(),encoding='utf-8')
        print(f'PASS {name}')

if __name__=='__main__':main()
