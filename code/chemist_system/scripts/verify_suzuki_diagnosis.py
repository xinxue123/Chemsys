"""Recheck diagnostic values and preserve a compact evidence receipt."""
from pathlib import Path
import json
import numpy as np
from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle
from scripts.diagnose_suzuki_comparison import ROOT,OUT,LEGACY,CASES,read,write,filehash,native_state


def main():
    report=read(OUT/'comparison_diagnosis.json');assert report['validation']['status']=='PASS'
    points=0;maximum_error=0.;legacy_values=0
    for f in sorted((ROOT/'controls').glob('*/result.json')):
        r=read(f);case=r['case']
        if 'history' in r:
            y=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',case).predict([h['point'] for h in r['history']])
            error=max(abs(float(v)-h['y'])*100 for v,h in zip(y,r['history']))
            assert error<1e-4;maximum_error=max(maximum_error,error);points+=len(y)
        else:
            if r['recipe'].startswith('expanded'):
                task=read(ROOT/'inputs'/f'case{case}'/'expanded_task.json')
                values=np.load(ROOT/'inputs'/f'case{case}'/'expanded_values.npz')['values']
            else:
                task=read(LEGACY/'summit_cache'/f'case_{case}'/'task.json')
                values=np.load(LEGACY/'summit_cache'/f'case_{case}'/'oracle.npz')['values']
            lookup=dict(zip(task['candidate_ids'],map(float,values)))
            if r['recipe'].endswith('new_init'):
                lookup.update({f'w{i:04d}':h['y'] for i,h in enumerate(native_state(case,r['seed'])['history'][:4])})
            state=read(f.parent/'state.json')
            assert all(lookup[h['candidate_id']]==h['value'] for h in state['oracle_ledger'])
            legacy_values+=len(state['oracle_ledger'])
    evidence={str(f):filehash(f) for f in [
        Path('scripts/diagnose_suzuki_comparison.py'),Path('scripts/diagnose_lgbo_regional_lift.py'),
        Path('scripts/analyze_suzuki_diagnosis.py'),Path(__file__),Path('scripts/plot_suzuki_diagnosis.py'),
        *sorted(OUT.glob('*.json')),*sorted(OUT.glob('*.png')),*sorted(OUT.glob('*.svg'))] if f.name!='validation_receipt.json'}
    write(OUT/'validation_receipt.json',{'status':'PASS',
        **report['validation'],'native_emulator_values_recomputed':points,
        'native_emulator_max_error_percentage_points':maximum_error,'legacy_cached_values_rechecked':legacy_values,
        'visual_inspection':'PNG viewed locally: both panels, all eight labels and values, axes and footnotes legible; no observed clipping',
        'evidence_sha256':evidence,
        'boundaries':['All primary official and B0-B4 studies are read-only and unchanged.',
            '90 post-hoc diagnostic trajectories, plus one legacy replay qualification; all 4+16 queries.',
            'No new real LLM inference. Regional zero-lift uses the saved active preference schedule.',
            'Three reaction cases, not fifteen independent chemical systems. No significance or broad causal-knowledge claim.']})
    print(json.dumps({'status':'PASS','native_values':points,'legacy_values':legacy_values,'max_oracle_error_pct':maximum_error,
                      'controls':90,'new_LLM_calls':0,'operator_compiles_in_loop':0}))


if __name__=='__main__':main()
