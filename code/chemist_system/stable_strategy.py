"""Read-only status, offline replay and per-decision inspection of the frozen strategy."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np

PROJECT=Path(__file__).resolve().parent


def main():
    os.chdir(PROJECT)
    from chem.benchmarks import stable_feedback_study as study,stable_feedback_report as report
    data=study.data
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['status','audit','inspect'],nargs='?',default='status')
    ap.add_argument('--tie-seed',type=int,choices=[0,1,2])
    ap.add_argument('--dataset',choices=list(data.SPECS),default='Perera_Suzuki')
    ap.add_argument('--round',type=int,choices=range(1,17),default=3);a=ap.parse_args()
    seeds=[a.tie_seed] if a.tie_seed is not None else [0,1,2]
    if a.action=='status':
        rows=[]
        for seed in seeds:
            variant=f'feedback_t{seed}';_,out=study.folders(variant)
            if (out/'results.json').exists():
                rows.extend({**r,'tie_seed':seed} for r in data.read(out/'results.json')['metrics'])
            else:
                rows.extend({'task':s['task'],'tie_seed':seed,'status':s['status'],'queries':len(s['decisions'])} for s in study.states(variant))
        group=study.OUT/'feedback_stability.json'
        result={'stability':data.read(group)['status'] if group.exists() else 'NOT_YET_AUDITED','rows':rows}
    elif a.action=='audit':
        result=[]
        for seed in seeds:
            variant=f'feedback_t{seed}';_,out=study.folders(variant)
            receipt=data.read(out/'release_receipt.json');inventory=data.read(out/'release_inventory.json')
            assert data.filehash(out/'release_inventory.json')==receipt['inventory_sha256']
            for f in inventory['files']:assert data.filehash(f['path'])==f['sha256'],f['path']
            target=out/'validation_receipt.json';before=data.filehash(target);captured=[]
            def capture(path,value):
                if Path(path).resolve()!=target.resolve():raise RuntimeError('Offline audit refused an unexpected artifact write: '+str(path))
                captured.append(value)
            # Recompute all choices/KF/GP information without rewriting sealed receipts.
            with patch.object(data,'write',capture):checked=report.validate(variant)
            assert len(captured)==1 and captured[0]==checked and data.filehash(target)==before
            result.append({k:checked[k] for k in ('status','variant','trajectories','choices','KF_updates','declared_acquisition_gate_exceptions')})
    else:
        seed=a.tie_seed if a.tie_seed is not None else 0;variant=f'feedback_t{seed}';root,_=study.folders(variant)
        pool=data.load_pool(a.dataset);state=next(s for s in study.states(variant) if s['task']==a.dataset)
        event=state['decisions'][a.round-1];j=pool['ids'].index(event['selected_id'])
        folder=root/'runs'/state['run_id'];table=folder/'rounds'/f'{a.round:02d}_candidates.npz'
        contexts=[p for p in (root/'requests').glob('*/context.json') if data.read(p)['task']==a.dataset and data.read(p)['public']['query_round']==a.round]
        with np.load(table) as z:
            candidate={'id':event['selected_id'],'conditions':pool['points'][j],
                'GP_mean_percent':float(z['mu_y'][j])*100,'GP_latent_SD_pp':float(z['sd_y'][j])*100,
                'base_qLogEI':float(z['base_qLogEI'][j]),'guided_qLogEI':float(z['guided_qLogEI'][j]),
                'information_proxy':float(z['information_proxy'][j]),'normal_gate_eligible':bool(z['eligible'][j]),
                'observed_percent_after_commit':event['y']*100}
        result={'variant':variant,'dataset':a.dataset,'query_round':a.round,'mode':event['execution_mode'],
            'selected':candidate,'normal_proposal':event['normal_R2_proposal_id'],'budget':event['budget'],
            'base_opportunity_loss':event['base_opportunity_loss'],'gate_exception':event['acquisition_gate_exception'],
            'B4':{'retained':event['retained'],'revoked':event['diagnosis']['revoked_ids'],'conflict_kind':event['diagnosis']['conflict_kind']},
            'KF':{'amplitude':event['amplitude'],'measurement':event['measurement'],'after':event['calibration_after']},
            'all_candidate_arrays':str(table.resolve()),'committed_before_observation':str((folder/'rounds'/f'{a.round:02d}_committed.json').resolve()),
            'LLM_contexts':[str(p.resolve()) for p in contexts],
            'LLM_validated_answers':[str((p.parent/'validated.json').resolve()) for p in contexts]}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
