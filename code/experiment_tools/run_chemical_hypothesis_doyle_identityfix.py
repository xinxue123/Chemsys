"""Prospectively register an isolated Doyle-only chemical-name correction.

Frozen parent code, algorithm, pools, GP, initialization and budgets are retained.
Only the erroneous common-name annotation for the unchanged SMILES is corrected.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from copy import deepcopy
import argparse
import json
import os
import sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/tools')]
from chemical_hypothesis_bo import study as s,transport as t,background,hypotheses as h
VARIANT='chemical_hypothesis_bo_doyle_identityfix_20261105'
BASE=ROOT/'experiments/runs'/VARIANT;REPORT=ROOT/'experiments/reports'/VARIANT
PARENT=s.BASE;PARENT_REPORT=s.REPORT;ORIGINAL_REQUEST=t.request
TASK='Doyle_Buchwald_Hartwig';SMILES='CN1CCCN2C1=NCCC2'
CORRECT='7-methyl-1,5,7-triazabicyclo[4.4.0]dec-5-ene (MTBD)'


def isolated_request(task,kind,query,public,validator):
    return ORIGINAL_REQUEST(task,'idfix_'+kind,query,public,validator)


def configure():
    background.COMMON_NAMES[SMILES]=CORRECT
    s.BASE=BASE;s.REPORT=REPORT
    t.BASE=BASE.resolve();t.VARIANT=VARIANT;t.request=isolated_request


def initialize():configure();s.initialize()


def prepare():
    parent=s.check(True)
    source=ROOT/'code/experiment_tools/run_chemical_hypothesis_doyle_identityfix.py'
    changes=[]
    for task in s.native.TASKS:
        pool=s.native.load_task(ROOT,task,seed=0)
        before=background.context(pool);background.COMMON_NAMES[SMILES]=CORRECT
        after=background.context(pool);background.COMMON_NAMES[SMILES]='1,8-diazabicyclo[5.4.0]undec-7-ene (DBU)'
        if before!=after:changes.append(task)
    assert changes==[TASK]
    graph=background.molecular_graph(SMILES)
    assert graph['formula']=='C8H15N3' and graph['atom_counts']['N']==3
    protected=dict(parent['protected_sha256']);protected.update(parent['sources_sha256'])
    for folder in (PARENT/'requests',PARENT/'runs'):
        for path in folder.rglob('*.json'):protected[path.relative_to(ROOT).as_posix()]=s.sha(path)
    protected[(PARENT/'protocol.json').relative_to(ROOT).as_posix()]=s.sha(PARENT/'protocol.json')
    p=deepcopy(parent);p.pop('protocol_sha256')
    p.update(variant=VARIANT,created_UTC=datetime.now(timezone.utc).isoformat(),jobs=[(TASK,a) for a in ('fixed','adaptive')],
        sources_sha256=dict(parent['sources_sha256'],**{source.relative_to(ROOT).as_posix():s.sha(source)}),
        protected_sha256=protected,maximum_new_queries=32,maximum_logical_LLM_requests=4,
        correction=dict(task=TASK,unchanged_SMILES=SMILES,old_incorrect_name='DBU',corrected_name=CORRECT,
            formula=graph['formula'],primary_source='https://pubchem.ncbi.nlm.nih.gov/compound/123583',
            parent_protocol_sha256=parent['protocol_sha256'],
            registered_after_parent_results=True,
            reason='Common-name-to-structure inconsistency; no algorithm or decision-parameter change.',
            limitation='Fresh independent responses differ; outcome changes cannot be attributed solely to the corrected name.'))
    p['protocol_sha256']=h.digest(p)
    s.write(BASE/'protocol.json',p)
    configure();pool=s.native.load_task(ROOT,TASK,seed=0)
    s.write(REPORT/'corrected_background.json',background.context(pool))
    s.write(REPORT/'correction.json',p['correction'])
    s.check(True)
    return dict(status='REGISTERED_CORRECTION',protocol_sha256=p['protocol_sha256'],task=TASK)


def audit():
    import audit_chemical_hypothesis_bo as a
    configure();s.check(True);s.initialize();assert not t.pending()
    rows=a.one(TASK)
    agents=[]
    for path in (BASE/'requests').glob('*/*/attempt*/response_receipt.json'):
        receipt=s.read(path);af=path.parent;assignment=s.read(af/'assignment.json');export=s.read(af/'export.json');req=s.read(af/'request.json')
        assert receipt['raw_sha256']==s.sha(af/'response.raw.json')
        assert receipt['assignment_sha256']==s.sha(af/'assignment.json')
        assert receipt['request_sha256']==req['request_sha256']
        assert s.read(ROOT/export['output'])==s.read(af/'response.raw.json')
        assert s.sha(ROOT/export['input'])==export['input_sha256']
        assert assignment['spawn_request']==s.read(af/'invocation.json')
        assert assignment['spawn_result']==dict(task_name=assignment['agent_id'])
        assert assignment['spawn_request']['fork_turns']=='none'
        agents.append(assignment['agent_id'])
    assert len(agents)==len(set(agents))<=4
    s.write(REPORT/'audit.json',dict(status='PASS_CORRECTED_DOYLE_REPLAY',rows=rows,fresh_agents=agents,
        decisions_replayed=32,protected_artifacts_verified=len(s.check(True)['protected_sha256']),
        corrected_background_sha256=s.sha(REPORT/'corrected_background.json')))
    return dict(status='PASS_CORRECTED_DOYLE_REPLAY',rounds=32,LLM_calls=len(agents))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','pending','assign','audit'))
    parser.add_argument('--spawned',nargs='*',default=[]);args=parser.parse_args()
    if args.action=='prepare':value=prepare()
    else:
        configure()
        if args.action=='run':
            s.check(True)
            with ProcessPoolExecutor(max_workers=2,initializer=initialize) as executor:
                value=[f.result() for f in [executor.submit(s.run_one,TASK,a) for a in ('fixed','adaptive')]]
        elif args.action=='audit':value=audit()
        else:
            for pending in t.pending():
                name=pending['invocation']['task_name']
                if name in args.spawned and not pending['assigned']:t.assign(pending['request'],dict(task_name='/root/'+name))
                if (Path(pending['request']).parent/'assignment.json').exists() and pending['output_exists']:t.seal(pending['request'])
            value=[dict(name=p['invocation']['task_name'],assigned=p['assigned'],ready=p['output_exists'],
                export=s.read(Path(p['request']).parent/'export.json')) for p in t.pending()]
    print(json.dumps(value))


if __name__=='__main__':main()
