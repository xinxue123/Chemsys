"""Seal a fully audited study; never alter experiments or previous controls."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
OUT=ROOT/'experiments/reports/grounded_baselines_20261105'
REVIEW=ROOT/'experiments/operations/grounded_baselines_20261105/figure_review.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_once(path,value):
    text=json.dumps(value,indent=2)+'\n'
    if path.exists():
        assert read(path)==value,'Existing seal differs'
    else:path.write_text(text,encoding='utf-8')


def seal():
    audit=read(OUT/'audit.json');p=read(BASE/'protocol.json')
    assert audit['status']=='PASS_COMPLETE_SAVED_EVIDENCE_AUDIT'
    assert audit['completed_new']==64 and audit['checked_committed_experiments']==1280
    assert audit['checked_fresh_LLM_requests']==1216
    assert audit['protocol_sha256']==p['protocol_sha256']
    assert read(OUT/'pending.json')==[]
    assert len(list(BASE.glob('runs/*/*/completion.json')))==64
    for cp in BASE.glob('runs/*/*/completion.json'):
        assert read(cp)==dict(status='COMPLETE',queries=20,LLM_calls=19)
    assert not list(BASE.glob('runs/*/*/execution_error.json'))
    review=read(REVIEW)
    assert review['status']=='PASS_RENDERED_FIGURE_REVIEW'
    assert review['PNG_sha256']==sha(OUT/'cumulative_best.png')
    analysis_paths=[ROOT/'code/tools/report_grounded_baselines.py',ROOT/'code/tools/audit_grounded_baselines.py',
        ROOT/'code/tools/seal_grounded_baselines.py',ROOT/'code/tools/finalize_grounded_baselines.py',
        ROOT/'docs/grounded_baselines_execution_notes_20261105.txt']
    for source in analysis_paths:
        dest=OUT/'analysis_source_snapshot'/source.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==sha(source)
        else:shutil.copyfile(source,dest)
    files={}
    for folder in (BASE,ROOT/'experiments/agent_io/grounded_baselines_20261105',OUT):
        for path in folder.rglob('*'):
            if not path.is_file() or path.suffix in ('.lock','.tmp','.pyc'):continue
            if path.name in ('completed_artifacts_sha256.json','completion_receipt.json'):continue
            files[path.relative_to(ROOT).as_posix()]=sha(path)
    value=dict(protocol_sha256=p['protocol_sha256'],files=files)
    write_once(BASE/'completed_artifacts_sha256.json',value)
    target=BASE/'completion_receipt.json'
    if target.exists():
        receipt=read(target)
        assert receipt['manifest_sha256']==sha(BASE/'completed_artifacts_sha256.json')
    else:
        receipt=dict(status='COMPLETE_MATCHED_GROUNDED_BASELINES',sealed_UTC=datetime.now(timezone.utc).isoformat(),
            completed_new_trajectories=64,paid_experimental_evaluations=1280,logical_post_first_LLM_requests=1216,
            actual_fresh_LLM_attempts=audit['actual_fresh_LLM_attempts'],schema_only_repairs=audit['schema_only_repairs'],
            frozen_control_trajectories=96,protocol_sha256=p['protocol_sha256'],
            manifest_sha256=sha(BASE/'completed_artifacts_sha256.json'),
            audit_sha256=sha(OUT/'audit.json'),figure_review_sha256=sha(REVIEW),all_planned_trajectories_included=True,
            inference='Conditional on shared fixed initial LLM point and initial knowledge; numerical seed0 and exact-tie seeds0..3; exploratory adapted mechanisms, not published-protocol reproduction')
        write_once(target,receipt)
    return dict(status=receipt['status'],hashed_artifacts=len(files),actual_fresh_LLM_attempts=audit['actual_fresh_LLM_attempts'])


if __name__=='__main__':
    if not REVIEW.exists():print('WAITING_FOR_RENDERED_FIGURE_REVIEW',flush=True)
    while not REVIEW.exists():time.sleep(10)
    print(json.dumps(seal()))
