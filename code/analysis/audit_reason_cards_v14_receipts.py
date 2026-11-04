"""Audit-only raw-byte ordering adapter; frozen V14 scientific code is untouched.

The registered replay read canonical commit copies before rebuilding embedded
JSON prompt strings. Rehydrate raw authored responses, as frozen V13 replay does,
before replaying feedback. Assert canonical commitment content is identical.
No experiment, model fit, LLM call, response editing, or oracle read occurs here.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/experiment_tools'),str(ROOT/'code/chemist_system')]
from reason_cards_v14 import study,transport,results,policy
from cold_start_v13 import control

NATIVE_FEEDBACK=study.feedback


def feedback_from_raw(pool,state,commit,observation):
    fixed=deepcopy(commit)
    if commit['memory']:
        raw=transport.read((ROOT/commit['memory']['request_relpath']).parent/'response.raw.json')
        plan=policy.apply_edits(raw,pool,state,commit['query'])
        fixed['memory']['raw']=raw;fixed['memory']['plan']=plan
        notes=plan['notes']
    else:notes=state['notes']
    raw=transport.read((ROOT/commit['reasoning']['request_relpath']).parent/'response.raw.json')
    folder=study.run_path(ROOT,state['task'],state['starting_seed'])
    with np.load(folder/f"rounds/{commit['query']:02d}_GP.npz",allow_pickle=False) as z:
        context=dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
            **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})
    packet=policy.validate_decision(ROOT,pool,notes,state,context,commit['display_ids'],commit['query'],raw)
    fixed['reasoning']['raw']=raw;fixed['reasoning']['plan']=packet
    fixed['decision']=packet;fixed['edited_notes']=notes
    # Only dictionary insertion order may differ, never commitment semantics.
    assert control.digest(fixed)==control.digest(commit)
    return NATIVE_FEEDBACK(pool,state,fixed,observation)


def replay_prefix(root,task,seed):
    protocol=study.check(root);state=transport.read(study.run_path(root,task,seed)/'state.json')
    with patch.object(study,'feedback',side_effect=feedback_from_raw):
        rows=results.replay(root,task,seed,state,protocol)
    return dict(task=task,seed=seed,prefix_replayed=len(rows),
                card_updates=state['decision_channel']['calibration']['updates'])


def audit(root=ROOT):
    with patch.object(study,'feedback',side_effect=feedback_from_raw):
        result=results.audit(root)
    result['audit_ordering_adapter']=dict(path=Path(__file__).relative_to(ROOT).as_posix(),
        sha256=transport.file_sha(__file__),scientific_commit_canonical_content_unchanged=True,
        purpose='Rehydrate immutable raw response object order before embedded prompt reconstruction')
    out=Path(root)/'experiments/reports'/study.VARIANT
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'audit_ordering_adapter.txt').write_text(
        'The frozen scientific implementation and protocol remain unchanged.\n'
        'Registered replay used sorted commit copies of raw responses, changing JSON object order inside prompt strings.\n'
        'This audit-only adapter reconstructs feedback from original response.raw.json bytes, verifies identical canonical\n'
        'commitments, then uses the original replay. No outcomes, selections, response bytes or parameters are changed.\n'
        'Run: python code/analysis/audit_reason_cards_v14_receipts.py\n',encoding='utf-8')
    return result


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False))
