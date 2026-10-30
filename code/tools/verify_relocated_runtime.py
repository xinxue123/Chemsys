"""Verify archived P0/P1 checks and R5 replay without rewriting old receipts."""
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys

WORKSPACE=Path(__file__).resolve().parents[2]
LEGACY=WORKSPACE/'chemist_system'
DEST=WORKSPACE/'experiments/verification/layout_20261030'


def main():
    sys.path.insert(0,str(LEGACY));os.chdir(LEGACY)
    from chem.benchmarks import stable_suzuki_study as p0
    from chem.benchmarks import stable_feedback_study as p1
    from chem.benchmarks import continuous_r5_study as r5
    from chem.benchmarks import continuous_r5_report as replay
    from chem.benchmarks import continuous_benchmark_data as data
    p0.check();p1.check('feedback_t0');r5.check()
    group0=p0.status()
    group1=[{'task':r['task'],'status':r['status'],'queries':len(r['decisions'])} for r in p1.states('feedback_t0')]
    assert len(group1)==2 and all(r['status']=='COMPLETE' and r['queries']==16 for r in group1)
    saved=[]
    with patch.object(data,'write',lambda path,value:saved.append((str(path),value))):
        audit=replay.audit()
    assert audit['status']=='PASS' and len(saved)==1
    assert all(r['max_prediction_acquisition_error']==0 and r['GP_refits_in_audit']==0 for r in audit['checks'])
    result={'status':'PASS','P0_source_protocol_check':'PASS','P0_status':group0,
        'P1_source_protocol_check':'PASS','P1_status':group1,'R5_audit':audit,
        'archived_receipts_written':0,'new_experiment_queries':0,'new_LLM_calls':0}
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'runtime_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
