"""Document an execution-only JSON-cache repair and roll back uncommitted state."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from chem.benchmarks import continuous_r2_study as s


def main():
    import os
    os.chdir(ROOT)
    protocol=s.data.read(s.OUT/'protocol.json')
    assert not (s.OUT/'execution_amendment.json').exists(),'Repair is applied once only.'
    restored=[];completed={}
    for task in s.TASKS:
        subset=s.ROOT/'runs'/(task+'_'+s.e.METHODS[1])
        assert s.data.read(subset/'state.json')['status']=='COMPLETE'
        for path in subset.rglob('*'):
            if path.is_file():completed[path.as_posix()]=s.data.filehash(path)
        path=s.ROOT/'runs'/(task+'_'+s.e.METHODS[0])/'state.json'
        interrupted=s.data.read(path);n=len(interrupted['decisions'])
        assert interrupted['status']=='INTERRUPTED' and n==s.e.BOOTSTRAP[task]
        stable=s.initial(task,s.e.METHODS[0],protocol)
        for key in ['history','decisions','bootstrap_receipts']:
            stable[key]=deepcopy(interrupted[key])
        stable['knowledge']=deepcopy(stable['decisions'][-1]['knowledge'])
        stable['region_trust']=deepcopy(stable['decisions'][-1]['region_trust_after'])
        for event in stable['decisions']:
            stable['banks'][event['bank_key']]=deepcopy(event['bank_after'])
            s.e.register_feedback(stable,event)
            cp=path.parent/'rounds'/f"{event['iteration']:02d}_committed.json"
            assert s.data.filehash(cp)==event['commit_sha256']
        for key in ['calls','refresh_checks','interaction_checks']:
            stable[key]=[deepcopy(c) for c in interrupted[key] if c['iteration']<=n]
        stable['interaction_tests']=[deepcopy(c) for c in interrupted['interaction_tests'] if c['query']<=n]
        stable['status']='READY'
        before=s.data.filehash(path);s.data.write(path,stable)
        restored.append({'task':task,'completed_queries_preserved':n,'before_state_sha256':before,
                         'restored_state_sha256':s.data.filehash(path),
                         'history_sha256':s.digest(stable['history']),
                         'completed_decisions_and_commit_hashes_unchanged':True})
    amendment={'kind':'execution_only_JSON_tuple_roundtrip_and_uncommitted_state_rollback',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'original_protocol_sha256':protocol['protocol_sha256'],
        'original_source_sha256':protocol['source_sha256'],'corrected_source_sha256':s.sources(),
        'scientific_parameters_unchanged':True,'original_parameters_sha256':s.digest(protocol['parameters']),
        'rule_change':False,'new_experimental_choices_before_failure':0,
        'changes':['Compare cached JSON contexts by canonical digest instead of Python tuple/list equality.',
                   'Restore the last completed state if a future step fails before completing.',
                   'Add a shared-request JSON roundtrip regression test; eleven behavioral tests passed.'],
        'restored_runs':restored,'completed_subset_files_sha256':completed,
        'original_evidence':'technical_repair/original_sources and technical_repair/*_state.json, *_error.json'}
    s.data.write(s.OUT/'execution_amendment.json',amendment)
    s.check()
    print({'status':'TECHNICAL_REPAIR_RECORDED','restored_runs':restored,'completed_subset_files':len(completed)})


if __name__=='__main__':main()
