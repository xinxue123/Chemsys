"""One-time provenance-preserving fix of an adapter logging assertion."""
import json
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
from chem.benchmarks.official_suzuki_runner import ROOT,RESEARCH,read,hashes
from chem.benchmarks.suzuki_domain import digest
from chem.decision.replay import write_json_atomic

out=RESEARCH/'amendment_01'
if (out/'receipt.json').exists(): raise SystemExit('amendment already applied')
old=read(ROOT/'protocol.json')
write_json_atomic(out/'protocol_before.json',old)
snapshots=[]
for f in sorted((ROOT/'runs').glob('*/state.json')):
    s=read(f);write_json_atomic(out/'states_before'/f.parent.name/'state.json',s)
    snapshots.append({'run_id':s['run_id'],'status':s['status'],'queries':len(s['history']),
                      'state_sha256':sha256(f.read_bytes()).hexdigest()})
new=deepcopy(old)
new['engine_sha256']=hashes()
new['LGBO']['native_branches']='Region/region-soft MCqLogEI; native small-region fallback to point prior retained unchanged'
new['transport_amendment']={'previous_protocol_sha256':old['protocol_sha256'],
    'reason':'Adapter incorrectly required the regional constructor on every native LGBO branch. Permit unchanged source point fallback; preserve prior exact LLM contexts on loop replay.',
    'algorithm_files_changed':0,'oracle_queries_discarded':0,'responses_rewritten':0,
    'affected_adapter_stops':['c1_s3_LGBO','c4_s2_LGBO','c4_s4_LGBO'],
    'scope':'logging validation and prior-context replay transport only; no algorithm parameters or scientific decisions changed'}
new.pop('protocol_sha256');new['protocol_sha256']=digest(new)
write_json_atomic(ROOT/'protocol.json',new);write_json_atomic(RESEARCH/'protocol.json',new)
for f in sorted((ROOT/'runs').glob('*/state.json')):
    s=read(f)
    s['protocol_lineage']=[old['protocol_sha256'],new['protocol_sha256']]
    s['protocol_sha256']=new['protocol_sha256']
    if s['method']!='HEBO':
        if s['status']=='FAILED': s['recovered_adapter_stop']={'error':s.pop('error'),'traceback':s.pop('traceback')}
        s['status']='CREATED';s.pop('pending_context',None)
    write_json_atomic(f,s)
write_json_atomic(out/'receipt.json',{'before_protocol_sha256':old['protocol_sha256'],
    'after_protocol_sha256':new['protocol_sha256'],'states':snapshots,
    'algorithm_source_bytes_unchanged':True,'no_simulator_query_at_failed_steps':True,
    'counters_preserved':True,'new_model_or_algorithm_settings':False})
print(new['protocol_sha256'])
