"""Interface/source guards, not additional optimization benchmarks."""
import json
from pathlib import Path
import pytest
import numpy as np

from chem.benchmarks.official_suzuki_adapter import (encode_point,decode_z,validate_z,CATALYSTS,check_sources)


def test_upstream_files_are_byte_identical():
    assert len(check_sources()['files'])==13


def test_all_category_centers_round_trip_without_onehot():
    for c in CATALYSTS:
        point={'catalyst':c,'t_res':330.,'temperature':70.,'catalyst_loading':1.5}
        z=encode_point(point)
        assert len(z)==4 and decode_z(z)==point
    assert decode_z([0,0,0,0])['catalyst']==CATALYSTS[0]
    assert decode_z([1,1,1,1])['catalyst']==CATALYSTS[-1]
    for k in range(1,8):
        assert decode_z([.5,.5,.5,k/8])['catalyst']==CATALYSTS[k]


@pytest.mark.parametrize('z',[[0,0,0,-.001],[0,0,0,1.01],[0,0,float('nan'),0],[0,0,0]])
def test_invalid_oracle_inputs_fail(z):
    with pytest.raises(ValueError):validate_z(z)


def test_resume_and_native_acquisition_qualification():
    q=json.loads(Path('research/suzuki_official/qualification.json').read_text(encoding='utf8'))
    assert q['LLINBO_direct_upstream_vs_paused_memoized_query_equality']
    assert q['LLINBO_final_rng_equality']
    assert q['LGBO_original_qLogEI_path']['acquisition_calls'][0]['mc_sample_shape']==[512]
    assert q['LGBO_original_qLogEI_path']['cand_size']==8192
    assert q['LGBO_original_qLogEI_path']['plan']['delta'] != .7


def test_native_small_region_point_fallback_is_retained():
    import torch
    from threadpoolctl import threadpool_limits
    from chem.benchmarks.official_suzuki_adapter import execute_lgbo
    s=json.loads(Path('data/runs/suzuki_official/runs/c1_s3_LGBO/state.json').read_text(encoding='utf8'))
    text='Thinking: small-region interface regression fixture only.\nFinal Answer:\n[region, [[350,100,2.0,0.88],[351,101,2.01,0.89]],0.7]'
    torch.set_num_threads(1)
    with threadpool_limits(1):z,details=execute_lgbo(s['history'][:4],text,1773)
    assert details['plan']['mode']=='point'
    assert details['acquisition_calls']==[]
    decode_z(z)
