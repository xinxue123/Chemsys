from copy import deepcopy
from chem.benchmarks import reasoning_bo_compat as compat


def test_null_is_absent_innovation_without_mutating_history():
    state={'method':'New_R2','calls':[],'decisions':[
        {'incumbent_gain':.01,'measurement':None},
        {'incumbent_gain':.01,'measurement':None},
        {'incumbent_gain':.01,'measurement':None}]}
    before=deepcopy(state)
    assert compat.event_reasons(state,{'fallback':False},4)==[]
    assert compat.event_reasons(state,{'fallback':True},4)==['B4_conflict']
    assert state==before


def test_present_measurement_preserves_original_results():
    for z in (0.,2.,3.):
        state={'method':'New_R2','calls':[],'decisions':[
            {'incumbent_gain':0.,'measurement':{'innovation_z':z}}]*3}
        d={'fallback':False}
        assert compat.event_reasons(state,d,4)==compat.ORIGINAL_EVENT_REASONS(state,d,4)
