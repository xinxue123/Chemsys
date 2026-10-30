"""Execution-only nullable-measurement adapter for frozen M2 event checks.

No KF update exists when B4 retains no regional clause. The frozen recorder
stores measurement=None, whereas event_reasons expected an absent key/dict.
Interpret None as absent evidence (z default0), exactly the intended contract.
All GP/acquisition/knowledge/state data and source files remain unchanged.
"""
from . import reasoning_bo_engine as engine
from . import reasoning_bo_study as study

ORIGINAL_EVENT_REASONS=engine.event_reasons


def event_reasons(state,decision,iteration):
    view={**state,'decisions':[{**d,'measurement':d.get('measurement') or {}}
        for d in state['decisions']]}
    return ORIGINAL_EVENT_REASONS(view,decision,iteration)


def run_one(spec,config_path='../config.txt'):
    engine.event_reasons=event_reasons
    return study.run_one(spec,config_path)
