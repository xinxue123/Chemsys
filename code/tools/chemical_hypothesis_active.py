"""Select the necessary metadata-corrected Doyle run without altering archives."""
from pathlib import Path
from chemical_hypothesis_bo import study as s,background
PARENT=s.ROOT/'experiments/runs/chemical_hypothesis_bo_20261105'
FIX=s.ROOT/'experiments/runs/chemical_hypothesis_bo_doyle_identityfix_20261105'
FIX_REPORT=s.ROOT/'experiments/reports/chemical_hypothesis_bo_doyle_identityfix_20261105'
TASK='Doyle_Buchwald_Hartwig'


def folder_for(task,arm):return (FIX if task==TASK else PARENT)/'runs'/s.SHORT[task]/arm


def cache_for(task):return (FIX if task==TASK else PARENT)/'cache'/s.SHORT[task]


def responses():
    return [p for p in (PARENT/'requests').glob('*/*/validated.json') if p.parent.parent.name!='doyle']+list((FIX/'requests').glob('*/*/validated.json'))


def activate(record=False):
    assert s.read(FIX_REPORT/'audit.json')['status']=='PASS_CORRECTED_DOYLE_REPLAY'
    s.folder_for=folder_for
    background.COMMON_NAMES['CN1CCCN2C1=NCCC2']='7-methyl-1,5,7-triazabicyclo[4.4.0]dec-5-ene (MTBD)'
    value=dict(Doyle_active_run=str(FIX.relative_to(s.ROOT)),corrected_audit_sha256=s.sha(FIX_REPORT/'audit.json'),
        corrected_protocol_sha256=s.sha(FIX/'protocol.json'),other_seven_tasks='unchanged parent runs',
        superseded_Doyle='Archived parent Doyle fixed/adaptive runs had an incorrect DBU annotation for MTBD.',
        active_decision_count=256,active_logical_LLM_calls=32,total_actual_LLM_calls_including_superseded=36,
        limitation='The corrective rerun also has fresh LLM responses; its outcome change is not an isolated estimate of the name correction.')
    if record:s.write(s.REPORT/'active_comparison.json',value,replace=True)
    return value
