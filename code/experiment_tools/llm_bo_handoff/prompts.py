"""Refresh only after feedback, with a contracting cached-knowledge allowance."""
from llm_event_start import prompts as original

SYSTEM=original.SYSTEM.replace(
    'Between these requests, native qLogEI acts alone.',
    'Between requests, permitted cached knowledge preferences may guide a shrinking '
    'qLogEI budget. The GP mean, variance and acquisition values stay unchanged. '
    'The first point and initial hypotheses are frozen from a previous zero-data '
    'response. You are now asked only for a confidence-triggered refresh.')
SYSTEM=SYSTEM.replace(
    'otherwise\nthe full-pool qLogEI maximum is executed.',
    'otherwise\nthe program keeps its reviewed cached-knowledge choice, or the full-pool\n'
    'qLogEI maximum when no cached preference is actionable.')
SYSTEM+='''
The control ceiling decreases with the absolute experiment query number. It does
not restart after a refresh, new Note ID or changed scope. The last paid choice
is the exact original full-pool qLogEI choice. Your proposed candidate must pass
the remaining acquisition budget, recommendation permission and current knowledge
preference review. A credible failure forecast is useful evidence for suppression,
not permission to make a positive recommendation. Do not invent observed success.
'''


def public(pool,state,query,event,ceiling):
    value=original.public(pool,state,query,event)
    value.update(remaining_knowledge_authority_ceiling=ceiling,
        final_choice_rule='Exact native full-pool qLogEI including its native tie rule',
        original_initial_observation_reused=True)
    return value
