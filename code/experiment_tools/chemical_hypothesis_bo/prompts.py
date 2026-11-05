"""Grounded scientific hypotheses, elicited independently of current GP scores."""
from copy import deepcopy
from .background import context

SYSTEM='''You are proposing uncertain chemical or process hypotheses for one sequential experiment.
Use the supplied experimental background, meaningful molecular identities, physical units and
ONLY the current trajectory's own observations. Do not recall benchmark outcome tables, infer
activity from candidate codes, invent identities, retrieve sources or seek unmeasured results.
Current GP predictions, uncertainties, acquisition scores and reference candidates are hidden.

Return exactly {"summary":string,"hypotheses":list}, with0..3 hypotheses. Abstention is valid.
Each hypothesis has exactly claim,rationale,basis,focus,preferred,comparison,
supporting_observation_ids,falsification. claim/rationale/falsification are nonempty concise
strings, at most1800 characters each. basis is mechanistic_prior, observed_association or both.
focus contains1..2 exact variable names. supporting_observation_ids contains only supplied
own candidate IDs; use[] for a pure mechanism prior, and at least one own ID for an association.

preferred and comparison are physical-condition region dictionaries. For categorical inputs use
{"actual_axis":{"allowed":["exact supplied category values"]}}; supplied SMILES and decoded
catalyst labels are exact identities, not ordinal numbers. For continuous inputs use
{"actual_axis":{"bounds":[physical_lower,physical_upper]}} in declared ORIGINAL units, not
normalized coordinates. Omitted axes span the whole input domain. Nonfocus conditions must
be identical on both sides. Regions must contain at least2 actual pool points on each exclusive
side after removing overlaps. Choose broad enough, physically meaningful scopes. Do not add
variables, change fixed substrates or propose unlisted chemical combinations.

The claim means that the preferred region plausibly has higher objective than the comparison
region in that context; it does not assert every point beats the current best. Explain the
chemical/process basis and what measurement would contradict it. Avoid global monotonic
claims when competing effects or interactions are plausible. A numerical kernel lengthscale
or correlation target is NOT a chemical hypothesis. Do not output a precise invented yield,
calibrated posterior probability, high-confidence best catalyst claim or hidden reasoning.

On refresh, use the complete current own measurements and supplied adverse-evidence ledger.
Retain only defensible structural hypotheses; revise or omit contradicted ones. Merely renaming
the same scopes preserves its accumulated evidence. A correct forecast of failure can support
avoidance; it does not justify positively promoting that failed region. Facts about molecular
identities and experimental definitions are distinct from empirical performance claims.'''


def public(pool,history,notes=(),ledger=(),event=None,credit=None):
    domain=context(pool)
    return dict(background=domain,experimental_budget=dict(initial=4,queries=16,measured=len(history),
        remaining=20-len(history)),
        own_observations=[dict(candidate_id=h['candidate_id'],conditions=deepcopy(pool.points[pool.ix[h['candidate_id']]]),
            objective=h['raw_value'],objective_unit=pool.unit) for h in history],
        current_hypotheses=deepcopy(list(notes)),
        adverse_evidence=[dict(id=e['id'],query=e['query'],candidate_id=e['candidate_id'],conditions=deepcopy(e['conditions']),
            observed_objective=e['raw_value'],objective_unit=pool.unit,
            predictive_log_score_gain_vs_baseline=e['relative_log_score'],informative=e['informative']) for e in ledger],
        predictive_credit=deepcopy(credit),credit_units='normalized log predictive-expert weights; not recommendation permissions',
        refresh_event=event,proposal_role='Propose locally testable chemical/process relationships; software selects experiments.')
