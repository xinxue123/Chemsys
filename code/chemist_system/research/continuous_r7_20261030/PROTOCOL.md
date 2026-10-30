# R7-D0 / R7-S — covariance-reachable qualitative BO validation

**Status:** implementation + deterministic mechanism tests; no real SnAr/HPLC
performance result is claimed until the archived private input pools, oracles,
and frozen GP stack are available on the original research machine.

## Frozen research boundary

- Independent research branch off R6 commit `de1f0bf1c76ef2588e0df068e82bae5b2717f6c9`.
- Main `cf5c01b32af8e8417b66e4aa9b314c8acb55ef5a` and PR #1 unchanged.
- No modifications to the original GP, MC512 qLogEI, candidate pools, R6 prompt,
  R6 code, R5 data, scoring units, time-varying acquisition gate or tie policy.
- Same input-only 4 initial points per task and seed; each arm gets 16 additional
  queries. SnAr target is STY; HPLC target is detector peak area, not yield.
- All actual oracle access is AFTER saving a complete prequery commitment.

## R7-D0: reachability only

`continuous_r7_diagnostics` independently rebuilds the GP at each R6 A4
prequery prefix (in a NEW R7 diagnostic cache). It computes for every accepted
hypothesis at every round:

1. Geometric R6 scope size and its intersection with original qLogEI gate.
2. GP-covariance-relevant candidates under the same original gate.
3. Which candidate would win if knowledge had its actual prequery weight.
4. Which candidate would win at the PREDECLARED R6 cap, without fitting a
   weight to an outcome. These are action-reachability counterfactuals, **not**
   SearchAUC causal estimates.
5. Optional after-the-fact evaluation of **prequery** conditional log-score
   against the actually selected outcome. It never feeds that outcome back
   into a candidate choice or prior score.

The diagnostic never reads an `oracle.npz` and never calls an LLM. Its GP
reconstruction may refit to its own prequery history; R6 caches are untouched.

## R7-S: minimal intervention

Use R6's single *anchored* preference event
`Z=f(better)-f(worse)+N(0,tau^2)>0`. R7-S replaces the R6 geometric output
scope with the original GP posterior cross-covariance `Cov(f(x),Z)`. This is
posterior belief propagation, **not** a claim of global chemical monotonicity.
The existing one-sign skew-normal conditional q=1 EI quadrature, R6's logEI
ratio cap (`±log(4)`), evidence ledger and max deployment weight 0.5 are
retained. The original qLogEI gate is always enforced.

Passive shadow scores are committed before the query and recorded for every
existing hypothesis whose numerical squared correlation is above `1e-10`.
No separate active probe, forced verification or gate exception is permitted.
The correlated-sign information diagnostic uses
`I(Y;1[Z>0])`, not the R6 Gaussian information proxy `I(Y;Z)`.

| Arm | Mechanism | LLM | Verification |
| --- | --- | --- | --- |
| B0 | exact original Pure BO | none | none |
| B2 | original GP + passive covariance-conditioned preference | one first-round comparison | none |
| B4 | B2 with the direction of the **same** first-round comparison reversed | exact B2 first-round request/response is reused | none |
| B2_online | same covariance policy, but adaptive later hypotheses | original R6 rounds 1, 3, 6, 9 | none |

One first-round comparison, shared by B2/B4 for the same task+seed, isolates
the propagation mechanism from stochastic later-round LLM generations. If the
LLM abstains, B2/B4 must reduce to Pure BO. `B2_online` is the separately
identified **adaptive/full-stack** test: later input prefixes may diverge,
so its LLM responses cannot be causally attributed to the same fixed knowledge
tape as B2/B4. A separate R7-V with cost-aware active verification is
**not implemented** in this stage.

## Original research environment

From `code/chemist_system` on the research machine, with the original
`data/benchmarks/continuous_20261029/{SnAr,HPLC}` input pools/oracles and
original GP/DeepSeek dependencies:

```bash
python -m unittest discover -s tests -p 'test_continuous_r*.py' -v
python continuous_r7.py qualify
python continuous_r7.py diagnose --seed 0 --r6-root <ABSOLUTE_R6_RUN_ROOT>
python continuous_r7.py prepare
python continuous_r7.py run --task SnAr --arm B0 --seed 1
python continuous_r7.py run --task SnAr --arm B2 --seed 1 --config ../config.txt
python continuous_r7.py run --task SnAr --arm B4 --seed 1 --config ../config.txt
python continuous_r7.py run --task SnAr --arm B2_online --seed 1 --config ../config.txt
python continuous_r7.py audit
python continuous_r7.py status
python continuous_r7.py report
```

Repeat for HPLC and development seeds 1–5. Seed 0 is for archived R6
diagnostics only. Lock the R7 source and predeclared protocol before touching
confirmation seeds 6–15. Never cherry-pick an arm from seed0.

`diagnose` writes only under new `research/continuous_r7_20261030/` and
`data/runs/continuous_r7_20261030/diagnostic_cache` paths. Full R7-S runs
write independent commitments and observations under
`data/runs/continuous_r7_20261030/runs/`.

## Go/no-go reporting

Report separately per task, for matched seeds:

- Reachable hypotheses and actual/capped potential rank flips;
- Real `knowledge_guided` decisions, not just positive weights;
- Precommitted score increments, reversals and abstention rates;
- Paired SearchAUC (mean of 16 post-initialization incumbents), final best,
  first-best total round, LLM and experiment counts;
- B2 versus both B0 and B4, without aggregating incompatible task units.

NO-GO for performance promotion if the real R7-S rollout never makes an
incremental knowledge-guided decision, or if its paired SearchAUC does not
improve on confirmation data. Passing unit tests cannot substitute for a
private-data closed-loop execution.
