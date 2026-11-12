# Research handoff

## Current scientific question

The paper separates the value of an LLM-selected starting point from the value of subsequent guidance. The scope controller makes each knowledge item explicit: a claim, a candidate region, forecasts fixed before measurement and a failure criterion. Experimental feedback changes permission to guide and can trigger a later knowledge update. The GP training data, posterior predictions and acquisition values are not rewritten by the scope controller.

The evidence supports useful starting points on the development tasks and task/model-dependent later guidance. It does not establish a uniformly stronger optimizer. The positive additional DeepSeek CPA result and the weaker Direct Arylation result remain separate outcomes.

## Which code is current

`code/experiment_tools/hypothesis_scope_v2/` is the paper's current study layer. It imports the frozen `hypothesis_scope_loop`, `hypothesis_scope_multiseed`, `grounded_first_start`, `grounded_baselines`, GP and calibration modules. Names such as `cold_start_v13` refer to retained dependencies; they do not mean that the clean project is centered on V13.

Useful files:

| Question | Files |
|---|---|
| How are observations committed and feedback applied? | `code/experiment_tools/hypothesis_scope_loop/study.py` |
| How are hypotheses scored and paused? | `code/experiment_tools/hypothesis_scope_loop/knowledge.py` |
| How do scopes change a qLogEI choice? | `code/experiment_tools/hypothesis_scope_loop/policy.py` |
| How are primary-model requests recorded? | `code/experiment_tools/hypothesis_scope_v2/transport.py` |
| How are common starts enforced? | `code/experiment_tools/hypothesis_scope_v2/shared_start.py`, `cpa.py` |
| How do comparator choices differ? | `code/experiment_tools/grounded_baselines/policy.py`, `prompts.py` |
| What changed for the additional model? | `code/extensions/deepseek/`, `provenance/deepseek/` |
| How can one recorded decision be inspected? | `python research.py inspect --task c1 --seed 0 --query 4` |

## Frozen evidence and new work

Treat `data/`, `experiments/`, `evidence/`, the two archive subsets and the frozen paper figures as read-only evidence. New computations belong in `output/` or another explicitly named new study directory. Rebuilding saved analyses writes only derived tables and logs. Do not treat a numerical replay or a successful import as evidence of optimization improvement.

The primary-model cohorts and the additional DeepSeek cohort retain their original model, request and source metadata. Responses are not relabeled or regenerated. The original claim coding and its hash are retained.

## Transfer checklist already performed

All 13 public pools load through the packaged input adapter. Pool/oracle hashes and ID order were checked. Development figure values match rebuilt results to the precision of the frozen CSV exports. The held-out model table contains 288 pair-level rows. The knowledge audit contains 99 distinct hypotheses, 123 hypothesis-by-stream rows and 68 comparative contrasts. Rebuilt analyses issue no new model calls, GP fits or experimental queries.

The complete trajectory index contains multiple study arms, repeated first points and reference states. Its row count is an index of saved state files, not a count of independent experimental comparisons. Six incomplete states are listed separately and excluded from that complete-state index.

`MANIFEST.json` and `VALIDATION.json` describe the delivered snapshot. Read `provenance/COVERAGE.md` before attempting historical whole-study replay.
