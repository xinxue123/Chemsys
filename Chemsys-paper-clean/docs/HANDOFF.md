# Research handoff

## Current question and method

The paper separates the benefit of an LLM-selected starting point from the benefit of subsequent guidance. Each scope-controller hypothesis contains a claim, a candidate region, a forecast recorded before measurement and a failure criterion. Feedback updates knowledge permission and can trigger refresh. The controller does not rewrite GP training data, posterior predictions or acquisition values.

The evidence is task- and model-dependent. Development gains largely came from starting-point choice; later guidance added little in the primary shared-start comparisons, while the additional DeepSeek CPA check showed a positive increment. These results do not establish a uniformly stronger optimizer.

## Current source files

| Question | File |
|---|---|
| Main study layer | `code/experiment_tools/hypothesis_scope_v2/` |
| Observation commits and feedback | `code/experiment_tools/hypothesis_scope_loop/study.py` |
| Knowledge scoring and permissions | `code/experiment_tools/hypothesis_scope_loop/knowledge.py` |
| Scope-dependent qLogEI choice | `code/experiment_tools/hypothesis_scope_loop/policy.py` |
| Shared starts | `code/experiment_tools/hypothesis_scope_v2/shared_start.py`, `cpa.py` |
| Comparator decisions | `code/experiment_tools/grounded_baselines/policy.py`, `prompts.py` |
| Additional-model transport | `code/extensions/deepseek/` |

The selected import graph retains historical module names to keep numerical and decision bodies unchanged. Do not substitute an earlier V13 experiment merely because one dependency has V13 in its name.

## What a fresh clone can verify

`python research.py verify` checks the 13 pool/oracle datasets, source hashes, derived development curves and task means, all 288 rows in the primary/DeepSeek paired comparison, 99 distinct coded hypotheses, 68 contrasts and the 11 displayed Table 6 rows. These are arithmetic and file-integrity checks, not a replay of private decisions or a causal test of a module.

New optimization is opt-in. `tools/run_bo.py` is a convenience I/O wrapper around the original GP/tie classes; it is not a new optimizer and did not generate a replacement baseline during synchronization. New output belongs in `output/`.

## What remains local

Original model responses, state/commit/feedback records, per-round GP arrays, numerical fit caches, full evidence archives and runtime installations remain in `Chemsys-paper-clean` and the original research workspace outside this GitHub projection. See [restoration](../LOCAL_DATA_REQUIREMENTS.md).

The full local package's verification and the GitHub edition's verification have different scopes. Historical same-machine development reference runs and standalone registration Markdown documents absent from the supplied archive are not fabricated. The frozen manuscript summaries and available design/provenance records are retained; [coverage](../provenance/COVERAGE.md) records the limits.
