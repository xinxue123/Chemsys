# Research handoff: the incremental utility of later knowledge decisions

> GitHub compact edition: code, summaries and derived curves are published here. Raw evidence, detailed inventories, checkpoints and large archives remain in the separate local bundle. See [local evidence requirements](../LOCAL_DATA_REQUIREMENTS.md). Descriptions of full-bundle audits refer to the recorded local validation.

Snapshot date: 5 November 2026. This handoff adds no experiments, GP fits or LLM calls.

## Working claim and evidence boundary

**An informative LLM starting experiment can improve search, while the additional value of subsequent knowledge decisions remains inconsistent.** Experimental feedback can update a knowledge state correctly without changing the next action, and an action change can help one trajectory and harm another. Evaluate these three links separately: feedback to state, state to selection, and selection to search value.

This is a finding about the supplied finite-pool studies and their registered controls. The archive contains useful gains, null effects and losses. It supports a conditional, heterogeneous-utility account; a universal claim about LLM knowledge or KF effectiveness would exceed the evidence.

The current hypothesis-constrained controller and historical V13 are different versions. Their evidence is identified separately throughout this bundle. Code correctness, state-transition replay and unchanged-source checks establish technical execution; they do not establish optimization benefit.

## Read these files first

1. [Evidence overview](../results/README.md): principal comparisons and the current results.
2. [Metrics and cohorts](METRICS_AND_COHORTS.md): 4+16, 0+20, inherited prefixes, backend differences and reused references.
3. [Data and code map](DATA_AND_CODE_MAP.md): where every type of evidence is stored.
4. [Reproduction guide](REPRODUCIBILITY.md): offline checks and independent workspace materialization.
5. [Primary record index](../results/combined_primary_metrics.csv): 487 comparison records with original state paths and hashes.
6. [Full archive catalogue](../data_catalogue/archive_catalogue.csv): all 63 run groups and the additional source, dataset, report, packet, manuscript and audit archives.

## Evidence supporting the research question

| Comparison | Scope | Observed SearchAUC result | Interpretation |
|---|---|---|---|
| Historical V13 vs BO, 4+16 | Eight tasks, seeds 0–3, 32 pairs | 17 wins, 0 ties, 15 losses | Shared initial observations do not yield a consistent V13 advantage. |
| Historical V13 vs Reasoning BO, 4+16 | The same 32 pairs | 11 wins, 7 ties, 14 losses | The Reasoning BO backend differs between seed 0 and seeds 1–3. |
| Historical V13 vs BO, 0+20 | Eight tasks, seeds 0–3, 32 pairs | 22 wins, 1 tie, 9 losses | Whole-trajectory gains include initialization and subsequent LLM decisions. |
| Historical V13 vs Reasoning BO, 0+20 | The same 32 pairs | 13 wins, 5 ties, 14 losses | A broad V13 advantage over this comparator is absent. |
| V13 shared-prefix continuations | Six fixed ten-observation prefixes, two LLM realizations per non-BO arm | Full V13 has no positive cell-mean continuation-AUC increment over fixed KF or disabled future refresh | The arm-specific LLM responses also differ; this is not a response-controlled numerical KF contrast. |
| Programmatic KF pilot | Perera seed 2, HPLC seed 3 | Dynamic minus fixed AUC: 0.00 and +33.45 native units | Same forecasts at a shared input context isolate a programmed ranking effect. |
| Programmatic KF confirmation | Perera seed 1, HPLC seed 1, Direct Arylation seed 0 | Relative AUC differences: 0.00%, +2.21%, −0.37% | A localized HPLC gain coexists with null and adverse transfer results; the registered cross-task confirmation gate was unmet. |
| Current grounded hypothesis controller vs BO | Eight tasks, four exact-tie streams, 32 pairs | 23 wins, 0 ties, 9 losses; higher task-mean AUC in 6/8 tasks | Fixed LLM first input and initial knowledge are shared across streams. |
| Current controller vs same-first BO | The same 32 pairs | 8 wins, 16 ties, 8 losses | The later knowledge policy has no consistent paired advantage in this sample. |

The [current complete comparison](../results/grounded_baselines_20261105/comparison_tables.txt) also includes LGBO and Reasoning BO. Their task-mean AUC exceeds random-start BO in 7/8 and 5/8 tasks, respectively. These are matched-interface adaptations with documented differences from the published protocols. Four streams and fixed initial knowledge do not establish broad statistical superiority.

The confirmation retains its positive HPLC effect. Direct Arylation loses 1.81 yield percentage points in final best under dynamic KF relative to fixed weight. The latest controller's Doyle mean AUC is 80.92 versus BO 78.15, while its final best is 85.68 versus BO 92.21. Search efficiency and terminal quality therefore require separate reporting.

## Which controller is which?

**Historical mean-correction/B 4 development.** Early Suzuki and P 1/P 2 studies modify a knowledge contribution to a GP-based decision, including regional tilts, KF, B 4 and event-refresh variants. Their four artificial/assistant knowledge conditions and differing early GP recipes remain separate development cohorts. They are all archived, including rejected hypotheses and failed variants.

**V13.** Predictive reliability and permission to recommend are separate states. The scalar KF scores prospective improvement forecasts relative to the current GP reference; permissions can pause or withdraw advice. Event-driven editing changes structured Notes. The original Reasoning BO style selection interface and subsequent programmatic-weight variants are documented in the original protocol and code. A state change alone is insufficient evidence of improved ranking.

**Current grounded hypothesis controller (`hypothesis_scope_loop` / `hypothesis_scope_multiseed`).** A chemically grounded, independent LLM first proposal is frozen. Blinded favour/avoid hypotheses are attached to that proposal. Later unchanged native qLogEI is maximized inside eligible favour scopes after removing eligible avoid scopes. This version does **not** use a global qLogEI-gap gate or modify GP predictions. Empty or ineligible guidance falls back to unrestricted BO. Prospective forecasts are committed before feedback; GP-relative Brier evidence updates KF credibility. Repeated adverse skill, forecast misses or genuinely constraint-caused nonimprovements can pause guidance. Semantic rules inherit adverse evidence across revisions. Explicit retests require an eligible event; accurate failure forecasts alone do not restore positive recommendation permission. Initial plus at most three refreshes use a three-query minimum gap and a reserved late call. See [the original scope protocol](original_protocols/hypothesis_scope_loop_20261105.txt).

**LGBO comparator.** Official regional posterior-mean tilting, unchanged posterior covariance, full-pool native qLogEI, finite mixed-domain quadrature cap 512. No KF or event permission lifecycle is inserted into this comparator.

**Reasoning BO comparator.** Official English Insight template and stable high/medium/low confidence extraction, adapted to a sequential GP-top-five-to-one interface. Shared blinded priors and own Insight history are supplied; the full published retrieval/KG/multi-agent system is not reconstructed.

## The latest LLM backend transition

The user explicitly continued the paused study with `gpt-6.1-terra`, medium, using the Responses API. The 552 previous independent-subagent responses were preserved;664 later responses used stateless API calls. No previous successful response was rerolled. The [per-round backend file](../results/grounded_baselines_20261105/LLM_backend_by_round.csv) gives the exact boundary. Seed 0 is subagent-only after the shared first point; seed 1 contains the transition; seeds 2–3 use API responses after the shared first point.

Four initial LGBO API requests failed before generation because JSON-object mode required the literal `json` in the unchanged input. A recorded transport correction requests that mode only when compatible with the original input. Rejection records and both transport registrations are retained. Scientific prompts, GP, candidate pools, first points, query budgets and successful replies remain unchanged.

This is an explicit mixed-backend continuation. Backend equivalence is unestablished, and the earlier fixed first proposal/knowledge were not independently resampled. Report these limits beside any aggregate claim.

## How to continue the research

First reproduce the existing outcome tables and inspect the first selection divergence against the same-first control. Use the locked prefix, GP arrays, candidate mask, knowledge version, forecast and updated weight from that exact round. Distinguish an inactive rule, a ranking change that preserves the winner, and an actual different experiment.

For a future causal feedback test, freeze candidate-level LLM forecasts and the experimental prefix before changing only the programmed weight or permission. Keep the same-first BO control and compare full trajectories after the intervention as well as immediate outcomes. The previous programmatic KF studies provide both a working implementation and mixed results to preserve.

Preserve experimental cohorts and preregister a new follow-up rather than editing the sealed records. Do not select seeds, tasks, LLM realizations or stopping rules after seeing which ones make the feedback mechanism look successful. Keep positive and negative evidence in the same table. New experimental work is outside this packaging task; the shipped verification commands issue no model calls.

## Custody and transfer

All 399,703 exported archive members were verified against their recorded SHA-256 and ZIP CRC. The 72 archives occupy about 15.89 decimal GB. File-level indices preserve original/exported hashes, source paths and exclusions. The 487 primary rows point to 464 distinct state-file artifacts; this artifact count is not a count of independent experiments. Reused references and separately saved clones require cohort-aware accounting.

Original source code and protocols are preserved inside their archives. Three pre-existing portable code adaptations are retained separately and listed in [portable_code_differences.csv](../data_catalogue/portable_code_differences.csv). Materialize the exact native source before invoking native audit scripts. Frozen manuscripts are historical snapshots; the current claim is explained by this handoff and the updated evidence overview.

All new navigation, metric definitions, summaries and handoff documents are in English. Original scientific prompts, logs, category labels and historical reports retain their original language and bytes for reproducibility. Credentials, environments, Git/dependency caches and unrelated materials/XRD caches are outside the bundle. The archive exclusion indices document transient files and alias handling.
