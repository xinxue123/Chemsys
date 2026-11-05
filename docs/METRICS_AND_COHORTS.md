# Metrics, budgets and comparison cohorts

> GitHub compact edition: code, summaries and derived curves are published here. Raw evidence, detailed inventories, checkpoints and large archives remain in the separate local bundle. See [local evidence requirements](../LOCAL_DATA_REQUIREMENTS.md). Descriptions of full-bundle audits refer to the recorded local validation.

## Definitions

For zero-start experiments with observed objectives `y_1,...,y_N`, let `M_t=max(y_1,...,y_t)`. SearchAUC is `sum(M_t for t=1..N)/N`. Every executed query contributes, including zeros and nonimprovements. Final best is `M_N`. A final-best attainment index identifies the first query at which that trajectory's terminal best was observed; it does not imply attainment of the pool's global maximum.

For the complete 0+20 cohorts here, `N=20`. For complete 4+16 cohorts, let `b_0` be the best of the four initial observations and `M_t=max(b_0,y_1,...,y_t)` for the 16 subsequent queries. Their reported SearchAUC is the mean of those 16 post-initialization maxima. The initial four establish the incumbent but are not four additional terms in that metric. Total historical experiment cost is 20 observations. The warm first-attainment index is expressed in post-initialization query units; zero can denote an initial best already attained.

For a shared prefix of ten observations, the primary continuation SearchAUC averages `M_11,...,M_20`, including the inherited incumbent. `SearchAUC20` also includes the ten inherited prefix maxima. A continuation spends ten new queries, not twenty new queries. The archived table retains both metrics.

The Suzuki 1+19 study retains both SearchAUC 20 (including the first point) and SearchAUC 19 (averaging subsequent 19 maxima with the first point inherited). Eight-query pilots use their own explicit definitions. Keep these fields and cohorts separate.

Incomplete trajectories remain in the raw catalogue with their status and observed count. Their prefix means are not full-budget SearchAUC values. Divide by the number of valid terms only for an explicitly named prefix metric; do not let an unfinished run masquerade as a completed budget.

The optional `ln_sum_cumulative_best` in the latest tables is exactly `ln(20*SearchAUC)` per complete trajectory. This defined diagnostic does not establish equivalence to any publication's Log AUC. A mean of logs cannot be exponentiated into an arithmetic mean SearchAUC.

## Primary exported cohorts

| Cohort | Records | Budget / source | Essential qualification |
|---|---:|---|---|
| Historical warm | 131 |4+16; [metrics](../results/cohorts/warm_4plus16/metrics.csv), [rounds (local evidence)](../LOCAL_DATA_REQUIREMENTS.md) |32 each V13/BO/RBO/V12-subagent plus 3 selected historical API V12 records. Eight tasks, seeds 0–3. RBO seed 0 uses historical API, seeds 1–3 subagents. |
| Historical V13 cold |96 |0+20; [metrics](../results/cohorts/v13_cold_0plus20/metrics.csv), [rounds (local evidence)](../LOCAL_DATA_REQUIREMENTS.md) |32 each V13/BO/RBO. First selection is made from an input-only five-point menu; it is not the later full-pool grounded-first design. |
| V13 continuations / transfer |75 |66 prefix continuations plus 9 standalone Direct Arylation traces; [data](../results/cohorts/v13_late_and_transfer/final_metrics.csv) |Six prefixes, eleven continuations per prefix. Full/fixed-KF/no-refresh/RBO/coupled have two LLM realizations; BO has one. Responses vary by arm. |
| Programmatic KF pilot |10 |Four new 0+20 traces plus six frozen references; [metrics](../results/kf_weight/metrics.csv) |Perera seed 2 and HPLC seed 3. A controlled ranking-weight contrast, with frozen LLM candidate forecasts at a shared context. |
| Programmatic KF confirmation |15 |Six new 0+20 traces plus nine frozen references; [metrics](../results/kf_confirmation/metrics.csv) |Perera seed 1, HPLC seed 1, Direct Arylation seed 0;27 additional locked diagnostic prefixes, with offline alternative-label lookups accounted separately. |
| Current grounded comparison |160 |Eight tasks x four streams x five methods; [metrics](../results/grounded_baselines_20261105/per_seed_results.csv) |64 new LGBO/RBO traces plus 96 frozen BO/same-first BO/current-hypothesis references. Fixed initial knowledge and LLM first point. GP numerical seed 0, exact-tie streams 0–3. |

The [combined primary index](../results/combined_primary_metrics.csv) contains 487 comparison records. Its state links and hashes identify source artifacts. These rows include reused references; summing the rows or budgets does not count unique new experiments. The full raw archive covers additional earlier development, artificial-knowledge stress, fitting/tie audits and short-budget pilots.

The index explicitly records objective units, response repetition where applicable, inherited observations, reported query count, continuation AUC, and reference/new-trajectory roles. `new_paid_queries_in_this_comparison` identifies additional costs in the later continuation, programmed-KF and latest-baseline cohorts; it is blank for the historical census. A blank historical cost is not zero. The three later additions sum to 2320 new paid queries, while reused references contribute zero to those additions.

## Units and datasets

| Task | Frozen pool size | Reported objective |
|---|---:|---|
| Reizman Suzuki Cases 1–4 |4692 each |Simulated yield (%) from a historical emulator |
| Perera Suzuki |3696 |UV chromatographic product area (%) |
| Doyle Buchwald–Hartwig |3955 |Reported experimental reaction yield (%) |
| Summit SnAr |4096 |Space-time yield, kg m^-3 h^-1 |
| HPLC |1007 |Peak area, source arbitrary units |

Direct Arylation is an additional transfer task. Its exact input/oracle table and provenance are in the `v13_late_validation_20261103` run archive under `holdout/`; it is also used as an already available task for the later KF confirmation. This is not an untouched holdout for every version of the research program.

No average of raw objectives across these task units is reported. Perera's historical `yield` shorthand does not convert detector product area into isolated chemical yield. Perera/Doyle candidate tables vary substrate identities as well as conditions; interpret them as the archived finite-table screening tasks. All optimization results are offline simulations/table evaluations, not new wet-laboratory experiments.

The published [dataset registry](../data/dataset_registry.csv) records pool sizes, units and hashes. The full local bundle separately retains `data/pools/<task>/candidates.csv`, containing 31,522 candidate rows for the eight primary tasks. Those post-hoc inspection tables were not supplied to the experimental decision agents and are not included in this GitHub checkout.

## Knowledge and comparison controls

Early Suzuki stress studies use `correct`, `wrong_category`, `mixed`, and `assistant_initial` conditions. Their definitions and source packets are retained in the early run/report archives. Artificial stress conditions are separate from later blinded, chemically grounded LLM hypotheses and from historical `agent_initial_terra` protocols.

The latest LLM methods share the same paid first input and blinded initial hypotheses. Pure BO retains its own registered zero-data initial draw; same-first BO retains the common LLM first input and uses BO afterwards. The latter is the relevant control for incremental later knowledge decisions. All methods have the same 20-query experimental budget, while LLM computation is unequal: the two newly run baselines make 19 post-first calls, whereas the current controller refreshes on events.

Paired win/tie/loss counts use a 1 e-9 metric equality tolerance in the reported summaries. The GP decision's exact acquisition ties use the registered exact-tie rule, not this reporting tolerance. Four streams are conditional on fixed initial knowledge and first input; they are not four independently sampled initial LLM hypotheses or four new GP numerical seeds.
