# Data and results

This index accompanies `Chemsys-paper-clean_20261112.zip`. The [coverage record](../provenance/COVERAGE.md) identifies the retained evidence and gaps in the historical records.

## Candidate pools

`data/tasks.csv` lists all 13 task aliases, relative pool/oracle paths, candidate counts, axes, input types and file hashes. The original eight-task registry is retained as `data/dataset_registry.csv`.

| Alias | Task | Candidates |
|---|---|---:|
| c1-c4 | Reizman Suzuki Cases 1-4 | 4,692 each |
| perera | Perera Suzuki | 3,696 |
| doyle | Doyle Buchwald-Hartwig | 3,955 |
| snar | Summit SnAr | 4,096 |
| hplc | HPLC | 1,007 |
| aryl | Direct Arylation | 1,728 |
| cpa | CPA asymmetric reaction | 1,075 |
| dfl | Deoxyfluorination screen | 740 |
| bor | Nickel-catalyzed borylation screen | 1,518 |
| chx | Palladium-catalyzed coupling screen | 1,536 |

`pool.json` contains the public inputs. `oracle.npz` contains the evaluator's objective values. Some older archives store only normalized `values`; their physical responses are recovered with the original task scale, not a new normalization. Full-pool outcomes are for retrospective evaluation and must not enter LLM decision prompts.

The preserved pools have 38,119 candidates in total. They are discrete finite evaluation pools; Reizman and SnAr use benchmark/emulator outputs rather than newly performed laboratory measurements. Units and data sources are described in the manuscript, bibliography and per-pool provenance.

## Metric

For a complete 0+20 trajectory with measured outcomes y[1],...,y[20], define M[t] as the maximum observed outcome through query t. SearchAUC is mean(M[1],...,M[20]); final best is M[20]. Percentage-of-pool-maximum scores multiply the raw metric by 100 / max(pool). No warm-start observation is silently inserted. Complete-state exports require exactly 20 distinct queried candidates. Incomplete saved states are reported separately rather than evaluated using a different denominator.

## Main result locations

| Evidence | Location |
|---|---|
| Eight-task development comparison and saved one-step audits | `results/hypothesis_scope_method_evidence_20261105/` |
| Original development rows and trajectories | `results/grounded_baselines_20261105/`, `hypothesis_scope_loop_20261105/`, `hypothesis_scope_multiseed_20261105/` |
| Shared-start Direct Arylation | `results/hypothesis_scope_v2_shared_start_20261107/` |
| CPA | `results/hypothesis_scope_v2_cpa_20261107/` |
| Additional model, paired scores, costs and sequences | `results/deepseek_table4_20261110/` |
| Unequal starts and first-point/stream comparisons | `results/hypothesis_scope_v2_heldout_20261105/`, `hypothesis_scope_v2_heldout_named_20261107/`, `hypothesis_scope_v2_first_point_streams_20261107/` |
| New-task screen and exploratory later guidance | `results/hypothesis_scope_v2_screen_20261107/` |
| Test F, low-yield starts on versus away from the best level | `results/hypothesis_scope_v2_gp_followups_20261106/` |
| Test G, gate versus persistent initial constraints | paper/hypothesis_scope_jcim/figure_data/test_g_gate_ablation/ (40 pairs and verification record) |
| Claim coding, effects, W/L, verdicts, forecasts and region retention | `results/hypothesis_scope_v2_mechanism_20261106/` |
| Table S8, full-pool region quality | `paper/hypothesis_scope_jcim/figure_data/table_s8_region_quality/` (69 records and the calculation receipt) |
| Complete-state index with cohort labels | `results/trajectory_metrics.csv` |

The two model cohorts use their corresponding paired BO rows. The development unit of analysis is the task. Held-out comparisons use matched streams and the intervals, thresholds, tie treatment and contrast definitions in the manuscript.

Table S8 uses the full candidate pool. The historical `hypothesis_quality.csv` uses the candidates still unqueried and supports the GP ranking comparison.

## Figure data

The exact frozen plotted CSV files are under `paper/hypothesis_scope_jcim/figure_data/`:

- `fig2_development/`: method/task scores, summary intervals and the running-best gap.
- `fig3_models/`: current Figure 3, with separate panels for the primary model and DeepSeek.
- `table4_deepseek/`: Table 4, paired sequences, request accounting and inputs for the current Figure 3.
- `fig3_heldout_shared/`: historical primary-model Figure 3 data.
- `fig4_screen/`: first-pick percentiles, gains and the three screen-task trajectories.
- `fig5_first_point/`: group means, individual starting-point averages and the three LLM points.

These files retain the printed/exported precision. Higher precision is available in rebuilt result tables or the underlying states. A 0.00005 discrepancy against a four-decimal figure export can be rounding, not a changed experiment.
