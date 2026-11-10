# Data, metrics and result locations

## Datasets

`data/tasks.csv` lists relative paths, candidate counts, input axes/types, task maxima and SHA-256 hashes for all 13 tasks. The original eight-task registry remains in `data/dataset_registry.csv`.

| Alias | Task | Candidates |
|---|---|---:|
| c1-c4 | Reizman Suzuki Cases 1-4 | 4,692 each |
| perera | Perera Suzuki | 3,696 |
| doyle | Doyle Buchwald-Hartwig | 3,955 |
| snar | Summit SnAr | 4,096 |
| hplc | HPLC | 1,007 |
| aryl | Direct Arylation | 1,728 |
| cpa | CPA asymmetric reaction | 1,075 |
| dfl | Deoxyfluorination | 740 |
| bor | Nickel-catalyzed borylation | 1,518 |
| chx | Palladium-catalyzed coupling | 1,536 |

The pools total 38,119 candidates. `pool.json` supplies public inputs; `oracle.npz` supplies evaluator outcomes. Older archives that contain only normalized `values` use their original task scale. Reizman/SnAr results include benchmark/emulator outputs rather than new laboratory measurements. Units and sources are retained in the manuscript, bibliography, original registry and per-pool provenance.

## SearchAUC and comparisons

For a complete 0+20 trajectory, M[t] is the largest measured outcome through query t. SearchAUC is mean(M[1],...,M[20]); final best is M[20]. Percentage-of-pool-maximum metrics multiply by 100 / max(pool). Development comparisons use tasks as the analysis unit; held-out comparisons use the registered pair/stream. The reporting equality tolerance, 90% intervals, thresholds and original pairings are unchanged.

Do not pool the primary and DeepSeek cohorts or substitute a BO reference from another cohort. The published running-best tables contain derived cumulative maxima rather than raw decision logs. The original raw-state index in `trajectory_metrics.csv` identifies local sources; its rows are not all independent comparison units and its referenced raw paths are not files claimed to be on GitHub.

## Results

| Evidence | Directory |
|---|---|
| Development methods, task means and saved one-step audits | `results/hypothesis_scope_method_evidence_20261105/` |
| Development metrics and derived curves | `results/grounded_baselines_20261105/` |
| Direct Arylation shared starts | `results/hypothesis_scope_v2_shared_start_20261107/` |
| CPA | `results/hypothesis_scope_v2_cpa_20261107/` |
| Both models, paired values and costs | `results/deepseek_table4_20261110/` |
| Unequal starts and first-point/stream comparisons | `results/hypothesis_scope_v2_heldout_20261105/`, `hypothesis_scope_v2_heldout_named_20261107/`, `hypothesis_scope_v2_first_point_streams_20261107/` |
| New-task screen | `results/hypothesis_scope_v2_screen_20261107/` |
| Test F | `results/hypothesis_scope_v2_gp_followups_20261106/` |
| Claim coding, effects, W/L, verdicts and forecast summaries | `results/hypothesis_scope_v2_mechanism_20261106/` |

The exact plotted data are under `paper/hypothesis_scope_jcim/figure_data/`. Frozen CSV exports retain their displayed precision; a difference of up to 0.00005 against a four-decimal export can be rounding. Higher precision is available in result tables. Small audit receipts report earlier checks; they are not substitutes for excluded raw records.
