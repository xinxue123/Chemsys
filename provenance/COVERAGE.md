# Provenance and coverage

The submission checkout contains all 13 pool and oracle pairs, source CSVs, scientific code, result summaries and figure and table inputs. The data attachment contains the numerical inputs for the five offline analyses and the checks for Tables 4 and S8, Test G, and Figures 2–5. Its size and checksums are recorded in RELEASE_DATA.json.

The two archives under archives/ were assembled from the exact 640 round decisions and 608 numerical GP arrays read by the analysis. The Release includes the selected trajectory states used for paper checks. Request-usage records contain the model, attempt and token counts used in the published analysis. Raw request and response files, full historical archives and caches are in the local research workspace. The source index covers the unchanged members. The Release data manifest identifies the reassembled archives and their source digests. The claim coding lineage records the source of the 99 frozen entries.

The Direct Arylation starting-point crossover combines 24 trajectories in the r3 evidence archive with 96 selected trajectories from the sealed original. The checksums and archive lineage are in first_point_crossover.json.

## Coverage limits

The original registration Markdown files for the v2 study, the CPA comparison and the new-task screen were unavailable in the checked sources. The hash-verified `screen/predictions.json` records the screening rule and summarizes 72 starting-point proposals. The CPA input receipt, 24 shared-start pairs and the separate DeepSeek protocol and amendments are included. The timing of the original registration remains unverified.

The 32 `h1` controller trajectories are included. The 32 `first_env` reference trajectories under `reproduction/runs` and the corresponding historical development-comparison summaries are unavailable. The manuscript omits that development contrast. The CPA paired comparison and the record of 41 changes among 608 decisions have separate evidence.

The cross-machine numerical example in Methods is retained in the manuscript. The analysis script is preserved, but the second-machine per-run export it reads (`results/hypothesis_scope_v2_20261105/per_seed_results.csv`) was unavailable. The 25-point example is outside the verified numerical coverage of this release.

The native runner source includes functions for experiments outside the condensed paper. Their presence follows the import graph and preserves method bodies; the unrelated experimental outputs and parameter sweeps are not included. The `rebuild_results.py` entry point deliberately selects the paper-related analyses that have retained evidence.

The pure-BO convenience runner is a new I/O entry point around the original GP/tie classes, not a new optimizer or a replacement for any archived baseline. No new 20-query optimization campaign was executed while preparing the package.
