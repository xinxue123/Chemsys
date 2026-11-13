# Where Language Models Add Value in Bayesian Reaction Optimization

This repository contains the search-region controller, the GP and comparator implementations used in the paper, the 13 candidate pools, and the data behind the reported figures and tables.

## Reproduce the results

Use Python 3.10 or later. Place [the r3 evidence archive][r3] in this directory. The installer checks its SHA-256 and the checksum of each file before extracting the saved trajectories and numerical inputs. The offline checks use saved observations and require no model credentials. Git ignores the Release archive and its extracted files.

```text
python -B tools/attach_release_data.py Chemsys-submission_20261112-r3-evidence.zip
python -B research.py verify
python -B research.py rebuild-results all
python -B tools/check_submission_evidence.py
```

Install the versions in `environment/requirements-submission-checks.txt` to recalculate the SNAr sensitivity and rebuild Figures 2–5.

```text
python -m pip install -r environment/requirements-submission-checks.txt
python -B tools/check_submission_evidence.py --snar
python -B tools/build_figures.py
```

The rebuild commands write their results under `output/`. The frozen tables and their numerical inputs stay in the recorded locations.

## Figure and table data

| Item | Files |
| --- | --- |
| Table 1 and task grids | [Task catalogue](data/tasks.csv), candidate pools, oracle arrays and pinned source tables under `data/` |
| Table 2, worked Direct Arylation example | Saved states and forecasts in the [r3 evidence archive][r3] |
| Table 3 and Figure 2 | [Development results](results/hypothesis_scope_method_evidence_20261105) and [plot data](data/figure_data/fig2_development) |
| Table 4 and Figure 3 | [Model-specific paired data](results/deepseek_table4_20261110), [plot data](data/figure_data/fig3_models) and [shared-start results](results/hypothesis_scope_v2_shared_start_20261107) |
| Table 5 and Figure 4 | [Screen results](results/hypothesis_scope_v2_screen_20261107) and [plot data](data/figure_data/fig4_screen) |
| Direct Arylation starting-point crossover | [96 additional sealed trajectories](data/first_point_crossover_states.zip), [file hashes](provenance/first_point_crossover.json) and the r3 evidence archive |
| Figure 5 and Test F | [First-point comparisons](results/hypothesis_scope_v2_gp_followups_20261106) and [plot data](data/figure_data/fig5_first_point) |
| Table 6 and forecast checks | [Claim coding, matched contrasts and forecast results](results/hypothesis_scope_v2_mechanism_20261106) |
| Table S8 and Test G | [69 region records](data/figure_data/table_s8_region_quality) and [40 gate-ablation pairs](data/figure_data/test_g_gate_ablation) |

The controller and the recorded comparator adapters are in `code/`. The [source coverage](provenance/COVERAGE.md) lists the original records available for independent verification. The candidate and response file formats are described in [the data README](data/README.md).

To reproduce the random-stream comparison in the Supporting Information after attaching the data, run:

```text
python -B code/analysis/analyze_hypothesis_scope_v2_first_point_crossover.py experiments/runs/hypothesis_scope_v2_20261105 output/first_point_crossover
```

[r3]: https://github.com/xinxue123/Chemsys/releases/download/submission-20261112-r3/Chemsys-submission_20261112-r3-evidence.zip
