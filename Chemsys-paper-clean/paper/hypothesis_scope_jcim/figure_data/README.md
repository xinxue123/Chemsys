# Figure data

The values plotted in each figure of the manuscript, one folder per figure. They are written by `tools/export_figure_data.py` from the same saved result tables that `tools/make_figures.py` reads, with no model calls, queries or GP fits. Scores are percentages of each task's pool maximum. Figure 1 is a diagram and has no data.

| Folder | Figure | Files |
|---|---|---|
| `fig2_development/` | Figure 2, development tasks | `panel_a_by_task.csv` (SearchAUC per method and task), `panel_a_summary.csv` (means, 90% t-intervals over 8 tasks, LLM calls), `panel_b_running_best_gap.csv` (LLM-start minus random-start BO running best, by query) |
| `fig3_heldout_shared/` | Figure 3, held-out tasks from 24 shared starts | `trajectories.csv` (every query of every pair), `panel_a_mean_running_best.csv`, `panels_b_c_pair_differences.csv` (each method minus random-start BO, per pair), `panels_b_c_summary.csv` (means and 90% t-intervals) |
| `fig4_screen/` | Figure 4, when an LLM start helps | `panel_a_pick_percentile_vs_gain.csv`, `panels_b_d_trajectories.csv` (24 streams per task), `panels_b_d_mean_running_best.csv` |
| `fig5_first_point/` | Figure 5, Test F | `points_by_first_point.csv` (each failed first point, averaged over its streams), `group_summary.csv` (group means, gap, bootstrap interval, permutation p, share of queries 2–10 on the best level), `llm_first_points.csv` (the three LLM first picks on Direct Arylation) |

Bootstrap bands in Figures 3–5 are resampled when the figures are drawn, so they are not stored; they can be recomputed from the trajectories and per-point files.

```powershell
python tools/export_figure_data.py
```
