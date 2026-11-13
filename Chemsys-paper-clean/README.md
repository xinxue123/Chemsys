# Falsifiable Search Scopes: paper companion

This is the GitHub edition of **Falsifiable Search Scopes: Auditing Language-Model Guidance in Bayesian Reaction Optimization**. The current study layer is `hypothesis_scope_v2`. Historical names such as `cold_start_v13` are retained dependencies, not the main study version.

The companion contains the method/comparator code, English documentation, **13 frozen candidate pools with 38,119 candidates**, compact results, and derived running-best curves. The manuscript and Supporting Information are kept in the local research archive. Raw experiment/LLM logs, original decision-state chains, per-round GP arrays, fit caches, environments and large archives remain in the separate local project.

## Start here

From this directory, with Python 3.10 or later:

```sh
python research.py datasets
python research.py verify
python research.py results
```

These commands use no model service, GP fit or experimental query. Verification checks pool/oracle hashes and candidate order, source provenance, 160 development curves, 288 held-out model/pair comparisons and the reported knowledge-audit tables. Arithmetic is recomputed from published derived curves and tables; raw decision execution is not replayed.

## Layout

| Directory | Contents |
|---|---|
| `code/experiment_tools/hypothesis_scope_v2/` | Current study runners and request transport |
| `code/experiment_tools/hypothesis_scope_loop/` | Frozen knowledge, feedback, permission and event logic |
| `code/experiment_tools/grounded_baselines/` | Matched-interface LGBO and Reasoning BO adapters |
| `code/chemist_system/` | Required GP/numerical dependencies and upstream reference files |
| `code/extensions/deepseek/` | Additional-model driver and transport |
| `data/` | All 13 public pools and evaluator oracle arrays; complete catalogue in `tasks.csv` |
| `results/` | Metrics, contrasts, knowledge audit and derived curves |
| `paper/hypothesis_scope_jcim/` | Figures, plotted data and experimental prompts |
| `provenance/` | Source hashes and filesystem/publication adaptations |
| `environment/` | Numerical requirements and observed environment |
| `docs/` | English handoff, metric definitions and reproducibility notes |
| `tools/` | Verification, local-evidence utilities and the new-campaign BO entry point |

The `.npz` files in `data/` are the 13 task oracle datasets, not fitted models or GP caches. Full-pool outcomes are evaluator data and must not enter LLM decision prompts.

## Continue research

For numerical work, use CPython 3.10 and install the recorded requirements:

```sh
python -m pip install -r environment/requirements.txt
python tools/run_bo.py --help
python tools/run_bo.py --task c1 --seed 0
```

The last command explicitly starts a new 0+20 pure-BO campaign under `output/new_bo/`. It reuses the frozen GP and exact-tie policy, warms operators once and commits a selected point before reading its outcome. It does not replace any archived baseline. No new campaign was run for this GitHub synchronization.

LLM arms require an external configuration through `CHEM_OPENAI_CONFIG` or `CHEM_DEEPSEEK_CONFIG`. New molecular background cards also require `environment/requirements-llm.txt`. Original study runners retain historical protocol/input dependencies; read [reproducibility](docs/REPRODUCIBILITY.md) before starting them.

## Evidence and restoration

Read [the handoff](docs/HANDOFF.md), [data and results](docs/DATA_AND_RESULTS.md), and [local evidence requirements](LOCAL_DATA_REQUIREMENTS.md). Full saved-evidence rebuilding and decision inspection require the separately retained local records; they are not claimed to work from a bare GitHub checkout.

The published package has its own `MANIFEST.json` and `VALIDATION.json`. It does not reuse the full local package's manifest as a checkout-integrity claim. Scientific algorithm bodies, datasets, budgets, thresholds and frozen paper results are preserved. Upstream provenance and supplied licenses remain with their components.
