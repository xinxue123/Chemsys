# Reproducibility

## Published-data verification

Run `python research.py verify` with Python 3.10 or later. No API key or scientific package is required. The checker validates dataset and source hashes, ID order, finite outcomes, published derived curves, metric arithmetic, knowledge-audit rows and the publication manifest. It does not fit a GP, reconstruct private decision histories or call a model.

The 160 development curves and 288 model/pair curves are evaluated with their original 20-query denominator. Cohorts, units and BO references remain separate. The code does not reinterpret missing raw evidence as a successful native replay.

## New numerical campaigns

Use CPython 3.10 and `environment/requirements.txt` for the original numerical stack: CPU PyTorch 2.5.1, BoTorch 0.14.0, GPyTorch 1.14, NumPy 1.24.4, SciPy 1.12.0 and pandas 1.5.3. `environment/requirements-llm.txt` adds the original RDKit version for new molecular background cards.

`python tools/run_bo.py --task c1 --seed 0` explicitly starts a new 0+20 BO campaign under `output/new_bo/`. It uses the original `SeededGP` and `ColdGP`, numerical seed 0 and feature-keyed exact-tie stream. The zero-data choice uses a neutral seeded order; qLogEI begins after a measured observation exists. Operators warm once before the loop, and the chosen point is committed before its oracle outcome is read. New runs are not replacements for frozen controls. No new campaign was run for publication.

## Original LLM and saved-evidence runners

The original source files are included. Their full study orchestration requires the historical protocol/input/record environment. Restore the relevant files from the separate local project as described in [local requirements](../LOCAL_DATA_REQUIREMENTS.md) before using `rebuild-results`, `inspect` or original study actions. The published tools report missing local evidence instead of substituting data or relaxing hash checks.

`python tools/prepare_inputs.py` materializes the original eight development input paths from the published pools without a query. This alone does not restore a historical experiment. Supply model configurations externally through `CHEM_OPENAI_CONFIG` or `CHEM_DEEPSEEK_CONFIG`. No credential or environment installation is bundled.

## Paper

The English manuscript, SI, references and frozen figures are supplied under `paper/hypothesis_scope_jcim/`. Compile SI first and make its `si.aux` available when compiling the manuscript, which uses `xr` for SI labels. The provided PDFs can be read without LaTeX. Scientific results, prompts, thresholds and figure layouts are unchanged in this publication projection.
