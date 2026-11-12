# Where Language Models Add Value in Bayesian Reaction Optimization

This directory contains the r2 analysis code, 13 candidate pools, and published result tables. The large numerical and trajectory inputs are in the [r2 evidence Release](https://github.com/xinxue123/Chemsys/releases/tag/submission-20261112-r2). The attachment contains data used by the offline analysis. The manuscript sources and PDFs are maintained with the local submission bundle.

## Offline reproduction

1. Download Chemsys-submission_20261112-r2-evidence.zip from the Release and keep it outside this directory.
2. Open a terminal in this directory with Python 3.10 or later.
3. Install the optional local analysis dependencies from environment/requirements-submission-checks.txt for the SNAr sensitivity and plots.

```text
python -B tools/attach_release_data.py PATH_TO_DOWNLOADED_ZIP
python -B research.py verify
python -B research.py rebuild-results all
python -B tools/check_submission_evidence.py --snar
python -B tools/build_figures.py
```

The installer verifies the attachment against RELEASE_DATA.json. Rebuilding the five analyses uses saved data and compares the results with the frozen tables. See [the evidence index](docs/EVIDENCE_INDEX.md), [the workflow](docs/REPRODUCIBILITY.md) and [the source coverage](provenance/COVERAGE.md).
