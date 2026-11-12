# Where Language Models Add Value in Bayesian Reaction Optimization

This directory contains the study code, the 13 candidate pools and their response arrays, the analysis tools, and the results summaries. Start with [the data catalogue](data/README.md) and [the evidence index](docs/EVIDENCE_INDEX.md).

Download the complete reproduction ZIP and SHA-256 file from the [private submission Release](https://github.com/xinxue123/Chemsys/releases/tag/submission-20261112-r1). The Release ZIP includes the saved trajectories used by the analyses and the manuscript and Supporting Information sources. Run these commands from the extracted ZIP root with Python 3.10 or later.

```text
python -B research.py verify
python -B research.py rebuild-results all
python -B tools/check_submission_evidence.py
```

The offline rebuild uses the saved observations and writes its output to a fresh directory under `output/`. The [reproducibility guide](docs/REPRODUCIBILITY.md) describes the optional figure and SNAr checks and the requirements for new BO runs.
