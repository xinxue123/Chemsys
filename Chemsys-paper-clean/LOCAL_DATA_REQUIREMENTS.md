# Local evidence and restoration

This GitHub companion publishes code, English documentation, all 13 task datasets and compact derived results. The complete local `Chemsys-paper-clean` project and original research archives remain separate from this checkout.

| Retained locally | Purpose |
|---|---|
| `experiments/runs/hypothesis_scope_v2_20261105/` | Primary-model states, commits, feedback, validated outputs and input/design records |
| `evidence/deepseek/` | New-model states, local matched controls and frozen references |
| `archives/run__hypothesis_scope_loop_20261105.zip` | Filtered development controller decisions and saved GP arrays |
| `archives/run__hypothesis_scope_multiseed_20261105.zip` | Corresponding seed 1-3 evidence |
| Full original sealed records and historical archives | Complete available raw custody and later diagnostic sweeps |
| Fit caches, checkpoints, runtime installations and console output | Original numerical environment and execution diagnostics |

For a full native saved-evidence analysis, use a separate copy of the **full local project**. It already contains the required record layout. From that project:

```sh
python research.py rebuild-results all
python research.py inspect --task c1 --seed 0 --query 4
```

The GitHub edition has its own verification scope and file manifest. Copying an archive into this checkout does not establish that its scientific recipe or source hashes match; use the preserved local project and its full validation receipt. Credentials remain external.

Some historical same-machine development reference runs and standalone original registration Markdown files were absent from the supplied research archive. They are not regenerated from substitute data or given new historical dates. Current manuscript summaries and the available design/provenance records are preserved.
