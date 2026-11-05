# Reproducing the published summaries

## Checks available in this checkout

Use Python 3.10 or later from the repository root:

```sh
python chemsys.py results
python chemsys.py integrity
python chemsys.py verify
python chemsys.py catalogue
```

These commands use the standard library and do not request an API key. They recompute metric arithmetic from derived cumulative-best curves, check 487 comparison records across their distinct budget/cohort definitions, verify paired win/tie/loss counts and check published file hashes. They do not reconstruct raw observations or repeat the full native execution audit.

Historical 4+16 curves average 16 post-initialization running maxima; original and current 0+20 curves average 20 paid-query maxima. Shared-prefix continuations retain their separate remaining-budget and full 20-position metrics. Never pool these denominators. Source paths/hashes in summary CSVs refer to local evidence.

## Native saved-evidence audits and new research

Restore the full local bundle separately as described in [LOCAL_DATA_REQUIREMENTS.md](../LOCAL_DATA_REQUIREMENTS.md). That bundle contains the original sources, protocols, raw logs, LLM packets, oracle pools, GP arrays/checkpoints, file times and archive indices required by native hash/audit scripts. Its original Windows CPython 3.10 RDKit 2024.9.6 support capture is local-only. Other platforms need their own matching RDKit installation.

The [reported isolated local audit](../data_catalogue/local_validation/relocation.json) passed with original-workspace scientific data and network access denied. This validates that full bundle's latest-study saved replay, not a fresh numerical rerun of every historical variant. Earlier versions can require their original registered environment/source capture. [Observed software versions](../environment/observed_environment.json) and `requirements-native-audit.txt` help identify dependencies; installing packages alone does not restore evidence.

The active code includes three pre-existing portable adaptations. Use exact native source captures for frozen native source/hash audits. New experiments should use a separately registered protocol and preserve the old pools, results and source/checkpoint hashes. The new experiment's first-input, GP/tie streams, knowledge conditions and backend must be explicit.

Warm/precompile common operators once per persistent worker. Use available CPU/GPU capacity when compatible with the registered numerical recipe; do not change precision or GP backend merely to increase utilization. Published checks perform zero experiments, GP fits and LLM calls.
