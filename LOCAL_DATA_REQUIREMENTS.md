# Local-only evidence and restoration

This repository is the **code/documentation/compact-results edition**. The sealed local `Chemsys-portable` bundle is a separate, approximately 16.22 GB research artifact. Copying or cloning this repository does not download that artifact.

| Kept local | Purpose |
|---|---|
| `archives/*.zip` (72 archives,399,703 source members) |Complete available native runs, reports, source captures, packets, manuscripts and milestones. |
| Raw `state.json`, commit/feedback JSON, LLM requests/responses and console logs |Original decision execution and actual-response custody. |
| GP `.npz` arrays, `.pt` checkpoints and numerical caches |Full candidate predictions, fitting and replay evidence. |
| `data/pools`, transfer oracle arrays and complete candidate outcomes |Full benchmark input/label inspection and future optimization. |
| `data_catalogue/files`, `states`, `file_times` |Detailed source/member/timestamp inventories. |
| Windows RDKit support package and runtime environments |Molecular-background reconstruction and original software setup. |

The [archive metadata](data_catalogue/archive_catalogue.csv) and [experiment register](data_catalogue/experiment_register.csv) describe what exists locally, including earlier development, failures and partial states. Paths and hashes in result CSVs identify local evidence; those paths are not claims that raw files are present on GitHub. Small package data already tracked with upstream code are preserved.

## Restore for native audit

Use the **full local bundle in a separate directory**. From that bundle, with the documented Python environment:

```sh
python chemsys.py materialize --destination C:/ChemReplay
# Add --all there to materialize every historical cohort.
```

Then, inside the new native workspace:

```sh
python code/tools/audit_grounded_baselines_api_v2.py --final
```

The GitHub edition's launcher deliberately exposes only results/integrity/catalogue checks. Native optimization and diagnostic source remains included under `code/`, but running those entrypoints requires the original data/protocol environment. Do not invent replacement files or relax frozen hash checks to make a missing-data audit pass.

## Reported local validation

The complete bundle passed 399,703 archive-member SHA-256/CRC checks. Its latest 64 trajectories passed an isolated native saved audit over 1280 committed choices and 1216 actual response receipts, with original-workspace scientific data and network/helper access denied. [Local validation receipts](data_catalogue/local_validation/) report that earlier completed check; they do not imply the same raw-state audit was rerun from this compact checkout.

Original logs, checkpoints, large archives and the sealed full-bundle manifest remain unchanged locally. The lightweight release uses a separate `PUBLICATION_MANIFEST.json`; it does not reuse the full bundle's 16 GB manifest as a checkout-integrity claim. No credentials or native runtime binaries are synchronized.
