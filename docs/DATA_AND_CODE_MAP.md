# Data and code map

> GitHub compact edition: code, summaries and derived curves are published here. Raw evidence, detailed inventories, checkpoints and large archives remain in the separate local bundle. See [local evidence requirements](../LOCAL_DATA_REQUIREMENTS.md). Descriptions of full-bundle audits refer to the recorded local validation.

| Location | Contents | Use |
|---|---|---|
| `results/combined_primary_metrics.csv` |487 indexed primary comparison rows, original state paths and hashes | Find a result and its cohort before reading a trace. |
| `results/cohorts/warm_4plus16/` |All 131 historical warm metric rows and derived post-initialization running-best curves |4+16 evidence; no cold/warm pooling. |
| `results/cohorts/v13_cold_0plus20/` |All 96 original V13 cold metric rows and derived running-best curves | Original zero-start V13/BO/RBO comparison. |
| `results/cohorts/v13_late_and_transfer/` |All 75 continuation/transfer metrics, conditional effects, coverage summaries and derived curves | Detailed mechanism calculations remain local. |
| `results/kf_weight/`, `results/kf_confirmation/` |Programmatic weight metrics, paired effects, divergence summaries and derived curves | Raw replay fixtures remain local. |
| `results/grounded_baselines_20261105/` |Latest 160 metric rows, 3200 derived running-best positions, task means/SD, initial-vs-continuation decomposition and backend summaries | Current main comparison and same-first control. |
| `results/hypothesis_scope_multiseed_20261105/` |Frozen current-controller and BO controls |Trace the source of imported reference results. |
| `data/pools/` |Eight input/oracle pools, complete candidate CSVs and provenance | Offline dataset inspection. Keep oracle labels outside decision prompts. |
| `archives/run__<study>.zip` |Every available scientific file for that run group | Exact states, commits, feedback, predictions, GP arrays, checkpoints, caches, source captures and request/response receipts. |
| `archives/reports.zip` |All available original development and final reports | Includes stress tests, partial studies, failures and diagnoses. |
| `archives/agent_packets.zip` |All available original public experimental LLM input/output packets | Reconcile actual response custody; no credentials. |
| `archives/source_code.zip` |Exact native source tree at this capture | Materialize before native source/hash audits. |
| `archives/original_protocols.zip` |Original protocol documents, verbatim | Historical language and source hashes preserved. |
| `archives/datasets.zip` |Complete available benchmark dataset subtree | Includes raw source tables/licenses alongside standardized pools. |
| `archives/historical_manuscripts.zip` |Original manuscript trees and release files | Historical evidence custody, not a rewritten current manuscript. |
| `archives/milestones.zip` |Original milestone releases and receipts | Preserve prior sealed baselines. |
| `archives/auxiliary_evidence.zip` |Metadata, console/debug logs, PR validation, verification copies and operational receipts | Technical issues and negative implementation evidence. |
| `data_catalogue/files/` |Member path, size, original/exported SHA-256 and redaction status for each archive | Locate or check any available source file. |
| `data_catalogue/states/` |Raw state-file inventories and recorded status/count fields | Inventory only; duplicate snapshots and partial states are not independent completed experiments. |
| `data_catalogue/experiment_register.csv` |English description, budget family, evidence tier and raw-status counts for all 63 run groups |Navigate the full development archive without pooling unrelated protocols. |
| `vendor/rdkit_runtime/` |Original RDKit 2024.9.6 Windows CPython 3.10 distribution and license metadata | Reconstruct deterministic molecular-graph background in the native audit; separate from the 72 evidence archives. |
| `data_catalogue/excluded/` |Per-archive transient/reparse/credential exclusions | Inspect the capture boundary explicitly. |
| `paper/v13_late/`, `paper/v13_confirmation/` |Previously exported English manuscript/supplement/addendum, unchanged | Original arithmetic verifiers and frozen documents. |

The archive, full-pool, compiled-runtime and detailed-index paths in this map refer to the separate local bundle. The GitHub edition contains their compact metadata, code and result summaries; see [LOCAL_DATA_REQUIREMENTS.md](../LOCAL_DATA_REQUIREMENTS.md) for restoration. The historical manuscript sources are kept with the local research archive.

The 72 local archives contain 399,703 exported source members from 63 run groups and nine other archive groups. Data are preserved by file, without selecting only successful trajectories. Runtime environments, Git repositories/dependency caches, transient lock/tmp/bytecode files, and unrelated Ce–Al–Ge/Materials Project/XRD assets are outside this chemical optimization capture.

## Decision-log fields

In the current grounded run, `rounds/NN_commit.json` records the chosen candidate, physical conditions, prestate hash, own-prefix BO reference, acquisition scores and actual LLM receipt before the selected label is read. `NN_feedback.json` records the resulting observation and commit hash. `NN_GP.npz` contains full-pool GP mean, latent/predictive SD, availability, base/final log acquisition, tie priority and regional mean shift where applicable. RBO commits record top-five membership and neutral display order; LGBO commits record physical region/quadrature details. `state.json` records the complete committed history, own preferences/Insights, decisions and call links.

The current hypothesis controller additionally records rule versions, semantic evidence inheritance, prediction credibility, recommendation permission, refresh events and constraints. Older schemas have their own registered field names; inspect the matching study's source capture rather than assuming the latest schema applies to all versions.

Request directories contain original scientific contexts, attempts, validation, assignment/completion receipts and raw public responses. For the API continuation, `transport_type=openai_responses_api` identifies the actual backend. The compatibility key `agent_id` then holds an HTTP call identifier; it is not a collaboration-agent claim. Provider response IDs, reported model and token usage are retained. Old planned `invocation.json` files remain historical records.

## Code versions and supported entrypoints

`code/` retains the previous standalone portable adapters and adds the later modules. Three differing adapters are explicitly listed in [portable_code_differences.csv](../data_catalogue/portable_code_differences.csv). They are `run_suzuki_initial_yield.py`, `llm_event_start/study.py` and `fast_hit_bo/study.py`. Native-source custody is provided by `archives/source_code.zip` and the per-study `source_snapshot` directories.

For current-controller inspection, start with `code/experiment_tools/hypothesis_scope_loop/knowledge.py` and `policy.py`; the multi-stream wrapper is `hypothesis_scope_multiseed/`. LGBO/RBO comparison code is `grounded_baselines/`. The API transport is `code/tools/grounded_baselines_api_dispatch_v2.py`. Its source is included for custody; offline verification does not start it.

Use the project launcher and the reproduction guide for supported commands. Native full-study audit uses `code/tools/audit_grounded_baselines_api_v2.py --final` **inside a newly materialized native workspace**. The old subagent-only auditor/finalizer does not describe the mixed-backend continuation. Completed studies are frozen and are not resumed or extended by the handoff tools.
