# Candidate pools

[`tasks.csv`](tasks.csv) indexes all 13 tasks and 38,119 candidates used in the paper. It gives the paths and SHA-256 digests of the candidate pool and oracle for each task, along with its row count, input axes, feature types and maximum raw response. The eight-task `dataset_registry.csv` is the original development benchmark index.

Each `pool.json` defines the ordered candidate identifiers and features. Its `oracle.npz` stores the responses in the same order. Where a task includes a `candidates.csv` inspection table, `objective_raw` is the response in the task's original units and `objective_normalized` is the recorded GP scale. The Direct Arylation files are in `transfer/Direct_Arylation/`. The other task paths are listed in `tasks.csv`. The optimization code queries the selected response through the oracle interface.

The pinned upstream CSVs for the three new-task screens are under `source_tables/new_task_screen/`. Its `source_manifest.json` records the upstream revision and checksums. The source license files included with the package are under `licenses/`.

See the [repository README](../README.md#figure-and-table-data) for the evidence behind each figure and table.
