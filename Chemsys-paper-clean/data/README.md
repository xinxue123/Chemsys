# Dataset inspection tables

`dataset_registry.csv` lists the eight primary tasks and exact input/oracle hashes. Each `pools/<task>/` folder contains original `pool.json`, `oracle.npz`, full physical/category-unit `candidates.csv` and provenance. Total:31,522 candidate rows.

`objective_raw` is the original task objective. `objective_normalized` is the frozen GP target scale; the conversion is in `provenance.json`. Full outcome tables are for post-hoc research inspection and were not supplied to experimental agents.

The complete benchmark subtree, source tables and licenses is in `archives/datasets.zip`. The Direct Arylation transfer pool is in `archives/run__v13_late_validation_20261103.zip` under `holdout/`. See `docs/METRICS_AND_COHORTS.md` for identities and units.

A direct inspection copy of that additional transfer input/oracle table is also available in `data/transfer/Direct_Arylation/`, with its original receipt and hashes. It is separate from the eight-task31,522-row primary export.
