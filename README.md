# Chem: feedback-guided experimental decisions

**Working finding: later knowledge decisions have inconsistent incremental utility beyond initialization in the tested BO studies.** Positive, null and adverse results are retained. Knowledge-state updates, changes of action and improved search are distinct evidence requirements.

This GitHub edition synchronizes code, English documentation and compact result summaries from the sealed local `Chemsys-portable` workspace. It includes **487 primary comparison records**, covering historical 4+16, original V13 0+20, shared-prefix ablations, programmed-KF contrasts and the latest grounded eight-task comparison. Reused controls are identified; these records are not 487 independent new experiments.

## Start here

- [English research handoff](docs/RESEARCH_HANDOFF.md)
- [Results and current comparison](results/README.md)
- [Metrics, cohorts and units](docs/METRICS_AND_COHORTS.md)
- [Code and data map](docs/DATA_AND_CODE_MAP.md)
- [Reproduction instructions](docs/REPRODUCIBILITY.md)
- [Local evidence requirements](LOCAL_DATA_REQUIREMENTS.md)

## Offline checks

Python 3.10 or later is sufficient; no API key or scientific packages are needed:

```sh
python chemsys.py results
python chemsys.py integrity
python chemsys.py verify
```

The checker recalculates SearchAUC/final-best summaries from **derived running-best curves**, verifies paired outcome counts and validates the published file hashes. It does not refit GP models, replay private decision-state chains, reconstruct discarded observations or call an LLM. Full native saved-evidence audits require the separately retained local evidence bundle.

## Principal evidence

| Comparison | Records | SearchAUC evidence |
|---|---:|---|
| Historical 4+16 |131 |V13 vs BO 17 wins/0 ties/15 losses over 32 pairs. |
| Original V13 0+20 |96 |V13 vs BO 22/1/9; vs Reasoning BO 13/5/14. |
| V13 continuations and transfer |75 |Six shared prefixes; no positive mean continuation-AUC increment of full V13 over fixed KF or disabled refresh. Arm-specific LLM responses also differ. |
| Programmatic KF pilot/confirmation |25 |Positive HPLC effects coexist with null Perera and adverse Direct Arylation results. |
| Latest grounded comparison |160 |Current controller vs BO 23/0/9; vs **same-first BO 8/16/8**. |

The latest controller, LGBO and Reasoning BO exceed random-start BO in 6/8,7/8 and 5/8 task means, respectively. The same-first control is needed to assess later knowledge decisions. Experimental budgets match, while LLM computation differs. These are registered matched-interface adaptations, not full reproductions of every published framework.

The latest study preserves 552 independent-subagent responses and adds 664 stateless API responses, all requesting `gpt-6.1-terra`, medium. Initial knowledge and the LLM first point are fixed across four exact-tie streams; GP numerical seed remains 0. [Backend summaries](results/grounded_baselines_20261105/LLM_backend_by_trajectory.csv) make the transition explicit. The evidence is conditional on these settings.

## Published and local-only assets

Published: research code, English handoff/protocol explanations, dataset descriptions/hashes, complete primary metric tables, paired summaries, derived cumulative-best curves and the current comparison figure.

Local-only: raw experiment/decision logs, original LLM packets, selected-condition histories, full candidate/oracle tables, GP arrays/checkpoints/caches,72 large evidence archives, detailed archive member/time indices, compiled RDKit support and Python environments. The complete local bundle is about 16.22 GB; its 399,703 archive members remain local. See [LOCAL_DATA_REQUIREMENTS.md](LOCAL_DATA_REQUIREMENTS.md) for restoration and provenance.

The original repository history and historical source documentation remain available. Newly synchronized narrative documents are English. Three portable source adaptations and their exact native counterparts are documented; native source/hash audits use the full local source capture. Credentials are excluded. No experiment or model service starts automatically.
