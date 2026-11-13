# R6: evidence-reversible qualitative knowledge in finite-pool chemical BO

**Status: new isolated implementation + deterministic unit tests. Real SnAr/HPLC reruns are pending.**

## What is being fixed

The archived R5 evidence reported only 2/8 executable LLM nominations, 1 actual
non-BO nomination, 0 scheduled validations, and 0 deployed local mean updates.
The R5 SnAr/HPLC SearchAUC differences against their own pure-BO comparators
were -5.904% and +6.438%, respectively. Those tasks have different objective
units and must not be aggregated into a two-task average.

The dominant failure modes were: numerical LLM forecasts hard to calibrate;
most blindly proposed real inputs outside the current qLogEI opportunity gate;
hard two-positive local-evidence thresholds; and continuation from a protected
decision prefix rather than from the same four initial observations as pure BO.
No prior R5 protocol, GP adapter, old policy or archived artifact is changed.

## R6 decision protocol

1. **Same GP as the existing experiment.** Call the original `SingleTaskGP`
   fit, original seed, original `q=1, MC=512` qLogEI, original tie rule, and
   original time-dependent gate `log(4)*(16-t)/15`, with `t=1..16`. The
   original GP is always fit to **actual observations only**.
2. **Real qualitative comparisons.** The LLM can only declare `x+ > x-` for
   two displayed, unqueried *physical* input points, with 1-2 signed focus
   axes and bounded off-focus changes. It cannot create measurements or
   numerical yields. Reversed/invalid declarations are retained as rejections.
3. **Executable candidate menus.** A1 selects a GP-blind physical menu;
   A2-A4 select a hidden-score menu inside the normal GP gate. Membership
   leaks feasibility and is correctly labelled *GP-filtered reranking*.
   A nearby actual pair is included if the farthest-first menu has no
   semantically comparable pair. If none exists, the call is skipped.
4. **One qualitative soft constraint.** Use a noisy inequality
   `f(x+) - f(x-) + noise > 0`, with a fixed engineering-normalized
   preference-noise SD of `0.06`. The baseline GP never treats the
   inequality as an observed objective value. The q=1 conditional EI ratio
   is evaluated by 64-point Gaussian-Hermite quadrature over the
   **skew-normal** one-sign posterior, with predeclared log-ratio cap
   `|log(EI_h/EI_0)| <= log(4)`. The ratio is applied to original MC qLogEI.
5. **Prospective evidence.** Before revealing an experimental result,
   commit GP mean, predictive variance, GP latent contrast mean and variance,
   and cross-covariance with the selected real point. After revealing the
   result, update the log predictive density ratio for each informative
   in-scope hypothesis. A negative update retracts deployment weight.
   Cumulative scores after GP refitting are *prequential scores*, not exact
   stationary Bayes factors or calibrated chemistry-truth probabilities.
6. **Limited active validation.** A3/A4 can spend up to two of the **same
   16 experiments**, at rounds `t<=10`, on a point with additional
   Gaussian contrast-information proxy relative to the pure-BO point.
   First attempt a point within the normal gate. A3/A4 allow at most one
   explicit extra `log(2)` opportunity-loss exception; `A4_no_exception`
   prohibits it. No regret/no-harm guarantee is claimed.
7. **Safety and locality.** Evidence deployment is initially zero until
   prospective evidence becomes positive, capped at 0.5; negative evidence
   deactivates it. Only one best supported local statement guides selection;
   outside its normalized `0.25`-radius focus/background neighborhood,
   acquisition remains the original GP score. The actual selected point
   remains in the original gate, except the explicitly logged verification.

## Six registered arms

| Arm | Menu / decision | Active validation | Evidential preference EI |
|---|---|---|---|
| A0 | Pure BO | No | No |
| A1 | GP-blind real-input nomination, then original gate | No | No |
| A2 | GP-filtered immediate qualitative rerank | No | No |
| A3 | A2 plus prequery contrast information tests | Up to 2 | No |
| A4 | GP-filtered; **no direct unverified LLM deployment** | Up to 2 | Yes |
| A4_no_exception | A4 with only normal-gate tests | Up to 2 | Yes |

All arms use the same 4 initial points for each task/seed, then independently
run 16 new queries. No R2/R3/R4/R5 decisions are copied. This removes the
impossible protected-prefix comparison. Query costs and LLM calls are logged
per arm; LLM-bearing arms are not claimed to be cost-matched to A0.

## Run on the original ChemSys research environment

The GitHub source tree does not contain the original `data/benchmarks` oracle
arrays or historical GP cache. The original research environment must provide:

- `data/benchmarks/continuous_20261029/{SnAr,HPLC}/pool.json`
- `data/benchmarks/continuous_20261029/{SnAr,HPLC}/oracle.npz`
- the original frozen `SingleTaskGP` / BoTorch / GPyTorch / filelock setup
  and configured DeepSeek transport for arms A1-A4.

From `code/chemist_system`:

```bash
python continuous_r6.py qualify
python continuous_r6.py prepare
python -m unittest discover -s tests -p test_continuous_r6.py -v
python continuous_r6.py run --task SnAr --arm A0 --seed 0
python continuous_r6.py run --task SnAr --arm A4 --seed 0 --config ../config.txt
python continuous_r6.py audit
python continuous_r6.py status
python continuous_r6.py report
```

Repeat the runs for both tasks, all six arms, and independently chosen seeds
(start with 0-5 for a development set). Run A0 first for each task/seed.
Use a different frozen seed set for a later confirmatory experiment.
`prepare` freezes source/data SHA256; changing implementation or source data
after that intentionally prevents silent continuation.

Each run has a separate checkpoint directory under
`data/runs/continuous_r6_20261030/runs/<task>_<arm>_seed<seed>`.
For each t, `rounds/TT_committed.json` is **persisted before**
`data.reveal()`. On interruption the runner restores that exact commitment
without another LLM request or another selection. GP fit cache is kept in
`data/runs/continuous_r6_20261030/cache`, not in historical R5 paths.

## Acceptance and reporting

Before any performance claim, require: 100% no future outcome access during
selection, all 16 queries committed, no duplicate points, normal gate
respected unless the predeclared validation exception was used, negative
knowledge capable of returning weight to zero, and a nonzero **incremental**
knowledge-driven selection count in A4 relative to A2/A3.

For each task separately, compare SearchAUC (mean incumbent of the 16 added
queries), final best, paired seed differences, confidence intervals,
verification frequency, exception AUC cost, prospective evidence scores,
deployment changes, abstention rates and actual LLM cost. The report computes
bootstrap intervals only with at least six paired seed trajectories. Neither
passing unit tests nor seed0 performance proves generalizable improvement.

### Local verification completed

`python -m unittest discover -s tests -p test_continuous_r6.py -v`
**30 tests passed** (19 standalone mechanisms + 10 repository adapter checks,
including joint-covariance scaling, plus 1 pre-observation
commit/crash-resume test).
These do not substitute for real BoTorch/DeepSeek optimization runs: the
authoring environment did not contain Torch/BoTorch or the original oracle.
