# Proposed intervention within acquisition-indistinguishable candidates

Status: design for a new experiment, 2026-11-04. No new optimization trajectories have been run for this proposal.

The Suzuki 1+19 study shows substantial recovery from low-yield starts under pure BO, with remaining differences in discovery speed and target-hit rates. Case 2 reaches the pool maximum in all five low and all five middle starts. Case 3 reaches 95% of the pool maximum in two low and four middle starts. These data support partial resilience to initialization and task-specific sensitivity.

The saved acquisition audit adds a numerical qualification. At every one-observation state, the logEI spread across all 4,691 unqueried points is at most 4.09 e-13. The singleton target is standardized to zero and the learned signal variance nearly collapses. Strict equality splits nearly flat scores into different tie sets. The effect of an LLM must be compared with a BO control using the same numerical grouping. Numerical optimization is a known source of BO performance variation. [LogEI paper](https://arxiv.org/abs/2310.20708)

Across queries 3–20 in forty distinct trajectories, exact top-score ties occur in 128 of 720 decisions. A diagnostic set within log(1.05) of the maximum has more than one candidate in 451 decisions. The latter is an opportunity diagnostic. Five percent has not been validated as a deployment threshold.

| Case | Later decisions | Exact top-score ties | More than one candidate within the diagnostic 5% band |
|---|---|---|---|
| 1 | 180 | 41 | 107 |
| 2 | 180 | 43 | 110 |
| 3 | 180 | 36 | 114 |
| 4 | 180 | 8 | 120 |

A new candidate interface would use the full acquisition-indistinguishable set. A fixed, feature-based coverage rule would expose representatives across catalyst and continuous-condition regions. The LLM could also propose a local region that is projected onto a measured candidate inside that set. Projection and admissibility would use inputs and acquisition values only. Representative construction would be identical across all experimental arms. Restricting the model to the first five tied IDs could omit its intended region before it is asked for advice.

1. The GP fits the observed prefix and supplies qLogEI. The controller identifies numerical ties. A separate, prospectively registered near-optimal band can be tested after the exact-tie intervention. The reported five-percent diagnostic does not set the experimental threshold automatically.
2. The LLM receives observed conditions and outcomes, executable candidates, and scoped adverse facts. It proposes a local comparison and forecasts improvement probability and improvement magnitude before selection. It receives no unqueried yields, global optimum, or outcome-defined candidate groups.
3. A bounded gate decides whether the controller follows the LLM or the shared BO tie rule. In a randomized version, the probability of following the LLM is rho. The controller records rho, the actual sampled decision, both proposals, and the random stream before measurement. An isolated validation allowance is needed for previously untested knowledge.
4. The scalar KF updates the calibration of forecasts at the executed point. It may adjust rho. Directional permission continues to depend on actual improving tests. Accurate failure can strengthen a warning while positive promotion remains suppressed. A single executed outcome does not reveal the outcome of the unexecuted BO alternative, so it cannot be labeled as a demonstrated LLM-over-BO selection gain.
5. Events such as inaccurate predictions, stagnation, or a newly executable conflicting hypothesis can request a refresh when another intervention opportunity exists. The editor must identify which claim changed and preserve relevant adverse evidence. After GP distinctions become clear, the controller follows the GP and carries the feedback state into the next eligible opportunity.

The gate addresses a structural limitation of the prior rank fusion. If GP rank is constant across a set, maximizing (1-w)c + w*LLM_score selects the same LLM favorite for every positive w. A KF can change w without changing the action. A gate or randomized mixture can switch between the LLM and BO proposals, so numerical trust has an observable deployment effect.

The first experiment should hold prefixes and candidate menus fixed and compare native tie selection, random or feature-diverse selection within the same set, and LLM selection. An evaluator may look up the committed alternative outcomes only after all choices are sealed. If the LLM improves these choices, an equal-budget continuation can compare fixed LLM deployment, KF deployment, and KF with event refresh. Common-input responses should be shared. After histories diverge, language-response variation should be reported.

The primary outcome remains SearchAUC over all twenty paid observations. Target-hit queries and final best supplement it. Record eligible opportunities, actual LLM interventions, acted-on feedback, forecast errors, and observed selection gains separately. The method needs to improve over a diversity rule to support a chemistry-knowledge contribution. Improvement over an arbitrary tie order alone leaves the source of benefit unresolved.

This proposal changes candidate admission and deployment at the selection stage. It preserves the GP posterior. Its performance and KF/refresh increments require new validation. Reasoning BO provides related precedent for using hypotheses to choose among BO candidates and accumulating feedback. The narrower test here concerns the value of knowledge when the acquisition provides little discrimination. [Reasoning BO](https://arxiv.org/abs/2505.12833)
