"""Exact joint diagnosis of up to four soft constraints on a finite pool.

Only acquisition scores and constraint membership enter the policy. Future
objective values must stay in the replay/evaluation layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence
from itertools import combinations

import numpy as np


class NoAvailableCandidates(ValueError):
    """The hard-feasible, unqueried candidate pool is exhausted."""


@dataclass(frozen=True)
class Decision:
    candidate_index: int
    candidate_id: str
    retained_ids: tuple[str, ...]
    revoked_ids: tuple[str, ...]
    global_ucb: float
    selected_ucb: float
    opportunity_loss: float
    budget: float
    available_count: int
    eligible_count: int
    conflict_kind: str
    minimal_conflict_sets: tuple[tuple[str, ...], ...]
    subset_checks: tuple[dict, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _mask(value, n: int, name: str, default: bool) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=bool)
    result = np.asarray(value)
    if result.shape != (n,) or result.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean vector of length {n}")
    return result


def select_candidate(
    ucb: Sequence[float],
    membership: np.ndarray,
    budget: float,
    *,
    candidate_ids: Sequence[str],
    hypothesis_ids: Sequence[str],
    hard_mask=None,
    queried_mask=None,
) -> Decision:
    """Maximize retained cardinality, then retained IDs, then UCB.

    Ties between equally large retained subsets prefer the lexicographically
    smallest sorted ID tuple. Candidate UCB ties prefer the smallest ID.
    These rules are fixed before any outcomes are examined. Revocation is a
    decision for this round, not a claim that a chemical statement is false.
    """
    scores = np.asarray(ucb, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("ucb must be a finite one-dimensional vector")
    n = len(scores)
    ids = tuple(candidate_ids)
    hids = tuple(hypothesis_ids)
    if len(ids) != n or len(set(ids)) != n or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("candidate_ids must be nonempty unique strings")
    if len(hids) > 4 or len(set(hids)) != len(hids) or any(not isinstance(x, str) or not x for x in hids):
        raise ValueError("hypothesis_ids must be unique strings, with at most four entries")
    h = np.asarray(membership)
    if h.dtype != np.bool_ or h.shape != (len(hids), n):
        raise ValueError("membership must be a boolean [hypotheses, candidates] matrix")
    if isinstance(budget, (bool, np.bool_)) or not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and nonnegative")
    available = _mask(hard_mask, n, "hard_mask", True) & ~_mask(queried_mask, n, "queried_mask", False)
    if not available.any():
        raise NoAvailableCandidates("no hard-feasible, unqueried candidates remain")
    best = float(scores[available].max())
    losses = best - scores
    # Test the actual subtraction used in the returned receipt. No hidden
    # tolerance expands the promised opportunity budget.
    eligible = available & (losses <= budget)
    checks = []
    masks = []
    for bits in range(1 << len(hids)):
        indices = [j for j in range(len(hids)) if bits & (1 << j)]
        feasible = available & (h[indices].all(axis=0) if indices else True)
        kept = tuple(sorted(hids[j] for j in indices))
        in_budget = feasible & eligible
        masks.append(in_budget)
        checks.append({
            "bits": bits, "retained_ids": kept,
            "feasible_count": int(feasible.sum()),
            "eligible_count": int(in_budget.sum()),
            "max_ucb": float(scores[feasible].max()) if feasible.any() else None,
        })
    feasible_subsets = [i for i, row in enumerate(checks) if row["eligible_count"]]
    winner = min(feasible_subsets, key=lambda i: (-len(checks[i]["retained_ids"]), checks[i]["retained_ids"]))
    candidates = np.flatnonzero(masks[winner])
    selected = min(candidates, key=lambda i: (-float(scores[i]), ids[i]))
    retained = checks[winner]["retained_ids"]
    full = checks[-1]
    if full["eligible_count"]:
        conflict = "none"
    elif full["feasible_count"]:
        conflict = "quality_exclusion"
    else:
        conflict = "empty_intersection"
    minimal_conflicts = []
    for bits in range(1, len(checks)):
        if checks[bits]["eligible_count"]:
            continue
        # Feasibility is monotone under deleting a constraint, so checking
        # immediate proper subsets establishes inclusion minimality.
        parents = [bits ^ (1 << j) for j in range(len(hids)) if bits & (1 << j)]
        if all(checks[p]["eligible_count"] for p in parents):
            minimal_conflicts.append(checks[bits]["retained_ids"])
    return Decision(
        candidate_index=int(selected), candidate_id=ids[selected],
        retained_ids=retained, revoked_ids=tuple(sorted(set(hids) - set(retained))),
        global_ucb=best, selected_ucb=float(scores[selected]),
        opportunity_loss=float(losses[selected]), budget=float(budget),
        available_count=int(available.sum()), eligible_count=int(eligible.sum()),
        conflict_kind=conflict,
        minimal_conflict_sets=tuple(sorted(minimal_conflicts)),
        subset_checks=tuple(checks),
    )


METHODS = ("B0", "B1", "B2", "B3", "B4", "A1", "A2", "A3")


def select_policy(method, ucb, membership, budget, *, candidate_ids, hypothesis_ids,
                  queried_mask=None, hard_mask=None, round_number=1, rng=None,
                  suggested_candidate_id=None, escape_rounds=(4, 8, 12, 16)):
    """Matched finite-pool policies. No objective values enter this function.

    A2 diagnoses the entire conjunction and revokes all on conflict; B3 only
    validates an externally proposed point. Their decision rules are distinct.
    A1 and A3 deliberately need not obey the opportunity budget.
    """
    if method not in METHODS:
        raise ValueError(f"unknown policy {method}")
    scores = np.asarray(ucb, dtype=float)
    n, hids, ids = len(scores), tuple(hypothesis_ids), tuple(candidate_ids)
    h = np.asarray(membership)
    if h.shape != (len(hids), n) or h.dtype != np.bool_ or len(hids) > 4:
        raise ValueError("membership must have at most four boolean hypothesis rows")
    if len(set(hids)) != len(hids):
        raise ValueError("hypothesis IDs must be unique")
    base = select_candidate(scores, np.empty((0, n), dtype=bool), budget,
                            candidate_ids=ids, hypothesis_ids=(),
                            queried_mask=queried_mask, hard_mask=hard_mask)
    available = _mask(hard_mask, n, "hard_mask", True) & ~_mask(queried_mask, n, "queried_mask", False)
    eligible = available & (base.global_ucb - scores <= budget)
    all_indices = tuple(range(len(hids)))

    def subset_mask(indices):
        return available & (h[list(indices)].all(axis=0) if indices else True)

    def best_index(mask):
        indices = np.flatnonzero(mask)
        return min(indices, key=lambda i: (-float(scores[i]), ids[i])) if len(indices) else None

    kept, reason, detail = all_indices, "constrained_ucb", None
    if method == "B4":
        detail = select_candidate(scores, h, budget, candidate_ids=ids, hypothesis_ids=hids,
                                  queried_mask=queried_mask, hard_mask=hard_mask).to_dict()
        index = detail['candidate_index']
        kept = tuple(j for j, key in enumerate(hids) if key in detail['retained_ids'])
        reason = "joint_minimum_revocation"
    elif method == "B0":
        index, kept, reason = base.candidate_index, (), "full_pool_ucb"
    elif method == "B2" and round_number in escape_rounds:
        if rng is None:
            raise ValueError("fixed random exploration requires an explicit RNG")
        index = int(rng.choice(np.flatnonzero(available)))
        kept, reason = (), "scheduled_uniform_escape"
    elif method in {"B1", "B2"}:
        index = best_index(subset_mask(all_indices))
    elif method == "B3":
        index = None
        if suggested_candidate_id in ids:
            candidate = ids.index(suggested_candidate_id)
            if eligible[candidate]:
                index = candidate
        if index is None:
            index, reason = base.candidate_index, "single_candidate_rejected_or_missing"
        else:
            reason = "single_candidate_accepted"
        # B3 does not select a subset; satisfied advice is diagnostic only.
        kept = ()
    elif method == "A1":
        kept = tuple(j for j in all_indices if (h[j] & eligible).any())
        index = best_index(subset_mask(kept))
        reason = "individual_checks_only"
    elif method == "A2":
        feasible = subset_mask(all_indices)
        if (feasible & eligible).any():
            index = best_index(feasible)
            reason = "full_conjunction_accepted"
        else:
            index, kept, reason = base.candidate_index, (), "joint_conflict_revoke_all"
    else:  # A3: same cardinality/ID optimization, with only nonemptiness.
        feasible_groups = []
        for size in range(len(hids) + 1):
            for group in combinations(all_indices, size):
                if subset_mask(group).any():
                    feasible_groups.append(group)
        kept = min(feasible_groups, key=lambda group: (-len(group), tuple(sorted(hids[j] for j in group))))
        index = best_index(subset_mask(kept))
        reason = "nonempty_intersection_without_budget"
    retained = tuple(sorted(hids[j] for j in kept))
    if index is None:
        return {"method": method, "status": "STOPPED", "reason": "empty_soft_intersection",
                "candidate_index": None, "candidate_id": None, "retained_ids": retained,
                "revoked_ids": tuple(sorted(set(hids) - set(retained))),
                "global_ucb": base.global_ucb, "budget": float(budget),
                "opportunity_loss": None, "budget_violation": False}
    loss = float(base.global_ucb - scores[index])
    return {"method": method, "status": "SELECTED", "reason": reason,
            "hypothesis_subset_decision": method not in {'B0', 'B3'},
            "candidate_index": int(index), "candidate_id": ids[index],
            "retained_ids": retained, "revoked_ids": tuple(sorted(set(hids) - set(retained))),
            "satisfied_ids": tuple(sorted(hids[j] for j in all_indices if h[j, index])),
            "global_ucb": base.global_ucb, "selected_ucb": float(scores[index]),
            "budget": float(budget), "opportunity_loss": loss,
            "budget_violation": loss > budget, "available_count": int(available.sum()),
            "eligible_count": int(eligible.sum()), "guard_detail": detail}
