"""Strict data-only hypothesis parsing; no generated functions or expressions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    bounds: tuple[float, float] | None = None
    levels: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or not self.unit or (self.bounds is None) == (not self.levels):
            raise ValueError("specify either numerical bounds or categorical levels")
        if self.bounds is not None:
            if len(self.bounds) != 2 or not np.isfinite(self.bounds).all() or self.bounds[0] >= self.bounds[1]:
                raise ValueError("numeric bounds must be finite with lower < upper")
        elif len(set(self.levels)) != len(self.levels) or any(not isinstance(v, str) for v in self.levels):
            raise ValueError("categorical levels must be unique strings")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    version: int
    kind: str
    unit: str
    evidence_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    rationale: str
    parameter: str | None = None
    lower: float | None = None
    upper: float | None = None
    values: tuple[str, ...] = ()
    coefficients: tuple[tuple[str, float], ...] = ()
    rhs: float | None = None


def _number(value, name: str) -> float:
    if isinstance(value, (str, bool)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def parse_hypotheses(
    payload: Mapping,
    schema: Mapping[str, ParameterSpec],
    *,
    evidence_ids: set[str],
    observed_ids: set[str],
) -> tuple[Hypothesis, ...]:
    """Validate one complete version. Caller stores earlier versions unchanged.

    Evidence IDs must refer to the task's prevalidated evidence bundle. An ID
    check establishes provenance linkage, not that the statement is true.
    """
    if set(payload) != {"version", "hypotheses"}:
        raise ValueError("only version and hypotheses are allowed at batch level")
    version = payload["version"]
    if type(version) is not int or version < 1:
        raise ValueError("version must be a positive integer")
    rows = payload["hypotheses"]
    if not isinstance(rows, list) or len(rows) > 4:
        raise ValueError("hypotheses must be a list with at most four entries")
    common = {"id", "kind", "unit", "evidence_ids", "observation_ids", "rationale"}
    extras = {"interval": {"parameter", "lower", "upper"}, "allowed_values": {"parameter", "values"},
              "linear_le": {"coefficients", "rhs"}}
    result = []
    predicates = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in extras:
            raise ValueError("unsupported hypothesis kind")
        if not common <= set(row) or set(row) - common - extras[row["kind"]]:
            raise ValueError("missing required fields or unsupported hypothesis fields")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in {r.id for r in result}:
            raise ValueError("hypothesis IDs must be nonempty and unique")
        for key, allowed in (("evidence_ids", evidence_ids), ("observation_ids", observed_ids)):
            if not isinstance(row[key], list) or any(not isinstance(v, str) or v not in allowed for v in row[key]):
                raise ValueError(f"{key} contains an unknown source")
        if not row["evidence_ids"] or not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("each hypothesis needs provenance and a rationale")
        fields = dict(id=row["id"], version=version, kind=row["kind"], unit=row["unit"],
                      evidence_ids=tuple(row["evidence_ids"]), observation_ids=tuple(row["observation_ids"]),
                      rationale=row["rationale"])
        if row["kind"] in {"interval", "allowed_values"}:
            name = row.get("parameter")
            if name not in schema:
                raise ValueError("unknown parameter")
            spec = schema[name]
            if row["unit"] != spec.unit:
                raise ValueError("unit mismatch")
            fields["parameter"] = name
            if row["kind"] == "interval":
                if spec.bounds is None:
                    raise ValueError("interval requires a numerical parameter")
                lo = _number(row.get("lower", spec.bounds[0]), "lower")
                hi = _number(row.get("upper", spec.bounds[1]), "upper")
                if not spec.bounds[0] <= lo <= hi <= spec.bounds[1]:
                    raise ValueError("soft interval must lie inside the immutable hard domain")
                fields.update(lower=lo, upper=hi)
            else:
                values = row.get("values")
                if not isinstance(values, list) or not values or any(v not in spec.levels for v in values):
                    raise ValueError("allowed_values requires known categorical levels")
                fields["values"] = tuple(sorted(set(values)))
        else:
            coeff = row.get("coefficients")
            if row["unit"] != "normalized" or not isinstance(coeff, dict) or not coeff:
                raise ValueError("linear constraints require coefficients on normalized coordinates")
            if any(k not in schema or schema[k].bounds is None for k in coeff):
                raise ValueError("linear constraint references an unknown or categorical parameter")
            pairs = tuple(sorted((k, _number(v, "coefficient")) for k, v in coeff.items()))
            if not any(v != 0 for _, v in pairs):
                raise ValueError("linear constraint needs a nonzero coefficient")
            fields.update(coefficients=pairs, rhs=_number(row.get("rhs"), "rhs"))
        item = Hypothesis(**fields)
        predicate = (item.kind, item.unit, item.parameter, item.lower, item.upper,
                     item.values, item.coefficients, item.rhs)
        if predicate in predicates:
            raise ValueError("duplicate predicates must have their evidence merged before parsing")
        predicates.add(predicate)
        result.append(item)
    return tuple(result)


def compile_membership(
    hypotheses: Sequence[Hypothesis],
    candidates: Sequence[Mapping],
    schema: Mapping[str, ParameterSpec],
) -> np.ndarray:
    """Build once per pool/hypothesis version and reuse between decisions.

    'Compile' here means vectorize predicates into a Boolean matrix. This
    implementation uses NumPy and has no runtime JIT compilation.
    """
    columns = {}
    for row in candidates:
        if set(row) != set(schema):
            raise ValueError("every candidate must contain the fixed full parameter vector")
    for name, spec in schema.items():
        values = [row[name] for row in candidates]
        if spec.bounds is not None:
            col = np.asarray([_number(v, name) for v in values])
            if np.any((col < spec.bounds[0]) | (col > spec.bounds[1])):
                raise ValueError("candidate violates a hard parameter bound")
        else:
            if any(v not in spec.levels for v in values):
                raise ValueError("candidate has an unknown categorical value")
            col = np.asarray(values, dtype=object)
        columns[name] = col
    h = np.empty((len(hypotheses), len(candidates)), dtype=bool)
    for j, item in enumerate(hypotheses):
        if item.kind == "interval":
            h[j] = (columns[item.parameter] >= item.lower) & (columns[item.parameter] <= item.upper)
        elif item.kind == "allowed_values":
            h[j] = np.isin(columns[item.parameter], item.values)
        elif item.kind == "linear_le":
            lhs = np.zeros(len(candidates))
            for name, coefficient in item.coefficients:
                lo, hi = schema[name].bounds
                lhs += coefficient * (columns[name] - lo) / (hi - lo)
            h[j] = lhs <= item.rhs
        else:
            raise ValueError("unsupported hypothesis kind")
    h.setflags(write=False)
    return h


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


class FeatureEncoder:
    """Fixed full numerical coordinates and full categorical one-hot tables."""
    def __init__(self, schema: Mapping[str, ParameterSpec]):
        self.schema = dict(schema)
        if any(key != spec.name for key, spec in schema.items()):
            raise ValueError("parameter schema keys must equal ParameterSpec names")
        names = []
        for name, spec in self.schema.items():
            names.extend([name] if spec.bounds is not None else [f"{name}::{level}" for level in spec.levels])
        if len(names) != len(set(names)):
            raise ValueError("encoded feature names collide")
        self.names = tuple(names)
        self.param_space = {name: (0.0, 1.0) for name in self.names}

    def encode(self, candidates: Sequence[Mapping]) -> list[dict]:
        # Reuse the same full-vector and hard-bound validation as predicates.
        compile_membership((), candidates, self.schema)
        result = []
        for row in candidates:
            encoded = {}
            for name, spec in self.schema.items():
                if spec.bounds is not None:
                    lo, hi = spec.bounds
                    encoded[name] = (float(row[name]) - lo) / (hi - lo)
                else:
                    encoded.update({f"{name}::{level}": float(row[name] == level) for level in spec.levels})
            result.append(encoded)
        return result


class LLMResponsePending(Exception):
    """The interactive assistant has not yet supplied a particular response."""
    def __init__(self, request_path: str):
        super().__init__(request_path)
        self.request_path = request_path


class HypothesisManager:
    """Version ledger and bounded refresh, independent of a model provider.

    A provider returns {batch, candidate_ids, provider}. Pending interactive
    requests do not count as completed model attempts; returned errors and
    malformed responses do. Earlier response records remain immutable copies.
    """
    def __init__(self, schema, candidates, candidate_ids, evidence_ids, *,
                 max_attempts=4, refresh_after=(0, 4, 8, 12)):
        self.schema = dict(schema)
        self.candidates = tuple(dict(row) for row in candidates)
        self.candidate_ids = frozenset(candidate_ids)
        self.evidence_ids = frozenset(evidence_ids)
        self.max_attempts = max_attempts
        self.refresh_after = tuple(refresh_after)
        if type(max_attempts) is not int or not 0 <= max_attempts <= 4:
            raise ValueError("model attempt budget must be an integer from zero to four")
        if (len(set(self.refresh_after)) != len(self.refresh_after)
                or any(type(n) is not int or n < 0 for n in self.refresh_after)):
            raise ValueError("refresh times must be unique nonnegative query counts")
        self._payload = {"version": 1, "hypotheses": []}
        self._events = []
        self._queue = []
        self._membership = compile_membership((), self.candidates, self.schema)

    @property
    def hypotheses(self):
        return tuple(row["id"] for row in self._payload["hypotheses"])

    @property
    def membership(self):
        return self._membership

    @property
    def attempts(self):
        return len(self._events)

    @property
    def events(self):
        return tuple(deepcopy(self._events))

    def install_fixed(self, payload, *, observed_ids=()):
        hypotheses = parse_hypotheses(payload, self.schema, evidence_ids=set(self.evidence_ids),
                                      observed_ids=set(observed_ids))
        matrix = compile_membership(hypotheses, self.candidates, self.schema)
        self._payload = deepcopy(payload)
        self._membership = matrix

    def refresh(self, completed_additional: int, context: dict, provider) -> bool:
        if (provider is None or completed_additional not in self.refresh_after
                or self.attempts >= self.max_attempts
                or any(e["completed_additional"] == completed_additional for e in self._events)):
            return False
        request = deepcopy(context)
        request.update(expected_version=self.attempts + 1, previous_batch=deepcopy(self._payload))
        observed = {row["candidate_id"] for row in request["observations"]}
        event = {"attempt": self.attempts + 1, "completed_additional": completed_additional,
                 "context_sha256": canonical_hash(request), "observed_ids": sorted(observed)}
        try:
            response = provider(deepcopy(request))
            event["raw_response"] = deepcopy(response)
            if not isinstance(response, dict) or set(response) != {"batch", "candidate_ids", "provider"}:
                raise ValueError("provider response must contain only batch, candidate_ids and provider")
            if response["batch"].get("version") != request["expected_version"]:
                raise ValueError("response version does not match the reserved attempt")
            hypotheses = parse_hypotheses(response["batch"], self.schema,
                                          evidence_ids=set(self.evidence_ids), observed_ids=observed)
            queue = response["candidate_ids"]
            if (not isinstance(queue, list) or len(queue) > 4 or len(set(queue)) != len(queue)
                    or any(not isinstance(key, str) or key not in self.candidate_ids for key in queue)):
                raise ValueError("suggestion queue must contain at most four distinct pool IDs")
            matrix = compile_membership(hypotheses, self.candidates, self.schema)
            self._payload = deepcopy(response["batch"])
            self._queue = list(queue)
            self._membership = matrix
            event.update(status="accepted", batch_sha256=canonical_hash(self._payload))
        except LLMResponsePending:
            raise
        except Exception as exc:
            event.update(status="failed_retaining_previous", error=f"{type(exc).__name__}: {exc}")
        self._events.append(event)
        return event["status"] == "accepted"

    def pop_suggestion(self, queried_ids):
        while self._queue:
            key = self._queue.pop(0)
            if key not in queried_ids:
                return key
        return None

    def state_dict(self):
        return deepcopy({"payload": self._payload, "events": self._events, "queue": self._queue})

    def restore(self, state, *, observed_ids):
        events = deepcopy(state["events"])
        if len(events) > self.max_attempts or len({e["completed_additional"] for e in events}) != len(events):
            raise ValueError("saved model ledger exceeds the budget or repeats refresh times")
        self.install_fixed(state["payload"], observed_ids=observed_ids)
        queue = list(state["queue"])
        if any(key not in self.candidate_ids for key in queue):
            raise ValueError("saved suggestion queue contains a pool-external candidate")
        self._events, self._queue = events, queue
