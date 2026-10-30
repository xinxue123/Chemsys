"""Resumable paired Suzuki study, with actual assistant responses as external data."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from chem.decision.replay import write_json_atomic
from . import suzuki_domain, suzuki_policies
from .suzuki_domain import (DOMAIN, digest, initial_design, validate_point, validate_region,
                            same_point, FrozenSuzukiOracle)
from .suzuki_policies import MixedGP, available_grid, justify_select, lgbo_select, hebo_select

METHODS = ("LLINBO", "LGBO", "HEBO")
ROOT = Path("data/runs/suzuki_baselines")
RESEARCH = Path("research/suzuki_baselines")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def engine_files():
    return {Path(module.__file__).name: sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (suzuki_domain, suzuki_policies, __import__(__name__, fromlist=["_"]))}


def installed_sources():
    dist = metadata.distribution("HEBO")
    files = {str(p): sha256(Path(dist.locate_file(p)).read_bytes()).hexdigest()
             for p in dist.files if str(p).startswith("hebo/") and str(p).endswith(".py")}
    return {"distribution": "HEBO", "version": dist.version, "python_sources_sha256": files}


def prepare(root=ROOT, research=RESEARCH):
    root, research = Path(root), Path(research)
    if (root / "protocol.json").exists():
        return check_protocol(root)
    qualification = read_json(root / "oracles/qualification.json")
    if qualification["status"] != "PASS" or len(qualification["cases"]) != 4:
        raise ValueError("all four continuous oracles must qualify before freezing")
    protocol = {
        "schema_version": 1, "study": "Suzuki_three_baselines_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Only pretrained Reizman Suzuki Cases 1-4; exploratory emulator optimization",
        "cases": [1, 2, 3, 4], "seeds": [0, 1, 2, 3, 4], "methods": list(METHODS),
        "initial_queries": 4, "additional_queries": 16, "domain": DOMAIN,
        "objective": "maximize Summit pretrained ensemble mean yld/100; no added noise; not a wet experiment",
        "initialization": "paired seeded 3D Latin hypercube; four distinct randomly permuted catalysts",
        "llm_refresh": "every additional query (16 responses per LLM trajectory)",
        "llm": {"provider": "current Codex assistant selected by user", "configured_model_alias": "gpt-6.1-sol",
                "configured_reasoning_effort": "max", "immutable_backend_snapshot": None,
                "isolated_conversations": False, "token_usage_available": False,
                "contamination_disclosure": "Shared conversation already contains historical Suzuki benchmark work; not a blind independent model evaluation",
                "output_origin": "actual assistant-authored answers; never a scripted proposer",
                "inputs": "qualitative chemistry bundle and this trajectory's revealed observations only; no simulator tensors or unqueried outcomes in requests"},
        "gp": {"implementation": "sklearn GaussianProcessRegressor", "input": "3 normalized numeric + 8 full one-hot catalyst coordinates",
               "kernel": "Constant(1,1e-3..1e3)*Matern(nu=2.5,ARD11,length=1,bounds=.01..100)",
               "normalize_y": True, "alpha": 1e-6, "hyperparameter_restarts": 5},
        "LLINBO": {"variant": "Justify", "ucb": "mu+sqrt(beta_argument)*sigma",
                    "beta_argument": "2*log(t*4*pi^2/0.1*6), exactly the public BBFO Justify code schedule",
                    "psi": "maximum initial posterior variance over fixed mixed 4096-point Sobol search / t",
                    "gate": "accept iff max_searched_UCB - LLM_UCB <= psi; include LLM point in searched set",
                    "paper_code_difference": "Public code uses initial variance (not standard deviation) and the beta expression above; use code convention and record units",
                    "acquisition_search": "fresh 512 Sobol points per catalyst per iteration plus actual LLM proposal; no global optimum claim",
                    "adaptations": ["shared non-LLM initialization for paired comparison", "mixed-category one-hot sklearn GP", "finite Sobol approximation to continuous acquisition maximization"]},
        "LGBO": {"variant": "paper regional exponential tilt", "mean": "mu(X)+lambda*Cov_post(X,G)*a",
                 "covariance": "unchanged", "lambda": "confidence/sqrt(max(a^T Cov_post(G,G) a,1e-12))",
                 "grid": "512 valid Sobol points inside numeric region, equal mass per selected categorical level",
                 "acquisition": "exact Gaussian log EI for q=1; same maximizer as exact q=1 EI",
                 "acquisition_search": "same 4096-point mixed Sobol search as LLINBO (without a point proposal)",
                 "paper_code_difference": "Use paper direct c normalization; public repo instead maps confidence through guidance_scale*Phi^-1(c) and uses MC qLogEI",
                 "adaptations": ["shared non-LLM initialization", "valid categorical product extension to continuous-only public interface", "q=1 analytic rather than Monte Carlo EI"],
                 "theory_boundary": "No guarantee for this fully adaptive LLM process; paper theory studies a fixed lift"},
        "HEBO": {"implementation": "official pip HEBO==0.3.6, unmodified", "model": "gp", "rand_sample": 4,
                 "acquisition": "official MACE", "evolution": "official NSGA2 mixed-variable selection, population100 generations100",
                 "query_sign": "minimize -yld/100", "seed": "deterministic per case/seed/iteration for exact resume",
                 "adaptations": ["shared initial 4 observations (rand_sample=4 instead of domain-dependent default)"]},
        "evaluation": {"primary": "mean best-so-far normalized yield over 16 additional-budget positions (SearchAUC)",
                       "secondary": ["final best yield", "queries to >=80% and >=90% absolute yield", "LLM accept rate / guidance changes", "wall time"],
                       "regret": "not reported: continuous-domain global optimum is unknown",
                       "unit_of_pairing": "same case and seed, not individual queries",
                       "inference": "5 seeds/case; descriptive paired differences and seed uncertainty; no superiority from software tests",
                       "failed_runs": "retain full planned denominator and carry last incumbent across missing budget positions; separately count failures"},
        "resources": {"workers": 4, "blas_threads_per_worker": 1, "torch_threads_per_worker": 1,
                      "device": "CPU", "jit": "none; prebuilt binary dependencies and one worker warm-up", "operator_compiles_in_loop": 0},
        "engine_files_sha256": engine_files(), "hebo_sources": installed_sources(),
        "evidence_bundle_sha256": sha256(Path("research/chapter5/suzuki_evidence.json").read_bytes()).hexdigest(),
        "oracle_qualification_sha256": sha256((root / "oracles/qualification.json").read_bytes()).hexdigest(),
        "oracle_files_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                for p in sorted((root / "oracles").rglob("*")) if p.is_file()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()}}
    }
    manifest = [{"run_id": f"c{c}_s{s}_{m}", "case": c, "seed": s, "method": m,
                 "initial_points": initial_design(c, s)} for c in protocol["cases"] for s in protocol["seeds"] for m in METHODS]
    protocol["manifest_sha256"] = digest(manifest)
    protocol["protocol_sha256"] = digest(protocol)
    write_json_atomic(root / "protocol.json", protocol)
    write_json_atomic(root / "manifest.json", manifest)
    write_json_atomic(research / "protocol.json", protocol)
    write_json_atomic(research / "manifest.json", manifest)
    return protocol


def check_protocol(root):
    p = read_json(Path(root) / "protocol.json")
    copied = deepcopy(p)
    saved = copied.pop("protocol_sha256")
    if digest(copied) != saved or p["engine_files_sha256"] != engine_files():
        raise ValueError("frozen protocol or algorithm source changed")
    if digest(read_json(Path(root) / "manifest.json")) != p["manifest_sha256"]:
        raise ValueError("planned study population changed")
    if sha256(Path("research/chapter5/suzuki_evidence.json").read_bytes()).hexdigest() != p["evidence_bundle_sha256"]:
        raise ValueError("frozen chemistry evidence changed")
    return p


def request_for(spec, state, protocol):
    previous = state.get("responses", [])
    request = {"schema_version": 1, "run_id": spec["run_id"], "case": spec["case"], "seed": spec["seed"],
               "method": spec["method"], "iteration": len(state["history"]) - 3,
               "domain": DOMAIN, "objective": "maximize yield in percent", "history": state["history"],
               "previous_response": previous[-1] if previous else None,
               "instructions": ("Propose one new valid point, not previously observed. GPR will accept or reject it using its UCB gap."
                                if spec["method"] == "LLINBO" else
                                "Specify a preferred subset of catalyst labels, intervals for all 3 numeric variables, and confidence [0,1]. The regional GP mean tilt is soft; the full domain stays available. May explicitly retain previous preference."),
               "evidence": read_json("research/chapter5/suzuki_evidence.json")["evidence"],
               "response_fields": ["run_id", "iteration", "context_sha256", "reasoning", "supporting_queries",
                                   *( ["point"] if spec["method"] == "LLINBO" else ["region", "confidence"])],
               "protocol_sha256": protocol["protocol_sha256"]}
    request["context_sha256"] = digest(request)
    return request


def validate_response(response, request):
    if any(response.get(k) != request[k] for k in ("run_id", "iteration", "context_sha256")):
        raise ValueError("response must be bound to this exact pending trajectory context")
    if not isinstance(response.get("reasoning"), str) or len(response["reasoning"].strip()) < 8:
        raise ValueError("assistant response requires an explicit rationale")
    support = response.get("supporting_queries")
    history = request["history"]
    if not isinstance(support, list) or not support or any(type(v) is not int or not 1 <= v <= len(history) for v in support):
        raise ValueError("citations must refer to queries revealed in this trajectory")
    answer = deepcopy(response)
    if request["method"] == "LLINBO":
        answer["point"] = validate_point(answer["point"])
        if any(same_point(answer["point"], r["point"]) for r in history):
            raise ValueError("LLM proposed a previously queried point")
    else:
        if answer.get("retain_previous") is True:
            previous = request["previous_response"]
            if previous is None:
                raise ValueError("no previous preference to retain")
            answer["region"], answer["confidence"] = deepcopy(previous["region"]), previous["confidence"]
        validate_region(answer["region"])
        c = answer["confidence"]
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not np.isfinite(c) or not 0 <= c <= 1:
            raise ValueError("LGBO confidence must be finite and in [0,1]")
    answer["provider"] = {"origin": "current_assistant", "model_alias": "gpt-6.1-sol", "reasoning_effort": "max",
                          "immutable_backend_snapshot": None, "shared_context": True}
    return answer


def _worker_init():
    threadpool_limits(limits=1)


def run_one(root, spec, protocol):
    root = Path(root)
    run_dir = root / "runs" / spec["run_id"]
    state_file = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = read_json(state_file) if state_file.exists() else {
        **spec, "protocol_sha256": protocol["protocol_sha256"], "history": [], "decisions": [],
        "responses": [], "status": "CREATED", "optimization_seconds": 0., "oracle_seconds": 0.}
    if state["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("resume protocol mismatch")
    if state["status"] in ("COMPLETED", "FAILED"):
        return {"run_id": spec["run_id"], "status": state["status"], "queries": len(state["history"])}
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        oracle = FrozenSuzukiOracle(root / "oracles", spec["case"])
        # Warm numeric binary kernels once before the optimization loop. No JIT calls.
        _ = np.linalg.cholesky(np.eye(4))
        if spec["method"] == "HEBO":
            import torch
            torch.set_num_threads(1)
            _ = torch.linalg.cholesky(torch.eye(4))
        if not state["history"]:
            now = time.perf_counter()
            y = oracle.predict(spec["initial_points"])
            state["oracle_seconds"] += time.perf_counter() - now
            state["history"] = [{"query": i + 1, "point": row, "yield_percent": float(v * 100), "initial": True}
                                for i, (row, v) in enumerate(zip(spec["initial_points"], y))]
            write_json_atomic(state_file, state)
        try:
            while len(state["history"]) < 20:
                t = len(state["history"]) - 3
                rows = [r["point"] for r in state["history"]]
                values = [r["yield_percent"] / 100 for r in state["history"]]
                response = None
                if spec["method"] != "HEBO":
                    request = request_for(spec, state, protocol)
                    request_dir = root / "requests" / request["context_sha256"]
                    write_json_atomic(request_dir / "request.json", request)
                    if not (request_dir / "response.json").exists():
                        state.update(status="WAITING_LLM", pending_context=request["context_sha256"])
                        break
                    response = validate_response(read_json(request_dir / "response.json"), request)
                now = time.perf_counter()
                numerical_seed = 20261026 + spec["case"] * 10000 + spec["seed"] * 100 + t
                if spec["method"] == "HEBO":
                    point, decision = hebo_select(rows, values, numerical_seed)
                else:
                    gp = MixedGP(rows, values, numerical_seed)
                    candidates = available_grid(spec["case"], spec["seed"], t, rows)
                    if spec["method"] == "LLINBO":
                        if "psi0" not in state:
                            _, sigma0 = gp.moments(available_grid(spec["case"], spec["seed"], 0, rows))
                            state["psi0"] = float(np.max(sigma0 ** 2))
                        point, decision = justify_select(gp, candidates, response["point"], t=t, psi0=state["psi0"])
                    else:
                        point, decision = lgbo_select(gp, candidates, response, best=max(values), grid_seed=numerical_seed)
                    decision["kernel"] = str(gp.model.kernel_)
                    decision["log_marginal_likelihood"] = float(gp.model.log_marginal_likelihood_value_)
                state["optimization_seconds"] += time.perf_counter() - now
                point = validate_point(point)
                if any(same_point(point, r) for r in rows):
                    raise ValueError("optimizer selected a duplicate query")
                now = time.perf_counter()
                value = float(oracle.predict([point])[0])
                state["oracle_seconds"] += time.perf_counter() - now
                decision.update(iteration=t, query=len(rows) + 1, point=point,
                                engine_files_sha256=protocol["engine_files_sha256"], operator_compiles_in_loop=0)
                if response is not None:
                    decision["response_sha256"] = digest(response)
                    state["responses"].append(response)
                state["decisions"].append(decision)
                state["history"].append({"query": len(rows) + 1, "point": point, "yield_percent": value * 100, "initial": False})
                state.pop("pending_context", None)
                state["status"] = "RUNNING"
                write_json_atomic(state_file, state)
            else:
                state["status"] = "COMPLETED"
        except Exception as exc:
            state.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        state["last_invocation_seconds"] = time.perf_counter() - started
        state["resources"] = {"operator_compiles_in_loop": 0, "device": "CPU", "threadpools": threadpool_info()}
        write_json_atomic(state_file, state)
    return {"run_id": spec["run_id"], "status": state["status"], "queries": len(state["history"]), "error": state.get("error")}


def run(root=ROOT, workers=4, methods=METHODS):
    root = Path(root)
    p = check_protocol(root)
    manifest = [s for s in read_json(root / "manifest.json") if s["method"] in methods]
    if workers < 1 or workers > p["resources"]["workers"]:
        raise ValueError("workers must stay within frozen 1..4 resource limit")
    output = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        futures = [executor.submit(run_one, root, spec, p) for spec in manifest]
        for f in as_completed(futures):
            output.append(f.result())
    result = {"runs": sorted(output, key=lambda r: r["run_id"]),
              "status_counts": {s: sum(r["status"] == s for r in output) for s in ("COMPLETED", "WAITING_LLM", "FAILED")}}
    write_json_atomic(root / "status.json", result)
    return result


def pending(root=ROOT, compact=False):
    root = Path(root)
    output = []
    for spec in read_json(root / "manifest.json"):
        path = root / "runs" / spec["run_id"] / "state.json"
        if not path.exists():
            continue
        state = read_json(path)
        if state["status"] == "WAITING_LLM":
            request = read_json(root / "requests" / state["pending_context"] / "request.json")
            if compact:
                # Observed outcomes only; no model predictions or benchmark solution.
                request = {k: request[k] for k in ("run_id", "iteration", "context_sha256") } | {
                    "history": [[r["query"], r["point"]["catalyst"], *[round(r["point"][n], 4) for n in suzuki_domain.NUMERIC],
                                 round(r["yield_percent"], 4)] for r in request["history"]]}
            output.append(request)
    return output


def submit(path, root=ROOT):
    root = Path(root)
    submitted = read_json(path)
    results = []
    for response in submitted["responses"]:
        request = read_json(root / "requests" / response["context_sha256"] / "request.json")
        answer = validate_response(response, request)
        target = root / "requests" / response["context_sha256"] / "response.json"
        if target.exists() and read_json(target) != answer:
            raise ValueError("never overwrite an existing assistant answer")
        write_json_atomic(target, answer)
        results.append({"run_id": answer["run_id"], "iteration": answer["iteration"], "response_sha256": digest(answer)})
    receipt = {"answers_file": str(path), "answers_file_sha256": sha256(Path(path).read_bytes()).hexdigest(), "submitted": results}
    write_json_atomic(Path(path).with_suffix(".receipt.json"), receipt)
    return {"submitted": len(results)}
