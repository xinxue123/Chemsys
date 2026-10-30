"""Unified command line interface.

Examples::

    python -m chem.cli doctor
    python -m chem.cli protocol CeAl2Ge2 -o /tmp/proto.json
    python -m chem.cli audit arc-melt --demo
    python -m chem.cli audit protocol /tmp/proto.json
    python -m chem.cli xrd estimate data/ce_al_ge/arc_melt/raw/CeAlGe_10-70-30.xy --system ceal2ge2
    python -m chem.cli xrd identify <file.dat>
    python -m chem.cli bo suggest --dataset data/ce_al_ge/datasets/arcmelt_dataset_rietveld.json
    python -m chem.cli loop ErTa7O19 --max-iter 2
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from . import paths


# ── helpers ────────────────────────────────────────────────────────────────
def _dump(obj, out: str | None = None) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"written: {out}")
    else:
        print(text)


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_module_main(module_name: str, argv: list[str]) -> int:
    """Call ``<module>.main()`` with a patched sys.argv (for legacy CLIs)."""
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "main"):
        print(f"{module_name} has no main(); import it instead", file=sys.stderr)
        return 2
    old = sys.argv
    sys.argv = [module_name] + argv
    try:
        mod.main()
    finally:
        sys.argv = old
    return 0


# ── doctor ─────────────────────────────────────────────────────────────────
def cmd_doctor(args) -> int:
    print("chemist_system — environment check\n")
    print(f"python           {sys.version.split()[0]}  {sys.executable}")
    print(f"repo root        {paths.HOME}")
    ok = True

    print("\nresolved paths:")
    for key, value in paths.describe().items():
        exists = Path(value).exists()
        print(f"  {'ok ' if exists else '-- '} {key:18s} {value}")

    print("\ndependencies:")
    for mod, required in (("numpy", True), ("scipy", True), ("pymatgen", True),
                          ("rxn_ca", False), ("rxn_network", False),
                          ("torch", False), ("cantera", False)):
        present = _has(mod)
        tag = "ok " if present else ("MISSING" if required else "optional")
        print(f"  {tag:8s} {mod}")
        if required and not present:
            ok = False

    print("\nexternal toolchains:")
    from . import config
    print(f"  MP key           {'configured' if config.has_mp_key() else 'NOT configured'}")
    print(f"  GSAS-II home     {paths.GSAS2_HOME} ({'present' if config.gsas2_available() else 'missing'})")
    print(f"  gsas2 conda env  {paths.GSAS2_CONDA_ENV} ({paths.GSAS2_PYTHON or 'not found'})")
    print(f"  rxn python       {paths.RXN_PYTHON}")
    print(f"  wiki             {paths.WIKI} ({'present' if paths.WIKI.is_dir() else 'missing'})")

    import subprocess
    probe = subprocess.run([sys.executable, "-c", "import mp_api.client"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        print("  mp_api import    ok")
    else:
        err = (probe.stderr.strip().splitlines() or [""])[-1]
        print(f"  mp_api import    FAILS in this interpreter: {err[:110]}")
        print("                   fix: use ./scripts/chem (sets LD_LIBRARY_PATH) "
              "or `conda activate` the env")

    print("\nce_al_ge data:")
    for sub in ("arc_melt/raw", "arc_melt/cifs", "datasets", "energetics"):
        d = paths.CE_AL_GE / sub
        n = len(list(d.glob("*"))) if d.is_dir() else 0
        print(f"  {n:4d} files  {d}")
    eng = paths.RIETVELD_ENGINE / "iter_agent.py"
    print(f"  production agent  {eng} ({'present' if eng.exists() else 'absent → packaged agent used'})")
    return 0 if ok else 1


# ── protocol / audit ───────────────────────────────────────────────────────
def cmd_protocol(args) -> int:
    from .protocol import generate_protocol
    proto = generate_protocol(args.material, initial_temp=args.temp,
                              route=("arc_melt" if args.arc_melt else None))
    _dump(proto, args.output)
    return 0


def cmd_audit(args) -> int:
    from .audit.gate import PhaseGraphAuditor, load_api_key

    if args.mode == "arc-melt":
        if args.demo or not args.file:
            from .protocol import generate_protocol, to_arc_recipe
            material = args.demo or "CeAl2Ge2"
            recipe = to_arc_recipe(generate_protocol(material))
            print(f"[audit] demo recipe for {material}")
        else:
            recipe = _load_json(args.file)
        auditor = PhaseGraphAuditor(api_key=load_api_key())
        result = auditor.audit_arc_melt(recipe)
        if args.json:
            _dump(result, args.output)
        else:
            print(f"status: {result.get('status')}   token: {result.get('security_token')}")
            for chk in result.get("checks", []):
                print(f"  [{chk.get('status'):7s}] {chk.get('check_id'):16s} {chk.get('message')}")
            for hz in result.get("hazard_summary", []) or []:
                print(f"  ! {hz}")
        return 0 if result.get("status") in {"APPROVED", "WARNING"} else 1

    # oxide / solid-state protocol
    proto = _load_json(args.file)
    try:
        auditor = PhaseGraphAuditor(api_key=load_api_key(required=True))
    except RuntimeError as exc:
        print(f"cannot audit an oxide protocol: {exc}", file=sys.stderr)
        print("note: arc-melt recipes need no key — try `chem audit arc-melt --demo`",
              file=sys.stderr)
        return 2
    result = auditor.audit_protocol(proto)
    if args.json:
        _dump(result, args.output)
    else:
        print(f"status: {result.get('status')}")
        print(result.get("report", ""))
        if result.get("audit_flags"):
            print(f"flags: {result['audit_flags']}")
    return 0 if (result.get("execution_status") == "OK"
                 and result.get("review_status") in {"APPROVED", "WARNING"}) else 1


# ── xrd ────────────────────────────────────────────────────────────────────
def cmd_xrd(args) -> int:
    if args.action == "estimate":
        from .xrd import create_estimator, SYSTEM_CONFIGS
        if args.system not in SYSTEM_CONFIGS:
            print(f"unknown system {args.system!r}; configured: {list(SYSTEM_CONFIGS)}",
                  file=sys.stderr)
            return 2
        estimator = create_estimator(args.system, backend=args.backend or "auto")
        result = estimator.estimate(args.file)
        meta = result.phase_peaks_used or {}
        if args.json:
            _dump({
                "method": result.method,
                "yield_pct": result.yield_pct,
                "sigma_pct": result.sigma_pct,
                "confidence": result.confidence,
                "confidence_note": result.confidence_note,
                "phase_fractions": result.phase_fractions,
                "phase_fractions_sigma": result.phase_fractions_sigma,
                "status": meta.get("status"),
                "use_for_bo": meta.get("use_for_bo"),
                "no_fallback": meta.get("no_fallback"),
                "metrics": meta.get("metrics"),
                "report_path": meta.get("report_path"),
            }, args.output)
        else:
            print(f"method:     {result.method}")
            value = "unknown" if result.yield_pct is None else f"{result.yield_pct:.2f}% ± {result.sigma_pct:.2f}%"
            print(f"yield:      {value}")
            print(f"confidence: {result.confidence}  (use_for_bo={meta.get('use_for_bo')})")
            if meta.get("metrics"):
                print(f"metrics:    {meta['metrics']}")
            for phase, wt in (result.phase_fractions or {}).items():
                print(f"  {phase:12s} {wt * 100:6.2f} wt%")
            if result.confidence_note:
                print(f"note:       {result.confidence_note}")
        return 0

    if args.action == "identify":
        return _run_module_main("chem.xrd.analyzer", ["scan", args.file])

    if args.action == "seed":
        from .xrd import analyzer
        analyzer.seed_common_phases()
        conn = analyzer.init_db()
        rows = list(conn.execute("select count(*) from phases"))[0][0]
        print(f"reference DB seeded: {rows} phases -> {paths.XRD_DB}")
        return 0

    if args.action == "db":
        from .xrd import analyzer
        analyzer.list_phases(analyzer.init_db())
        return 0

    if args.action == "batch":
        from .xrd.gsas2.batch import main as batch_main
        summary = batch_main(args.dir or paths.ARC_RAW, system=args.system,
                             out_dir=args.output)
        print(f"batch done: n_ok={summary['n_ok']} n_error={summary['n_error']}")
        return 0
    return 2


# ── bo ─────────────────────────────────────────────────────────────────────
def _observations_from_dataset(path: str) -> list[dict]:
    from .decision.replay import load_cealge_observations
    return load_cealge_observations(path)


def cmd_bo(args) -> int:
    from .bo.llm import LLMDrivenBO

    if args.action == "bench":
        from .bo.rietveld import benchmark
        result = benchmark(reps=args.reps)
        _dump(result, args.output)
        return 0

    observations = (json.loads(Path(args.observations).read_text(encoding="utf-8")) if args.observations
                    else _observations_from_dataset(args.dataset))
    if not observations:
        print("no usable observations", file=sys.stderr)
        return 2

    material_data = not args.observations
    space = {"atom_fraction_Ce": (0.0, 1.0), "atom_fraction_Al": (0.0, 1.0)} if material_data else None
    simplex = (lambda x: x["atom_fraction_Ce"] > 0 and x["atom_fraction_Al"] > 0
               and x["atom_fraction_Ce"] + x["atom_fraction_Al"] < 1) if material_data else None
    bo = LLMDrivenBO(system_description=args.system_description, full_param_space=space,
                     candidate_filter=simplex, max_llm_calls=0 if material_data else 4)
    for obs in observations:
        bo.add_observation(float(obs["yield"]), yield_sigma=obs.get("yield_sigma"), **obs["params"])
    suggestion = bo.suggest(method=args.method)
    _dump({
        "n_observations": len(observations),
        "mode": getattr(bo, "_mode", None),
        "recommendation_status": "needs_validation" if material_data else "legacy_reference",
        "qualification": "material ratio provenance and processing conditions remain unverified" if material_data else None,
        "coordinate_basis": "nominal mass ratios converted to atom fractions" if material_data else None,
        "llm_calls": len(bo.llm_events),
        "suggestion": {
            "params": suggestion.params,
            "mu_pred": suggestion.mu_pred,
            "sigma_pred": suggestion.sigma_pred,
            "acq_method": suggestion.acq_method,
            "acq_value": suggestion.acq_value,
            "ci": [suggestion.ci_lo, suggestion.ci_hi],
            "interval_calibration": "not_established",
        },
    }, args.output)
    return 0


# ── energetics ─────────────────────────────────────────────────────────────
def cmd_energetics(args) -> int:
    if args.action == "hull":
        from .energetics import hull
        out = {
            "assemblage": hull.assemblage(args.ce, args.al),
            "selectivity": hull.selectivity(args.ce, args.al),
            "competitors": hull.competitors(args.ce, args.al),
        }
        _dump(out, args.output)
        return 0
    if args.action == "enthalpy":
        from .energetics import miedema
        _dump({"formula": args.formula,
               "H_form_kJ_per_mol_atom": miedema.formation_enthalpy(args.formula)},
              args.output)
        return 0
    return 2


# ── screening ──────────────────────────────────────────────────────────────
def cmd_screen(args) -> int:
    mod = {"magnetocaloric": "chem.screening.magnetocaloric",
           "furnace-xrd": "chem.screening.furnace_xrd",
           "fetch-shortlist": "chem.screening.fetch_shortlist",
           "fetch-furnace-shortlist": "chem.screening.fetch_furnace_shortlist"}[args.action]
    return _run_module_main(mod, [])


# ── loop ───────────────────────────────────────────────────────────────────
def cmd_loop(args) -> int:
    from .loop import full_loop
    result = full_loop(
        material=args.material,
        max_audit_iterations=args.max_iter,
        run_simulation=args.simulate,
        force_regenerate=args.force_regen,
        xrd_file=args.xrd,
        size_override=args.size,
    )
    _dump(result, args.output)
    return 0 if result.get("success") else 1


def cmd_decision(args) -> int:
    from .decision.evaluation import (freeze_math_test, prepare_math_suite, run_data_audit,
                                      prepare_summit_suite, run_math_suite, run_mechanism_suite, suite_status)

    if args.action == "mechanism":
        result = run_mechanism_suite(random_states=args.random_states, seed=args.seed)
    elif args.action == "data-audit":
        result = run_data_audit(args.dataset_dir)
    elif args.action == 'report':
        from .decision.reporting import audit_suite, summarize_audit
        if not args.suite_dir:
            raise ValueError('--suite-dir is required for an independent report')
        result = summarize_audit(audit_suite(args.suite_dir))
    elif args.action == 'material-demo':
        from .decision.material_adapter import material_interface_demo
        result = material_interface_demo(paths.HOME)
    elif args.action == 'verify':
        from .decision.delivery import verify_delivery
        result = verify_delivery(paths.HOME)
    elif args.action == "summit-cache":
        from .decision.replay import cache_summit_case
        root = Path(args.suite_dir) if args.suite_dir else paths.RUNS / 'chapter5' / 'summit_cache' / f'case_{args.case}'
        evidence = _load_json(args.evidence)['evidence']
        result = cache_summit_case(root, args.case, pool_seed=args.seed + args.case * 1000, evidence=evidence)
    elif args.action.startswith('assistant-'):
        from .decision.assistant_suite import (assistant_status, freeze_assistant_test,
                                               prepare_assistant_suite, request_view,
                                               run_assistant_suite, submit_answers)
        split = 'test' if args.action == 'assistant-freeze' else args.split
        root = Path(args.suite_dir) if args.suite_dir else paths.RUNS / 'chapter5' / f'assistant_{split}'
        if args.action == 'assistant-prepare':
            result = prepare_assistant_suite(root, _load_json(args.protocol), args.oracle_cache_dir,
                                             args.model_metadata, args.prompt_file, split=split)
        elif args.action == 'assistant-run':
            result = run_assistant_suite(root, workers=args.workers)
        elif args.action == 'assistant-freeze':
            dev = Path(args.development_dir) if args.development_dir else paths.RUNS / 'chapter5' / 'assistant_development'
            result = freeze_assistant_test(root, dev)
        elif args.action == 'assistant-submit':
            if not args.answers_file:
                raise ValueError('--answers-file is required')
            result = submit_answers(root, args.answers_file)
        elif args.action == 'assistant-pending':
            result = [request_view(_load_json(path)) for path in assistant_status(root)['pending_requests']]
        else:
            result = assistant_status(root)
    else:
        benchmark = 'summit' if args.action.startswith('summit-') else 'math'
        split = 'test' if args.action.endswith('-freeze') else args.split
        root = Path(args.suite_dir) if args.suite_dir else paths.RUNS / 'chapter5' / f'{benchmark}_{split}'
        if args.action == 'summit-prepare':
            result = prepare_summit_suite(root, _load_json(args.protocol), args.oracle_cache_dir, split=split)
        elif args.action == 'math-prepare':
            result = prepare_math_suite(root, _load_json(args.protocol), split=split)
        elif args.action.endswith('-run'):
            result = run_math_suite(root, workers=args.workers, max_groups=args.max_groups)
        elif args.action.endswith('-freeze'):
            dev = Path(args.development_dir) if args.development_dir else paths.RUNS / 'chapter5' / f'{benchmark}_development'
            result = freeze_math_test(root, dev)
        else:
            result = suite_status(root)
    _dump(result, args.output)
    return 0


# ── parser ─────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chem", description="Autonomous materials chemist")
    p.add_argument("--version", action="version", version="chem " + __import__("chem").__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check environment, paths and data").set_defaults(func=cmd_doctor)

    pp = sub.add_parser("protocol", help="generate a synthesis protocol from stoichiometry")
    pp.add_argument("material")
    pp.add_argument("--temp", type=float, default=None)
    pp.add_argument("--arc-melt", action="store_true")
    pp.add_argument("-o", "--output")
    pp.set_defaults(func=cmd_protocol)

    pa = sub.add_parser("audit", help="audit a protocol or an arc-melt recipe")
    pa.add_argument("mode", choices=["protocol", "arc-melt"])
    pa.add_argument("file", nargs="?")
    pa.add_argument("--demo", nargs="?", const="CeAl2Ge2", default=None,
                    help="build a demo recipe (arc-melt mode)")
    pa.add_argument("--json", action="store_true")
    pa.add_argument("-o", "--output")
    pa.set_defaults(func=cmd_audit)

    px = sub.add_parser("xrd", help="XRD quantification / identification")
    px.add_argument("action", choices=["estimate", "identify", "batch", "seed", "db"])
    px.add_argument("file", nargs="?")
    px.add_argument("--system", default="ceal2ge2")
    px.add_argument("--backend", default=None,
                    choices=[None, "auto", "pawlik", "peak", "gsas2", "gsas2_legacy"])
    px.add_argument("--dir", default=None, help="raw dir for batch")
    px.add_argument("--json", action="store_true")
    px.add_argument("-o", "--output")
    px.set_defaults(func=cmd_xrd)

    pb = sub.add_parser("bo", help="Bayesian optimization suggestions / benchmark")
    pb.add_argument("action", choices=["suggest", "bench"])
    pb.add_argument("--dataset", default=str(paths.ce_al_ge_dataset()))
    pb.add_argument("--observations", default=None)
    pb.add_argument("--system-description", default="CeAl2Ge2 Ce-Al-Ge intermetallic arc-melt")
    pb.add_argument("--method", default="EI", choices=["EI", "UCB"])
    pb.add_argument("--reps", type=int, default=300)
    pb.add_argument("-o", "--output")
    pb.set_defaults(func=cmd_bo)

    pd = sub.add_parser("decision", help="Chapter 5 reference mechanism checks / material input audit")
    pd.add_argument("action", choices=["mechanism", "data-audit", "material-demo", "report", "verify", "math-prepare", "math-run", "math-freeze", "math-status", "summit-cache", "summit-prepare", "summit-run", "summit-freeze", "summit-status", "assistant-prepare", "assistant-run", "assistant-freeze", "assistant-submit", "assistant-pending", "assistant-status"])
    pd.add_argument("--random-states", type=int, default=250)
    pd.add_argument("--seed", type=int, default=20261026)
    pd.add_argument("--dataset-dir", default=str(paths.CE_AL_GE / "datasets"))
    pd.add_argument("--protocol", default=str(paths.HOME / 'research' / 'chapter5' / 'protocol.json'))
    pd.add_argument("--split", choices=['development', 'test'], default='development')
    pd.add_argument("--suite-dir")
    pd.add_argument("--development-dir")
    pd.add_argument("--workers", type=int, default=4)
    pd.add_argument("--max-groups", type=int)
    pd.add_argument("--case", type=int, choices=[1, 2, 3, 4], default=1)
    pd.add_argument("--evidence", default=str(paths.HOME / 'research' / 'chapter5' / 'suzuki_evidence.json'))
    pd.add_argument("--model-metadata", default=str(paths.HOME / 'research' / 'chapter5' / 'assistant_provider.json'))
    pd.add_argument("--prompt-file", default=str(paths.HOME / 'research' / 'chapter5' / 'hypothesis_prompt.txt'))
    pd.add_argument("--answers-file")
    pd.add_argument("--oracle-cache-dir", default=str(paths.RUNS / 'chapter5' / 'summit_cache'))
    pd.add_argument("-o", "--output")
    pd.set_defaults(func=cmd_decision)

    pe = sub.add_parser("energetics", help="offline formation enthalpy / hull")
    pe.add_argument("action", choices=["hull", "enthalpy"])
    pe.add_argument("--ce", type=float, default=0.25)
    pe.add_argument("--al", type=float, default=0.40)
    pe.add_argument("--formula", default="CeAl2Ge2")
    pe.add_argument("-o", "--output")
    pe.set_defaults(func=cmd_energetics)

    ps = sub.add_parser("screen", help="Materials Project screening scripts")
    ps.add_argument("action", choices=["magnetocaloric", "furnace-xrd",
                                       "fetch-shortlist", "fetch-furnace-shortlist"])
    ps.set_defaults(func=cmd_screen)

    pl = sub.add_parser("loop", help="run the full knowledge→audit→plan loop")
    pl.add_argument("material")
    pl.add_argument("--max-iter", type=int, default=5)
    pl.add_argument("--simulate", action="store_true")
    pl.add_argument("--force-regen", action="store_true")
    pl.add_argument("--xrd", default=None)
    pl.add_argument("--size", type=int, default=None)
    pl.add_argument("-o", "--output")
    pl.set_defaults(func=cmd_loop)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
