#!/usr/bin/env python3
"""
Autonomous Materials Chemist — Core Loop
=========================================
Phase 1: Knowledge base (wiki) lookup
         → Literature search (S2 API + arXiv fallback)
         → Protocol generation
Phase 2: Audit iteration (PASS or max_iterations)
Phase 3: rn simulation (thermal evolution)
Phase 4: Execution plan (ready for XRD feedback)

Usage:
    python -m chem.cli loop ZrNCl
    python -m chem.cli loop BaTiO3 --max-iter 5
    python -m chem.cli loop ErTa7O19 --simulate
"""

import json
import re
import sys
import os
import subprocess
import tempfile
import urllib.request
import urllib.parse
import time
from datetime import datetime
from pathlib import Path

# ── paths (single source of truth: chem/paths.py) ─────────────────────────
from . import paths
from .config import mp_api_key

WIKI_PATH      = paths.WIKI
DELIVERABLES   = paths.DELIVERABLES
PYTHON12       = paths.RXN_PYTHON
RXN_CA_PYTHON  = paths.RXN_PYTHON
RXN_CA_SRC     = paths.RXN_CA_SRC
PROTOCOL_TOOLS = os.environ.get("CHEMIST_PROTOCOL_TOOLS", "")  # optional external generator
WIKI_CACHE     = None   # lazy singleton WikiCache (built on first use)
paths.ensure(DELIVERABLES, WIKI_PATH)

# ── Hindsight endpoint (optional memory service; failures are silent) ─────
HINDSIGHT_URL  = os.environ.get(
    "CHEMIST_HINDSIGHT_URL",
    "http://127.0.0.1:8888/v1/default/banks/hermes/memories",
)
HINDSIGHT_AUTO = os.environ.get("CHEMIST_HINDSIGHT", "false").lower() == "true"

# ── S2 API ──────────────────────────────────────────────────────────────────
# Key stored in hermes secrets; fall back to env or blank (free tier has 1 req/s limit)
S2_API_KEY = os.environ.get("S2_API_KEY", "")

# ────────────────────────────────────────────────────────────────────────────
# Layer 1 checkpoint — atomic JSON write after every phase
# ────────────────────────────────────────────────────────────────────────────

def _slug(material: str) -> str:
    """Sanitize material formula into a safe filename slug."""
    return re.sub(r'[^A-Za-z0-9]', '_', material)

def checkpoint_write(material: str, phase: str, data: dict, *,
                    best_protocol: dict = None) -> Path:
    """
    Write phase result to Layer 1 (disk JSON) under:
        ~/.hermes/deliverables/{slug}_{phase}.json

    Also writes the master deliverable if best_protocol is provided.
    Hindsight write is controlled by AUTO_HINDSIGHT env (default: true).
    """
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    slug = _slug(material)

    # Phase-level checkpoint
    cp_path = DELIVERABLES / f"{slug}_{phase}.json"
    cp_data = {
        "material": material,
        "phase": phase,
        "checkpointed_at": datetime.now().isoformat(),
        "data": data,
    }
    tmp = cp_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(cp_data, ensure_ascii=False, indent=2))
    tmp.replace(cp_path)   # atomic on POSIX

    # Master deliverable: always accumulate into one file
    master = DELIVERABLES / f"{slug}_synthesis_plan.json"
    master_data = {}
    if master.exists():
        try:
            master_data = json.loads(master.read_text())
        except Exception:
            pass
    master_data.setdefault("material", material)
    master_data.setdefault("checkpoints", {})
    master_data["checkpoints"][phase] = {
        "checkpointed_at": datetime.now().isoformat(),
        "data": data,
    }
    master_data["_last_phase"] = phase
    if best_protocol is not None:
        master_data["best_protocol"] = best_protocol
    tmp_m = master.with_suffix('.tmp')
    tmp_m.write_text(json.dumps(master_data, ensure_ascii=False, indent=2))
    tmp_m.replace(master)

    return cp_path


def hindsight_write(material: str, phase: str, data: dict, *,
                    tags: list = None, confidence: float = None) -> bool:
    """
    Write phase result to Hindsight via HTTP API (Layer 3).
    Returns True on success, False on failure.
    """
    if not HINDSIGHT_AUTO:
        return False

    body_parts = [f"phase={phase}", f"material={material}"]
    if confidence is not None:
        body_parts.append(f"confidence={confidence:.3f}")
    body_parts.append(json.dumps(data, ensure_ascii=False)[:500])  # truncate for safety

    payload = json.dumps({
        "items": [{
            "content": " | ".join(body_parts),
            "context": f"chem.loop: {phase}",
            "tags": (tags or []) + ["autonomous-chemist", phase],
        }],
        "async": False,
    }).encode()

    try:
        req = urllib.request.Request(
            HINDSIGHT_URL, data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 202)
    except Exception:
        return False


def checkpoint_and_hindsight(material: str, phase: str, data: dict, *,
                             best_protocol: dict = None,
                             htags: list = None,
                             hconfidence: float = None) -> Path:
    """
    Convenience: write to both Layer 1 and Layer 3 atomically.
    """
    cp_path = checkpoint_write(material, phase, data, best_protocol=best_protocol)
    hindsight_write(material, phase, data, tags=htags, confidence=hconfidence)
    return cp_path


# ────────────────────────────────────────────────────────────────────────────
# Phase 1a — Wiki lookup
# ────────────────────────────────────────────────────────────────────────────

def _parse_formula(text: str) -> dict[str, int]:
    """Parse a chemical formula string into element counts.
    Handles parenthesized groups with trailing multipliers: Er(NO3)3 → {Er:1, N:3, O:9}"""
    result = {}
    i = 0
    n = len(text)
    while i < n:
        # Element + optional count
        m = re.match(r'([A-Z][a-z]?)(\d*)\s*', text[i:])
        if m:
            elem, cnt = m.group(1), m.group(2)
            i += m.end()
            result[elem] = result.get(elem, 0) + (int(cnt) if cnt else 1)
        elif text[i] == '(':
            # Find matching ')' and parse the group
            depth = 0
            j = i
            while j < n:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            inner = text[i+1:j]
            coeff_m = re.match(r'(\d+)\s*', text[j+1:])
            coeff = int(coeff_m.group(1)) if coeff_m else 1
            inner_counts = _parse_formula(inner)
            for k, v in inner_counts.items():
                result[k] = result.get(k, 0) + v * coeff
            i = j + 1 + (coeff_m.end() if coeff_m else 0)
        else:
            i += 1
    return result


def _format_formula(comp: dict[str, int]) -> str:
    """Format element-count dict back into a formula string."""
    return ''.join(k + (str(v) if v > 1 else '') for k, v in sorted(comp.items()))


def _parse_precursor_block(block: str) -> list:
    """Parse precursor block strings into canonical formulas.
    Handles: Er2O3 + Ta2O5, Er(NO3)3 + Ta-ethoxide, etc.
    Takes the first ' or ' route with ≥2 precursors (multi-element route)."""
    # Normalize: remove LaTeX \text{} wrappers, $, -, and _ tokens
    block = re.sub(r'\\text\{([^}]+)\}', r'\1', block)
    block = block.replace('$', '').replace('-', '').replace('_', '')

    # Split on ' or ' to get candidate routes
    or_parts = re.split(r'\s+or\s+', block)

    def extract_formula(seg: str) -> str:
        seg = seg.strip().rstrip('.')
        if not seg:
            return ''
        comp = _parse_formula(seg)
        return _format_formula(comp)

    # Try each route; pick the first that has ≥2 precursors
    for route in or_parts:
        route = route.strip()
        if not route:
            continue
        segs = [s.strip() for s in route.split(' + ')] if ' + ' in route else [route]
        result = []
        for seg in segs:
            f = extract_formula(seg)
            if f and len(f) > 1:
                result.append({'chemical': f, 'molar_ratio': 1.0})
        if len(result) >= 2:
            return result

    # Fallback: return first route's first precursor
    route0 = or_parts[0].strip()
    f = extract_formula(route0)
    return [{'chemical': f, 'molar_ratio': 1.0}] if f else []


# -- Wiki access (optimized with WikiCache) -----------------------------------
def _get_wiki_cache():
    """Lazy singleton WikiCache. Built once on first use, ~2ms."""
    global WIKI_CACHE
    if WIKI_CACHE is None:
        from .knowledge.wiki import WikiCache
        WIKI_CACHE = WikiCache(WIKI_PATH)
    return WIKI_CACHE


def wiki_lookup(material: str) -> dict | None:
    """
    Optimized wiki lookup using WikiCache with pre-built index.

    Returns synthesis protocol dict or None if material not in wiki.
    First call: ~3ms (index build + lookup). Subsequent: <0.1ms.
    """
    cache = _get_wiki_cache()
    protocol = cache.get_synthesis_protocol(material)
    if protocol is not None:
        return protocol
    return None


def build_default_thermal_profile(target_temp: float):
    """Build a realistic thermal profile matching experimental solid-state synthesis.

    Uses real experimental parameters:
      - 5°C/min ramp (typical muffle furnace)
      - 6.5h dwell @ peak T (verified for ErTa7O19 @ 1200°C in wiki)
      - Intermediate 800°C dwell for precursor decomposition
      - Air atmosphere
    """
    return [
        {"stage": 1, "action": "ramp_up",   "target_temperature_c": 100, "ramp_rate_c_per_min": 5,  "dwell_time_min": 0,   "description": "Dry sample"},
        {"stage": 2, "action": "dwell",     "target_temperature_c": 100, "ramp_rate_c_per_min": 0,  "dwell_time_min": 60,  "description": "Remove adsorbed water"},
        {"stage": 3, "action": "ramp_up",   "target_temperature_c": min(800, target_temp * 0.6), "ramp_rate_c_per_min": 5, "dwell_time_min": 0,   "description": "Intermediate decomposition"},
        {"stage": 4, "action": "dwell",     "target_temperature_c": min(800, target_temp * 0.6), "ramp_rate_c_per_min": 0,  "dwell_time_min": 90,  "description": "Precursor decomposition"},
        {"stage": 5, "action": "ramp_up",   "target_temperature_c": target_temp, "ramp_rate_c_per_min": 5, "dwell_time_min": 0,   "description": "Ramp to sintering"},
        # Real experimental: 6.5h @ 1200°C for ErTa7O19 (from wiki experimental records)
        {"stage": 6, "action": "dwell",     "target_temperature_c": target_temp, "ramp_rate_c_per_min": 0,  "dwell_time_min": 390, "description": "Main crystallization (6.5h)"},
        {"stage": 7, "action": "ramp_down", "target_temperature_c": 25,  "ramp_rate_c_per_min": -5, "dwell_time_min": 0,   "description": "Cool to room temperature"},
    ]


def infer_temperature_from_class(material: str) -> float:
    """Infer typical sintering temperature from material composition."""
    formula = material.lower()
    if any(e in formula for e in ["la", "er", "yb", "lu", "gd", "dy", "ho", "tm"]):  # lanthanides
        return 1100
    if any(e in formula for e in ["ta", "nb", "w", "mo"]):  # high-valent metals
        return 1200
    if "ti" in formula and "ba" in formula:  # titanate perovskite
        return 1100
    if "al" in formula:
        return 1200
    if "li" in formula and any(e in formula for e in ["fe", "co", "ni", "mn"]):  # Li-ion cathode
        return 700
    return 950  # default


# ────────────────────────────────────────────────────────────────────────────
# Phase 1b — Literature search (Semantic Scholar)
# ────────────────────────────────────────────────────────────────────────────

def search_semantic_scholar(query: str, limit: int = 8) -> list:
    """Search Semantic Scholar Graph API. Returns list of paper dicts."""
    api_key = S2_API_KEY
    if not api_key:
        secrets_path = Path.home() / ".hermes" / "secrets.yaml"
        if secrets_path.exists():
            import yaml
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f)
                api_key = secrets.get("semantic_scholar_api_key", "")

    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": "title,authors,abstract,year,venue,citationCount,openAccessPdf,externalIds"
    })
    
    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("data", [])
    except Exception as e:
        print(f"  [S2 search warning] {e}")
        return []


def extract_synthesis_params_from_abstract(abstract: str | None) -> dict:
    """Extract synthesis temperature, precursors, atmosphere from abstract text.

    Semantic Scholar returns ``abstract: null`` for some records; coerce to "" so
    literature parsing degrades to "no parameters found" instead of crashing.
    """
    abstract = abstract or ""
    result = {"temperature_c": None, "precursors": [], "atmosphere": None, "raw": abstract[:300]}
    
    # Temperature patterns
    t_matches = re.findall(r'(\d{3,4})\s*°?C|(?:sintered?|heated?|annealed?|calcined?)\s+at\s+(\d{3,4})', abstract, re.IGNORECASE)
    for m in t_matches:
        t = next((x for x in m if x), None)
        if t:
            result["temperature_c"] = float(t)
            break
    
    # Precursor patterns (common chemicals)
    precursor_patterns = [
        r'Er2O3', r'Yb2O3', r'La2O3', r'BaCO3', r'BaO', r'TiO2', r'Ta2O5',
        r'ZrCl4', r'ZrN', r'NH4Cl', r'Li2CO3', r'Al2O3', r'Fe2O3',
    ]
    for pp in precursor_patterns:
        if re.search(pp, abstract):
            result["precursors"].append(pp)
    
    # Atmosphere
    if re.search(r'\b(Ar|argon|nitrogen|N2|air|O2|vacuum)\b', abstract, re.IGNORECASE):
        m = re.search(r'\b(Ar|argon|nitrogen|N2|air|O2|vacuum)\b', abstract, re.IGNORECASE)
        result["atmosphere"] = m.group(1).capitalize() if m else "Air"
    
    return result


def literature_search(material: str) -> dict:
    """
    Search S2 for synthesis papers on the target material.
    Returns a dict with papers + best protocol extracted.
    """
    print(f"\n  🔍 [Literature] Searching Semantic Scholar for: {material}")
    
    # Build search queries — try multiple formulations
    queries = [
        f"{material} solid-state synthesis",
        f"{material} ceramic sintering synthesis",
        f"{material.replace('3', '')} oxide synthesis temperature",
    ]
    
    all_results = []
    for q in queries:
        results = search_semantic_scholar(q, limit=5)
        all_results.extend(results)
        time.sleep(1.1)  # rate limit
    
    # Deduplicate by title
    seen = set()
    unique_results = []
    for r in all_results:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_results.append(r)
    
    best_protocol = None
    best_conf = 0.0
    
    for paper in unique_results:
        params = extract_synthesis_params_from_abstract(paper.get("abstract", ""))
        conf = 0.0
        if params["temperature_c"]:
            conf += 0.4
        if params["precursors"]:
            conf += 0.3
        if params["atmosphere"]:
            conf += 0.1
        if paper.get("citationCount", 0) > 20:
            conf += 0.2
        
        if conf > best_conf and conf > 0.3:
            best_conf = conf
            target_t = params["temperature_c"] or infer_temperature_from_class(material)
            best_protocol = {
                "material": material,
                "confidence_score": round(conf, 2),
                "source": f"S2: {paper.get('title','')[:60]}... (citations={paper.get('citationCount',0)})",
                "protocol": {
                    "precursors": [{"chemical": p, "molar_ratio": 1.0, "purity": "99.9%"} for p in params["precursors"]],
                    "mechanical_processing": {"pressing_pressure_mpa": 200, "tooling": "uniaxial press"},
                    "thermal_profile": build_default_thermal_profile(target_t),
                    "atmosphere": params["atmosphere"] or "Air",
                    "crucible_material": "Alumina",
                },
                "warnings": [f"Extracted from abstract — verify in full text"],
            }
    
    return {
        "material": material,
        "papers_found": len(unique_results),
        "papers": [
            {
                "title": p["title"],
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "venue": p.get("venue", ""),
                "abstract_snippet": (p.get("abstract") or "")[:200],
                "doi": p.get("externalIds", {}).get("DOI", ""),
            }
            for p in unique_results[:6]
        ],
        "best_protocol": best_protocol,
    }


# ────────────────────────────────────────────────────────────────────────────
# Phase 1c — Protocol generation (via protocol_optimizer tools)
# ────────────────────────────────────────────────────────────────────────────

def generate_protocol(material: str, initial_temp: float = None) -> dict:
    """Generate a synthesis protocol.

    Uses the optional external generator named by CHEMIST_PROTOCOL_TOOLS if it
    exists, otherwise the built-in stoichiometry rules in chem.protocol.
    """
    if PROTOCOL_TOOLS and Path(PROTOCOL_TOOLS).exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("protocol_tools", PROTOCOL_TOOLS)
        pt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pt)
        protocol = pt.generate_generic_protocol(material, target_temp=initial_temp)
        protocol["source"] = f"Auto-generated (external: {PROTOCOL_TOOLS})"
        return protocol

    from .protocol import generate_protocol as _rules_protocol
    return _rules_protocol(material, initial_temp=initial_temp)


# ────────────────────────────────────────────────────────────────────────────
# Phase 2 — Audit loop (wrapping run_optimization)
# ────────────────────────────────────────────────────────────────────────────

def _run_audit_result(xrd_phases: dict, yield_pct: float,
                      protocol: dict) -> dict | None:
    """Posterior audit: XRD feedback → semantic priors, run in-process."""
    try:
        from .audit.gate import PhaseGraphAuditor, load_api_key
        auditor = PhaseGraphAuditor(api_key=load_api_key(required=True))
        return auditor.audit_result(xrd_phases, float(yield_pct), protocol)
    except Exception as exc:
        print(f"  [audit] posterior audit unavailable: {exc}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# rxn-ca 集成：Phase 3a — Cellular Automaton 反应模拟
# ────────────────────────────────────────────────────────────────────────────
# 数据流（基础版，无需声子/AIMD）：
#   MP API → get_entries_in_chemsys → rxn_network ReactionSet
#                                                   ↓
#   SolidPhaseSet (体积/密度/熔点 from pymatgen) → get_scored_rxns
#                                                   ↓
#   ReactionLibrary → HeatingSchedule → ReactionRecipe
#                                                   ↓
#   run_single_sim → CA 演化 → RxnCAResultDoc → phase_fraction
#
# 输出格式：
#   {
#     "status": "ok"|"error",
#     "recipe": {temperature_steps, size, packing_fraction},
#     "phases": [phase_name, ...],
#     "phase_fractions": {phase_name: weight_frac, ...},
#     "dominant_pathways": [rxn_string, ...],
#     "melt_events": [温度点, ...],
#     "error": str,
#   }
# ────────────────────────────────────────────────────────────────────────────

def _build_heating_schedule_from_protocol(protocol: dict):
    """
    从 synthesis protocol dict 提取温度程序。
    protocol 格式示例（来自 wiki）：
      {"thermal_profile": [{"stage": 1, "action": "dwell",
                           "target_temperature_c": 1200, "dwell_time_min": 240,
                           "ramp_rate_c_per_min": 3}, ...]}
    返回 HeatingSchedule（rxn-ca 内部格式），其 steps 属性是 List[HeatingStep]。
    """
    from rxn_ca.core.heating import HeatingSchedule, HeatingStep

    # ── Parse protocol (supports flat or thermal_profile format) ──────
    thermal_profile = protocol.get("thermal_profile") or []
    if not thermal_profile:
        inner = protocol.get("protocol", {})
        thermal_profile = inner.get("thermal_profile") or []
    if not thermal_profile:
        # Flat format: temperatures_c=[...], durations_min=[...] (one dwell per temp)
        # Example: temps=[25,100,800,1200], durations=[60,90,390,0]
        #   → ramp 25→100 (hold 60min), 100→800 (hold 90min), 800→1200 (hold 390min)
        # The LAST temperature typically has dwell=0 (cooling ramp handled separately)
        temps_c = protocol.get("temperatures_c", [])
        durations = protocol.get("durations_min", [])
        if temps_c and durations:
            ramp = protocol.get("ramp_rate_c_per_min", 5.0)
            thermal_profile = []
            prev_t = 25
            for t_c, dur_min in zip(temps_c, durations):
                if dur_min > 0:
                    thermal_profile.append({
                        "target_temperature_c": t_c,
                        "dwell_time_min": dur_min,
                        "ramp_rate_c_per_min": ramp,
                    })
                prev_t = t_c
            # Edge case: if ALL durations are 0 (e.g., from audit_loop flat output),
            # use the max temperature as the dwell step
            if not thermal_profile and temps_c:
                max_t = max(temps_c)
                thermal_profile.append({
                    "target_temperature_c": max_t,
                    "dwell_time_min": 240,
                    "ramp_rate_c_per_min": ramp,
                })

    # ── Build HeatingSchedule ─────────────────────────────────────
    steps_out: list[dict] = []
    for step in thermal_profile:
        t_c = float(step.get("target_temperature_c", 25))
        t_k = t_c + 273.15
        # GibbsEntry.get_new_temperature 支持范围：300-2000 K
        if t_k < 300:
            print(f"  [WARN] Temperature {t_k:.1f}K < 300K, skipping")
            continue
        if t_k > 2000:
            print(f"  [WARN] Temperature {t_k:.1f}K > 2000K, skipping")
            continue
        # 兼容两种格式：dwell_time_min（wiki）或 hold（内部格式）
        dwell_min = float(step.get("dwell_time_min") or step.get("hold", 0))
        duration_s = dwell_min * 60.0  # minutes → seconds
        # ramp_rate handled at Recipe level, not HeatingStep
        steps_out.append({
            "temperature": t_k,
            "duration": duration_s,
        })
    # 去重（保留顺序）
    seen: dict = {}
    for s in steps_out:
        key = round(s["temperature"], 10)
        if key not in seen:
            seen[key] = s

    # 转换为 rxn-ca HeatingSchedule(HeatingStep...) 格式
    heating_steps = [
        HeatingStep(temperature=s["temperature"], duration=s["duration"])
        for s in list(seen.values())
    ]
    return HeatingSchedule(heating_steps)


class _DeltaGScore:
    """基于 ΔG 的评分器：直接使用 DFT ΔG 作为反应驱动力。

    sigmoid(k=5)：ΔG = -0.1 eV → score=0.69, ΔG = -0.05 eV → score=0.58
    ΔG < 0（放热）→ 高分，ΔG > 0（吸热）→ 低分。
    相比 TammanScore（仅 T/T_m），此评分器注入实验验证的热力学驱动力。
    """
    def __new__(cls, phase_set, temp: int = None):
        import math
        from rxn_ca.reactions.scorers.core import BasicScore

        k = 5.0

        class DeltaGScore(BasicScore):
            def score(self, rxn):
                delta_g = rxn.energy_per_atom  # eV/atom
                return 1 / (1 + math.exp(k * delta_g))

        return DeltaGScore(phase_set=phase_set, temp=temp)


def _build_delta_g_scorer(phase_set, temp: int):
    return _DeltaGScore(phase_set=phase_set, temp=temp)


def _build_reaction_recipe(material: str,
                           precursor_formulas: list[str],
                           protocol: dict,
                           solid_phase_set,
                           entry_set=None,
                           size: int = 15) -> dict:
    """Build a ReactionRecipe and run rxn-ca CA simulation."""
    print(f"  [_build_reaction_recipe] ENTERED")
    print(f"  [_build_reaction_recipe] protocol.get('precursor_ratios'): {protocol.get('precursor_ratios', 'KEY_MISSING')}")
    precursor_ratios = protocol.get("precursor_ratios", {})
    from rxn_ca.core.heating import HeatingSchedule
    from rxn_ca.core.recipe import ReactionRecipe
    from rxn_ca.utilities.get_entries import get_entries
    from rxn_ca.utilities.setup_reaction import setup_noise_reaction
    from rxn_ca.utilities.single_sim import run_single_sim
    from rxn_ca.reactions.reaction_library import ReactionLibrary as RxnLib
    from rxn_ca.reactions.scorers import score_rxns
    from rxn_ca.reactions.scored_reaction_set import ScoredReactionSet
    from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator

    # ── 1. Parse protocol ─────────────────────────────────────────────
    # Accept two formats:
    #   (A) raw protocol dict with nested protocol.thermal_profile
    #   (B) flat protocol dict with temperatures_c + durations_min
    raw_protocol = protocol.get("protocol", protocol)  # unwrap if nested
    thermal_profile = raw_protocol.get("thermal_profile", [])
    if thermal_profile:
        temps_c = [s.get("target_temperature_c", s.get("temperature_c", 0)) for s in thermal_profile]
        durations_min = [s.get("duration_min", s.get("dwell_min", 0)) for s in thermal_profile]
    else:
        temps_c = raw_protocol.get("temperatures_c", protocol.get("temperatures_c", []))
        durations_min = raw_protocol.get("durations_min", protocol.get("durations_min", []))

    ramp_rate = raw_protocol.get("ramp_rate_c_per_min", protocol.get("ramp_rate_c_per_min", 5.0))
    precursor_ratios = raw_protocol.get("precursor_mole_ratios",
                                        protocol.get("precursor_mole_ratios",
                                            protocol.get("precursor_ratios", {})))
    print(f"  [_build_reaction_recipe] precursor_mole_ratios: {precursor_ratios}")
    size = raw_protocol.get("simulation_size", protocol.get("simulation_size", 15))
    packing = raw_protocol.get("packing_fraction", protocol.get("packing_fraction", 0.55))

    if not temps_c or not durations_min:
        return {"status": "error", "error": "Protocol missing temperatures_c or durations_min"}

    # ── 2. Heating schedule ──────────────────────────────────────────
    heating_steps = []
    prev_t = 25
    for t_c, dur_min in zip(temps_c, durations_min):
        ramp_min = abs(t_c - prev_t) / ramp_rate if ramp_rate > 0 else 0
        heating_steps.append({"target_temperature_c": t_c, "duration_min": dur_min, "ramp_rate_c_per_min": ramp_rate})
        prev_t = t_c

    # Build inner protocol for _build_heating_schedule_from_protocol
    inner_protocol = {
        "temperatures_c": temps_c, "durations_min": durations_min,
        "ramp_rate_c_per_min": ramp_rate,
        "precursor_mole_ratios": precursor_ratios,
        "precursor_formulas": precursor_formulas,
        "simulation_size": size, "packing_fraction": packing,
    }

    # Filter to valid (non-dwell-only ramp) temperature steps
    hs = _build_heating_schedule_from_protocol(inner_protocol)
    for s in hs.steps:
        print(f"    T={s.temperature}K, dur={s.duration}s")
    filtered_hs = HeatingSchedule(steps=[s for s in hs.steps if s.duration > 0])
    valid_temps = list(dict.fromkeys(round(s.temperature) for s in filtered_hs.steps))

    # ── 3. Use passed-in solid_phase_set, extract entries ───────────────
    # solid_phase_set and entry_set are passed from phase3a (already from MP API)
    # If not provided, fall back to MP API lookup
    if solid_phase_set is None:
        from pymatgen.ext.matproj import MPRester
        mpr = MPRester()
        precursor_names = precursor_formulas if precursor_formulas else _canonicalize(material)
        entries = get_entries(precursor_names, mpr)
        solid_phase_set = SolidPhaseSet(entries)
    else:
        # Use entries from solid_phase_set or entry_set
        if entry_set is not None and hasattr(entry_set, 'entries'):
            # entries can be dict-like {entry_id: entry} or list-like
            entries_src = entry_set.entries
            if isinstance(entries_src, dict):
                entries = list(entries_src.values())
            elif isinstance(entries_src, list):
                entries = entries_src
            else:
                entries = list(entries_src)
        else:
            entries = list(solid_phase_set.entries)

    # ── 4. Enumerate reactions ────────────────────────────────────────
    # Use BasicEnumerator with precursor/target constraints.
    # This only enumerates reactions involving these phases (fast: ~1 reaction).
    # Pass GibbsEntrySet (not list) so it resolves phase names.
    from rxn_network.enumerators.basic import BasicEnumerator
    from pymatgen.core import Composition

    precursor_names = [f for f in precursor_formulas if f] if precursor_formulas else []
    # Standardize material formula: "erta7o19" -> "ErTa7O19"
    # Use Composition to get reduced_formula, but first extract element+count from entries
    # Build a lookup from lowercase formula to canonical formula
    formula_canonical = {}
    for e in entries:
        rf = e.composition.reduced_formula
        formula_canonical[rf.lower()] = rf
    canonical_mat = formula_canonical.get(material.lower(), material)

    enumerator = BasicEnumerator(
        precursors=precursor_names,
        targets=[canonical_mat],
    )
    rxn_set = enumerator.enumerate(entry_set)
    all_rxns = list(rxn_set)
    print(f"    Enumerated {len(all_rxns)} reactions (precursors={precursor_names}, targets=[{material}])")

    # ── 5. Build ReactionLibrary with _DeltaGScore ───────────────────
    # For each temperature, convert entries to GibbsComputedStructureEntry,
    # recompute ΔG manually, and build ScoredReactionSet.
    class _DeltaGScore:
        def __init__(self, temp=None):
            self.temp = temp
        def score(self, rxn):
            import math
            return 1 / (1 + math.exp(5 * rxn.energy_per_atom))
        def __call__(self, rxn):
            return self.score(rxn)

    rxn_lib: RxnLib = RxnLib(phases=solid_phase_set)
    temps_int = [int(t) for t in valid_temps]

    for t in temps_int:
        T_K = float(t)
        # Build Gibbs entries for temperature correction
        from pymatgen.analysis.compatibility.computed_entries import GibbsComputedStructureEntry
        entry_dict = {e.composition.reduced_formula: e for e in entries}
        gibbs_entries = {}
        for formula, entry in entry_dict.items():
            try:
                gibbs_entries[formula] = GibbsComputedStructureEntry(
                    structure=entry.structure,
                    formation_enthalpy_per_atom=entry.energy_per_atom,
                    temp=T_K,
                    gibbs_model="SISSO",
                )
            except Exception:
                gibbs_entries[formula] = entry  # fallback

        # Recompute reaction energies at this temperature
        rxn_energies = {}  # rxn_id -> energy_per_rxn
        for r in all_rxns:
            try:
                # Build ComputedReaction with Gibbs entries
                from pymatgen.analysis.reaction_calculator import ComputedReaction
                rxn_gibbs = ComputedReaction(
                    [gibbs_entries.get(e.composition.reduced_formula, e) for e in r.reactants],
                    [gibbs_entries.get(e.composition.reduced_formula, e) for e in r.products],
                )
                rxn_energies[id(r)] = rxn_gibbs.calculated_reaction_energy_per_rxn
            except Exception:
                rxn_energies[id(r)] = r.energy_per_atom

        # Build ScoredReactionSet with temperature-corrected energies
        # ScoredReaction(reactants_dict, products_dict, competitiveness, energy_per_atom)
        # where dicts map {phase_name: stoich_coefficient}
        import math
        from rxn_ca.reactions.scored_reaction import ScoredReaction
        scored_rxns = []
        for r in all_rxns:
            e = rxn_energies.get(id(r), r.energy_per_atom)
            sigmoid = 1 / (1 + math.exp(5 * e))
            # Build stoichiometry dicts from rxn_network reaction
            # r is a BasicReaction: r.reactant_coeffs is {Composition: coefficient}
            def coeffs_to_dict(coeffs_dict):
                return {comp.reduced_formula: round(abs(coef), 4) for comp, coef in coeffs_dict.items()}
            scored_rxns.append(
                ScoredReaction(
                    coeffs_to_dict(r.reactant_coeffs),
                    coeffs_to_dict(r.product_coeffs),
                    sigmoid,
                    energy_per_atom=e,
                )
            )
        scored_rset = ScoredReactionSet(scored_rxns, solid_phase_set)
        rxn_lib.add_rxns_at_temp(scored_rset, t)

    if not filtered_hs.steps:
        return {
            "status": "error",
            "error": (
                f"No valid heating steps after filtering. "
                f"valid_temps={valid_temps}, "
                f"raw hs.steps={len(hs.steps)}, filtered={len(filtered_hs.steps)}. "
                f"Check temperatures_c and durations_min in protocol."
            ),
        }

    # ── 6. Build Recipe ───────────────────────────────────────────────
    recipe = ReactionRecipe(
        heating_schedule=filtered_hs,
        reactant_amounts=precursor_ratios,
        simulation_size=size,
        packing_fraction=packing,
    )

    initial_sim = setup_noise_reaction(
        solid_phase_set,
        precursor_mole_ratios=recipe.reactant_amounts,
        size=recipe.simulation_size,
        packing_fraction=recipe.packing_fraction,
    )
    if hasattr(initial_sim, 'structure'):
        from collections import Counter
        try:
            sites = initial_sim.structure.sites()
            species = [str(s.specie) for s in sites]
            print(f"  initial species: {dict(Counter(species))}")
        except Exception:
            pass  # species debug not critical

    # ── 6. Run CA ────────────────────────────────────────────────────
    try:
        result: "ReactionResult" = run_single_sim(
            recipe=recipe,
            reaction_lib=rxn_lib,
            initial_simulation=initial_sim,
            phase_set=solid_phase_set,
            compress_freq=10,
        )
    except Exception as ca_exc:
        import traceback
        return {
            "status": "error",
            "error": f"run_single_sim failed: {ca_exc}",
            "traceback": traceback.format_exc()[-3000:],
        }

    # ── 7. Extract phase fractions ───────────────────────────────────
    phase_fracs = _extract_phase_fractions(result)
    if "__error__" in phase_fracs:
        return {"status": "error", "error": phase_fracs["__error__"]}

    return {
        "status": "ok",
        "recipe": {
            "n_temperature_steps": len(valid_temps),
            "size": recipe.simulation_size,
            "packing_fraction": recipe.packing_fraction,
            "temperatures_K": valid_temps,
        },
        "phases": list(phase_fracs.keys()),
        "phase_fractions": phase_fracs,
    }



def _get_reaction_step_analyzer(result):
    """从 RxnCAResultDoc 获取 ReactionStepAnalyzer（如果可用）。"""
    try:
        from rxn_ca.analysis import ReactionStepAnalyzer
        # Primary: use result.phases (SolidPhaseSet on RxnCAResultDoc)
        if hasattr(result, 'phases') and result.phases is not None:
            return ReactionStepAnalyzer(phase_set=result.phases)
        # Fallback: try results[i].phase_set
        if hasattr(result, 'results') and result.results:
            for r in result.results:
                phase_set = getattr(r, 'phase_set', None)
                if phase_set is not None:
                    return ReactionStepAnalyzer(phase_set=phase_set)
    except Exception:
        pass
    return None


def _extract_phase_fractions(result) -> dict[str, float]:
    """
    从 run_single_sim 的 RxnCAResultDoc 中提取最终相重量分数。

    RxnCAResultDoc 结构：
      - results: List[ReactionResult] — 每个温度步骤一个
      - final_simulation: Simulation — 最后一步快照
        - Simulation.state: SimulationState (pylattica)
    备选路径：result.results[-1].last_step → SimulationState
    """
    # ── 1. 获取 SimulationState ───────────────────────────────────
    state = None
    # Priority 1: result.results[-1].last_step (SimulationResult → SimulationState)
    if hasattr(result, 'results') and result.results:
        last = result.results[-1]
        state = getattr(last, 'last_step', None)
    # Priority 2: result.final_simulation.state (Simulation → SimulationState)
    if state is None and hasattr(result, 'final_simulation') and result.final_simulation:
        sim = result.final_simulation
        state = getattr(sim, 'state', None)

    if state is None:
        return {"__error__": f"No SimulationState found in {type(result).__name__}"}

    # ── 2. ReactionStepAnalyzer（如果可用）────────────────────────────
    try:
        from rxn_ca.analysis import ReactionStepAnalyzer
        analyzer = _get_reaction_step_analyzer(result)
        if analyzer is not None:
            analyzer.set_step_group(state)
            fracs = analyzer.get_all_mole_fractions()
            if fracs:
                return {k: round(v, 4) for k, v in fracs.items() if v > 0.001}
    except Exception:
        pass  # analyzer unavailable — fall through to raw SITES

    # ── 3. 直接读 state.get_state()['SITES'] ───────────────────────
    try:
        raw = state.get_state()
        if isinstance(raw, dict) and 'SITES' in raw:
            sites_dict = raw['SITES']
            if isinstance(sites_dict, dict) and len(sites_dict) > 0:
                from collections import Counter
                occ_counts = Counter()
                for site_data in sites_dict.values():
                    if isinstance(site_data, dict) and 'DISCRETE_OCCUPANCY' in site_data:
                        occ_counts[site_data['DISCRETE_OCCUPANCY']] += 1
                total = sum(occ_counts.values())
                if total > 0:
                    id_to_name = _get_phase_id_to_name(result)
                    return {
                        id_to_name.get(occ, occ): round(count / total, 4)
                        for occ, count in sorted(occ_counts.items(), key=lambda x: -x[1])
                        if count / total > 0.001
                    }
    except Exception:
        pass

    return {"__error__": f"Could not extract phase fractions from {type(result).__name__}"}


def _get_phase_id_to_name(result) -> dict:
    """从 RxnCAResultDoc 追溯 phase_set，构建 phase_id → formula 映射。"""
    try:
        if hasattr(result, 'final_simulation') and result.final_simulation:
            phase_set = getattr(result.final_simulation, 'phase_set', None)
            if phase_set is not None:
                return {ph.phase_id: getattr(ph, 'formula', str(ph)) for ph in getattr(phase_set, 'phases', [])}
        if hasattr(result, 'results') and result.results:
            for r in result.results:
                phase_set = getattr(r, 'phase_set', None)
                if phase_set is not None:
                    return {ph.phase_id: getattr(ph, 'formula', str(ph)) for ph in getattr(phase_set, 'phases', [])}
    except Exception:
        pass
    return {}


def _build_solid_phase_set(material: str,
                            precursor_formulas: list[str],
                            precursor_ratios: dict | None = None) -> "SolidPhaseSet":
    """
    从 MP 构建 SolidPhaseSet（包含体积、密度、熔点），用于 rxn-ca。
    不需要额外 DFT 计算。

    数据流：
      get_entries(chem_sys="Er-Ta-O") → GibbsEntrySet
      SolidPhaseSet.from_entry_set(entry_set) → SolidPhaseSet
    """
    import sys as _sys
    _sys.path.insert(0, RXN_CA_SRC)

    from rxn_ca.utilities.get_entries import get_entries
    from rxn_ca.phases.solid_phase_set import SolidPhaseSet
    import os

    # 获取 MP API key（惰性：环境变量 / 密钥文件）
    key = mp_api_key(required=True)


    # 构建化学体系字符串（例如 "Er-Ta-O"）
    # 注意：material（erta7o19）是产物而非前驱体，只用 precursor_formulas 提取元素
    all_formulas = precursor_formulas
    from pymatgen.core import Composition
    elements_set = sorted(set(
        el.symbol
        for f in all_formulas
        for el in Composition(f).elements
    ))
    chemsys_str = "-".join(elements_set)  # "Er-Ta-O"

    # 获取 GibbsEntrySet（直接用 MPRester，绕过 rxn-ca get_entries 的 ThermoDoc 问题）
    from pymatgen.ext.matproj import MPRester
    from rxn_network.entries.entry_set import GibbsEntrySet

    key = mp_api_key(required=True)

    mpr = MPRester(key)
    mp_entries = mpr.get_entries_in_chemsys(
        elements=chemsys_str,
        additional_criteria={"thermo_types": ["GGA_GGA+U"]},
    )
    # get_densities_and_vols_from_entry_set uses MPRESTER_API_KEY internally
    os.environ["MPRESTER_API_KEY"] = key

    # 构建两个 entry set：
    # 1. entry_set_for_rxns：保留所有条目（包括 O2），用于反应枚举
    entry_set: GibbsEntrySet = GibbsEntrySet(list(mp_entries))

    # 2. solid_entries：过滤掉 O2 和无化学式条目，用于 SolidPhaseSet
    solid_entries = [
        e for e in mp_entries
        if e.composition.reduced_formula and e.composition.reduced_formula != "O2"
    ]
    solid_entry_set: GibbsEntrySet = GibbsEntrySet(solid_entries)

    # 构建 SolidPhaseSet（使用原始 ComputedStructureEntry，用于 CA 的体积/密度）
    phase_set: SolidPhaseSet = SolidPhaseSet.from_entry_set(
        entry_set=solid_entry_set,
        entry_metadata={},
    )
    # 同时返回 entry_set（包含 GibbsEntrySet）用于 ReactionSet 构建
    return phase_set, entry_set


def run_audit(protocol: dict) -> dict:
    """Forward audit. Runs in-process and routes arc-melt recipes to the
    arc-melt gate (10 hardware/process/element checks, security token).

    Returns the auditor's dict, or {"status": "ERROR", "raw": ...} on failure —
    never raises, so the loop can iterate on feedback.
    """
    try:
        from .audit.gate import PhaseGraphAuditor, load_api_key
        from .protocol import to_arc_recipe
    except Exception as exc:  # pymatgen missing etc.
        return {"status": "ERROR", "raw": f"audit backend unavailable: {exc}"}

    route = protocol.get("route") or protocol.get("protocol", {}).get("route")
    if route is None:
        from .protocol import infer_route
        route = infer_route(protocol.get("material", ""))
    try:
        if route == "arc_melt":
            auditor = PhaseGraphAuditor(api_key=load_api_key())
            result = auditor.audit_arc_melt(to_arc_recipe(protocol))
            checks = result.get("checks", [])
            result.setdefault(
                "report",
                "### Arc-melt safety audit: {}\n".format(protocol.get("material"))
                + "\n".join(f"- [{c.get('status')}] {c.get('check_id')}: {c.get('message')}"
                            for c in checks),
            )
            return result
        auditor = PhaseGraphAuditor(api_key=load_api_key(required=True))
        return auditor.audit_protocol(protocol)
    except Exception as exc:
        return {"status": "ERROR", "raw": str(exc)}


def parse_audit_result(audit_result: dict) -> dict:
    """Parse audit result into structured feedback."""
    fb = {
        "passed": False,
        "issues": [],
        "suggestions": [],
        "dG": None,
        "Ehull": None,
        "report_text": "",
    }
    
    report = audit_result.get("report", audit_result.get("message", ""))
    fb["report_text"] = report
    
    status = audit_result.get("status")
    execution = audit_result.get("execution_status", "ERROR" if status == "ERROR" else "OK")
    review = audit_result.get("review_status")
    checks = audit_result.get("checks") or []
    stages = audit_result.get("stage_results") or []
    critical = audit_result.get("critical_failures") or []
    failed_check = any(c.get("status") == "FAIL" for c in checks)
    unknown_check = any(c.get("status") in {"HOLD", "UNKNOWN", "ERROR"} for c in checks)
    critical_stage = (bool({"KINETIC_ARREST", "REDOX_RISK_CRITICAL"} & set(audit_result.get("audit_flags") or []))
                      or any(s.get("sintering_status") == "FAIL" or s.get("redox_status") == "FAIL"
                             or {"KINETIC_ARREST", "REDOX_RISK_CRITICAL"} & set(s.get("flags") or [])
                             for s in stages))
    unknown_stage = any(s.get("sintering_status") in {None, "UNKNOWN"}
                        or s.get("redox_status") in {None, "UNKNOWN"} for s in stages)
    if critical or failed_check or critical_stage:
        review = "REJECTED"
    elif execution != "OK" or unknown_check or unknown_stage or not (checks or stages):
        review = "HOLD"
    elif review not in {"APPROVED", "WARNING", "REJECTED", "HOLD"}:
        review = status if status in {"APPROVED", "WARNING", "REJECTED", "HOLD"} else "HOLD"
    fb["execution_status"] = execution
    fb["review_status"] = review
    fb["passed"] = execution == "OK" and review in {"APPROVED", "WARNING"}
    
    # Extract numeric data
    dg_m = re.search(r'dG[=:]?\s*([-+]?\d+\.?\d*)\s*eV', report, re.IGNORECASE)
    if dg_m:
        fb["dG"] = float(dg_m.group(1))
    
    ehull_m = re.search(r'Ehull[=:]?\s*(\d+\.?\d*)\s*meV', report, re.IGNORECASE)
    if ehull_m:
        fb["Ehull"] = float(ehull_m.group(1)) / 1000  # meV → eV
    
    # Issue detection
    issue_kw = {
        "thermodynamic_instability": ["unstable", "metastable", "dG > 0", "positive dG", "thermodynamically unfavorable"],
        "insufficient_driving_force": ["driving force", "driving_force", "driving force insufficient", "dG 绝对值小"],
        "kinetic_barrier": ["kinetic barrier", "diffusion limited", "activation energy", "slow diffusion"],
        "competing_phases": ["competing phase", "secondary phase", "impurity", "competing"],
        "decomposition": ["decomposition", "volatilization", "decompose"],
    }
    
    for issue, kws in issue_kw.items():
        if any(k.lower() in report.lower() for k in kws):
            fb["issues"].append(issue)
    
    # Suggestions
    if "thermodynamic_instability" in fb["issues"]:
        fb["suggestions"].append("提高烧结温度 50–150°C 以增加热力学驱动力")
    if "insufficient_driving_force" in fb["issues"]:
        fb["suggestions"].append("提高温度或调整前驱体配比")
    if "kinetic_barrier" in fb["issues"]:
        fb["suggestions"].append("延长保温时间 60–120 min 或降低升温速率")
    if "competing_phases" in fb["issues"]:
        fb["suggestions"].append("优化升温曲线，增加中间保温段以促进目标相选择")
    if "decomposition" in fb["issues"]:
        fb["suggestions"].append("降低烧结温度或改用保护气氛")
    
    return fb


def refine_protocol(protocol: dict, feedback: dict, iteration: int) -> dict:
    """Apply feedback to refine a protocol."""
    # Standardize material formula to prevent audit crashes
    mat = protocol.get("material", "")
    if mat and mat.lower() == "erta7o19":
        protocol["material"] = "ErTa7O19"

    thermal = protocol.get("protocol", {}).get("thermal_profile", [])
    
    if not thermal:
        return protocol
    
    sintering_stage_idx = max(
        range(len(thermal)), key=lambda i: thermal[i].get("target_temperature_c", 0)
    )
    current_temp = thermal[sintering_stage_idx]["target_temperature_c"]
    
    # Temperature adjustment
    temp_adjustments = {
        "thermodynamic_instability": +100,
        "insufficient_driving_force": +75,
    }
    
    for issue in feedback["issues"]:
        if issue in temp_adjustments:
            new_t = current_temp + temp_adjustments[issue]
            thermal[sintering_stage_idx]["target_temperature_c"] = new_t
            # Extend dwell proportionally
            thermal[sintering_stage_idx]["dwell_time_min"] = min(
                thermal[sintering_stage_idx].get("dwell_time_min", 240) + 60, 480
            )
    
    # Dwell extension for kinetic barriers
    if "kinetic_barrier" in feedback["issues"]:
        for i, stage in enumerate(thermal):
            if stage.get("action") == "dwell" and stage.get("target_temperature_c", 0) > 400:
                thermal[i]["dwell_time_min"] = min(
                    stage.get("dwell_time_min", 120) + 90, 480
                )
    
    protocol["protocol"]["thermal_profile"] = thermal
    protocol["confidence_score"] = max(0.5, protocol.get("confidence_score", 0.7) - 0.05 * iteration)
    protocol["warnings"] = protocol.get("warnings", []) + [
        f"Iter {iteration}: {s}" for s in feedback["suggestions"]
    ]
    
    return protocol


def audit_loop(protocol: dict, max_iterations: int = 5) -> dict:
    """
    Run audit iterations until PASS or max_iterations reached.
    Returns full log dict.
    """
    log = {
        "material": protocol.get("material"),
        "started_at": datetime.now().isoformat(),
        "iterations": [],
        "final_protocol": None,
        "success": False,
        "final_status": "",
    }

    # Standardize material formula to prevent audit crashes from "erta7o19"
    if protocol.get("material", "").lower() == "erta7o19":
        protocol = {**protocol, "material": "ErTa7O19"}
        log["material"] = "ErTa7O19"

    current_protocol = protocol
    
    for it in range(1, max_iterations + 1):
        print(f"\n  ┌─ Audit iteration {it}/{max_iterations}")
        
        audit_result = run_audit(current_protocol)
        feedback = parse_audit_result(audit_result)
        
        print(f"  │  passed={feedback['passed']}")
        if feedback['dG'] is not None:
            print(f"  │  dG={feedback['dG']:.4f} eV")
        if feedback['issues']:
            print(f"  │  issues={feedback['issues']}")
        
        log["iterations"].append({
            "iteration": it,
            "audit_status": audit_result.get("status"),
            "passed": feedback["passed"],
            "dG": feedback["dG"],
            "Ehull": feedback["Ehull"],
            "issues": feedback["issues"],
            "suggestions": feedback["suggestions"],
            "sintering_temp_c": max(
                (s.get("target_temperature_c", 0) for s in current_protocol.get("protocol", {}).get("thermal_profile", [])),
                default=0
            ),
            "report_snippet": feedback["report_text"][:300] if feedback["report_text"] else "",
        })
        
        if feedback["passed"]:
            print(f"  └─ ✅ PASS — protocol approved")
            log["success"] = True
            log["final_status"] = "PASS"
            log["final_protocol"] = current_protocol
            break
        
        print(f"  │  suggestions: {feedback['suggestions']}")
        current_protocol = refine_protocol(current_protocol, feedback, it)
        new_t = max(
            (s.get("target_temperature_c", 0) for s in current_protocol.get("protocol", {}).get("thermal_profile", [])),
            default=0
        )
        print(f"  └─ 🔄 refined → new sintering T = {new_t}°C")
    
    if not log["success"]:
        log["final_status"] = f"MAX_ITERATIONS ({max_iterations})"
        log["final_protocol"] = current_protocol
        print(f"\n  ⚠️  Reached max iterations — best effort protocol returned")
    
    log["ended_at"] = datetime.now().isoformat()
    return log


# ────────────────────────────────────────────────────────────────────────────
# Phase 3 — rn simulation (thermal evolution)
# ────────────────────────────────────────────────────────────────────────────

# rn deprecated — Phase 3a removed (2026-05-07)


# ────────────────────────────────────────────────────────────────────────────
# Phase 3b — Execution plan generator
# ────────────────────────────────────────────────────────────────────────────

def generate_execution_plan(protocol: dict) -> dict:
    """Generate a step-by-step laboratory execution plan from a protocol."""
    thermal = protocol.get("protocol", {}).get("thermal_profile", [])
    precursors = protocol.get("protocol", {}).get("precursors", [])
    atmosphere = protocol.get("protocol", {}).get("atmosphere", "Air")
    crucible = protocol.get("protocol", {}).get("crucible_material", "Alumina")
    mech = protocol.get("protocol", {}).get("mechanical_processing", {})
    
    steps = []
    step_num = 1
    
    # Step 1: Weighing
    total_moles = sum(p.get("molar_ratio", 1.0) for p in precursors)
    steps.append({
        "step": step_num,
        "phase": "Weighing",
        "action": "Weigh precursors according to stoichiometry",
        "details": {
            "precursors": [
                {
                    "chemical": p["chemical"],
                    "molar_ratio": p["molar_ratio"],
                    "mole_fraction": round(p.get("molar_ratio", 1.0) / total_moles, 4),
                    "purity": p.get("purity", "99.9%"),
                }
                for p in precursors
            ],
            "balance_precision_g": 0.0001,
            "note": "Wear gloves and eye protection. Dry precursors at 120°C overnight if hygroscopic."
        }
    })
    step_num += 1
    
    # Step 2: Mixing
    steps.append({
        "step": step_num,
        "phase": "Mixing",
        "action": "Mix precursors thoroughly",
        "details": {
            "method": "Wet ball milling (isopropanol medium, 24h) or dry mortar & pestle (30 min)",
            "equipment": "Agate mortar and pestle, or Y-stabilized ZrO2 balls",
            "note": "Ensure homogeneous mixing — critical for final phase purity"
        }
    })
    step_num += 1
    
    # Step 3: Pressing
    pressing_mpa = mech.get("pressing_pressure_mpa", 200)
    steps.append({
        "step": step_num,
        "phase": "Pelleting",
        "action": f"Press mixed powder into pellet",
        "details": {
            "pressure_mpa": pressing_mpa,
            "tooling": mech.get("tooling", "uniaxial press"),
            "die_diameter_mm": 10,
            "holding_time_min": 5,
            "note": f"Apply {pressing_mpa} MPa uniaxial pressure. Diameter 10 mm die recommended for lab-scale synthesis."
        }
    })
    step_num += 1
    
    # Step 4: Thermal stages
    for stage in thermal:
        action = stage.get("action", "")
        target_t = stage.get("target_temperature_c", 0)
        ramp = stage.get("ramp_rate_c_per_min", 0)
        dwell = stage.get("dwell_time_min", 0)
        desc = stage.get("description", "")
        
        if action == "ramp_up":
            steps.append({
                "step": step_num,
                "phase": "Thermal",
                "action": f"Ramp to {target_t}°C",
                "details": {
                    "rate_c_per_min": abs(ramp),
                    "direction": "up",
                    "atmosphere": atmosphere,
                    "crucible": crucible,
                    "note": f"Use {crucible} crucible. Flow {atmosphere} at 100 mL/min if inert."
                }
            })
        elif action == "dwell" and dwell > 0:
            steps.append({
                "step": step_num,
                "phase": "Thermal",
                "action": f"Dwell at {target_t}°C",
                "details": {
                    "duration_min": dwell,
                    "atmosphere": atmosphere,
                    "crucible": crucible,
                    "note": desc
                }
            })
        elif action == "ramp_down":
            steps.append({
                "step": step_num,
                "phase": "Thermal",
                "action": f"Cool to {target_t}°C",
                "details": {
                    "rate_c_per_min": abs(ramp),
                    "direction": "down",
                    "furnace_cooling": abs(ramp) < 2,
                    "note": "Natural furnace cooling is acceptable below 5°C/min"
                }
            })
        step_num += 1
    
    # Step 5: Characterisation
    steps.append({
        "step": step_num,
        "phase": "Characterisation",
        "action": "Characterize product",
        "details": {
            "XRD": {
                "instrument": "PANalytical/Bruker D8 Advance",
                "geometry": "Bragg-Brentano θ–θ",
                "scan_range_deg": "20–80° 2θ",
                "step_size_deg": 0.02,
                "integration_time_s": 1,
            },
            "SEM_EDS": "Optional — check morphology and composition",
            "Raman": "Optional — phase identification",
            "note": "Use XRD phase identification to feed back into the autonomous loop (Phase 4)"
        }
    })
    
    return {
        "execution_plan": steps,
        "protocol_summary": {
            "material": protocol.get("material"),
            "sintering_temp_c": max((s.get("target_temperature_c", 0) for s in thermal), default=0),
            "atmosphere": atmosphere,
            "crucible": crucible,
            "total_time_h": round(sum(s.get("dwell_time_min", 0) / 60 for s in thermal), 1),
            "confidence": protocol.get("confidence_score", "N/A"),
        }
    }


# ────────────────────────────────────────────────────────────────────────────
# Phase 4 stub — XRD feedback (to be wired later)
# ────────────────────────────────────────────────────────────────────────────

def update_from_xrd(xrd_result: dict, material: str, wiki_path: Path = WIKI_PATH) -> dict:
    """
    Phase 4: Update knowledge base from XRD feedback.
    
    xrd_result expected format:
    {
        "phase_identified": bool,
        "phases": ["BaTiO3", "BaO", "TiO2"],  # list of identified phases
        "major_phase": "BaTiO3",
        "impurities": ["TiO2 rutile"],          # secondary phases
        "pattern_match_rFactor": 0.05,          # match quality
        "notes": "..."
    }
    
    Updates the wiki entity page with real synthesis outcome.
    """
    formula_slug = material.lower().replace(" ", "-").replace("{", "").replace("}", "")
    entity_file = wiki_path / "entities" / f"{formula_slug}.md"
    
    if not entity_file.exists():
        return {"status": "skipped", "reason": f"No wiki entry for {material}"}
    
    content = entity_file.read_text()
    
    today = datetime.now().strftime("%Y-%m-%d")
    new_section = f"""

## Synthesis Validation (XRD Feedback — {today})

- **Outcome**: {'✅ Target phase identified' if xrd_result.get('phase_identified') else '❌ Target NOT identified'}
- **Major phase**: {xrd_result.get('major_phase', 'unknown')}
- **Impurities**: {', '.join(xrd_result.get('impurities', ['none']))}
- **Pattern match R-factor**: {xrd_result.get('pattern_match_rFactor', 'N/A')}
- **Notes**: {xrd_result.get('notes', '')}

### Lessons Learned
"""
    
    # Detect what went wrong based on impurities
    phases = xrd_result.get("phases", [])
    impurities = xrd_result.get("impurities", [])
    
    if "TiO2 rutile" in impurities or "TiO2" in phases:
        new_section += "\n- **TiO2 impurity**: Suggests incomplete reaction between Ba source and TiO2. Consider: (1) longer dwell at intermediate temp, (2) finer TiO2 powder, (3) higher sintering temperature."
    if "BaO" in impurities:
        new_section += "\n- **BaO impurity**: Excess BaO suggests incomplete solid reaction. Consider: (1) increase sintering temperature by 50–100°C, (2) extend dwell time."
    if xrd_result.get("phase_identified"):
        new_section += "\n- **Success**: Target phase confirmed. Protocol is validated — update confidence_score to 0.95."
    
    # Append to content
    lines = content.split("\n")
    # Find last H2 or end of file
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_idx = i
    
    lines.insert(insert_idx, new_section)
    updated_content = "\n".join(lines)
    
    # Update frontmatter updated date
    updated_content = re.sub(
        r'^updated:\s*\d{4}-\d{2}-\d{2}',
        f'updated: {today}',
        updated_content,
        flags=re.MULTILINE
    )
    
    entity_file.write_text(updated_content)
    
    # Log to log.md
    log_file = wiki_path / "log.md"
    log_entry = f"\n## [{today}] xrd_update | {material}\n- Phase identified: {xrd_result.get('phase_identified')}, Major: {xrd_result.get('major_phase')}, Impurities: {impurities}\n"
    with open(log_file, "a") as f:
        f.write(log_entry)
    
    return {"status": "updated", "entity_file": str(entity_file)}


# ────────────────────────────────────────────────────────────────────────────
# MAIN — Orchestrate the full loop
# ────────────────────────────────────────────────────────────────────────────

def full_loop(
    material: str,
    max_audit_iterations: int = 5,
    run_simulation: bool = False,
    force_regenerate: bool = False,
    xrd_file: str = None,
    size_override: int = None,
) -> dict:
    """
    Run the complete autonomous materials chemist loop:
    
    1. Wiki lookup → protocol
    2. If no wiki: literature search → protocol
    3. Audit loop (iterate until PASS or max_iterations)
    4. rn simulation (optional)
    5. Execution plan
    6. Return full report
    """
    print(f"\n{'='*60}")
    print(f"  Autonomous Materials Chemist — {material}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    report = {
        "material": material,
        "started_at": start_time.isoformat(),
        "phase1_knowledge": {},
        "phase2_audit": {},
        "phase3_simulation": {},
        "phase3_execution": {},
        "success": False,
    }
    
    # ── Phase 1: Knowledge → Protocol ──────────────────────────────────────
    print(f"\n[Phase 1] Knowledge acquisition")
    
    wiki_protocol = wiki_lookup(material) if not force_regenerate else None
    
    if wiki_protocol:
        print(f"  ✅ Found in wiki: confidence={wiki_protocol.get('confidence_score')}")
        report["phase1_knowledge"] = {
            "source": "wiki",
            "protocol": wiki_protocol,
        }
        current_protocol = wiki_protocol
    else:
        print(f"  ⚠️  Not found in wiki — searching literature")
        try:
            lit_result = literature_search(material)
        except Exception as exc:   # network/ratelimit/parser: rules still work
            print(f"  [literature] search failed: {exc}")
            lit_result = {"papers": [], "best_protocol": None, "error": str(exc)}
        report["phase1_knowledge"]["literature"] = lit_result

        if lit_result.get("best_protocol"):
            print(f"  ✅ Literature protocol extracted (confidence={lit_result['best_protocol'].get('confidence_score')})")
            current_protocol = lit_result["best_protocol"]
        else:
            print(f"  🔧 Generating from chemistry rules (no literature hit)")
            current_protocol = generate_protocol(material)
            report["phase1_knowledge"]["generation_method"] = "chemistry_rules"

        report["phase1_knowledge"]["protocol"] = current_protocol
        report["phase1_knowledge"]["papers"] = lit_result.get("papers", [])[:4]
    
    print(f"\n  📋 Protocol overview:")
    print(f"     material: {current_protocol.get('material')}")
    print(f"     confidence: {current_protocol.get('confidence_score')}")
    thermal = current_protocol.get("protocol", {}).get("thermal_profile", [])
    sintering_t = max((s.get("target_temperature_c", 0) for s in thermal), default=0)
    print(f"     sintering temp: {sintering_t}°C")
    print(f"     atmosphere: {current_protocol.get('protocol', {}).get('atmosphere', 'Air')}")
    print(f"     precursors: {[p['chemical'] for p in current_protocol.get('protocol', {}).get('precursors', [])]}")

    # ── Hindsight: Phase 1 ────────────────────────────────────────────────
    hindsight_write(material, "phase1",
        {"source": report["phase1_knowledge"].get("source"),
         "confidence": current_protocol.get("confidence_score"),
         "precursors": [p["chemical"] for p in current_protocol.get("protocol", {}).get("precursors", [])]},
        tags=["phase1", "knowledge"],
        confidence=current_protocol.get("confidence_score"))

    # ── Phase 2: Audit Loop ─────────────────────────────────────────────────
    print(f"\n[Phase 2] Audit loop")
    audit_log = audit_loop(current_protocol, max_iterations=max_audit_iterations)
    report["phase2_audit"] = audit_log
    
    final_protocol = audit_log.get("final_protocol", current_protocol)
    report["success"] = audit_log["success"]
    
    if audit_log["success"]:
        print(f"\n  ✅ Audit PASSED after {len(audit_log['iterations'])} iteration(s)")
    else:
        print(f"\n  ⚠️  Audit did not fully pass — best-effort protocol returned")
        print(f"     Consider reviewing literature or adjusting max_iterations")

    # ── Hindsight: Phase 2 ────────────────────────────────────────────────
    hindsight_write(material, "phase2",
        {"success": audit_log["success"],
         "n_iterations": len(audit_log.get("iterations", [])),
         "confidence": audit_log.get("final_protocol", {}).get("confidence_score")},
        tags=["phase2", "audit"],
        confidence=audit_log.get("final_protocol", {}).get("confidence_score"))

    # ── Phase 3a: rxn-ca Cellular Automaton Simulation ───────────────────────
    phase3_result = {"status": "skipped", "reason": "No MP_API_KEY"}
    precursor_formulas = []
    precursor_ratios  = {}

    # 提取前驱体（从 final_protocol 的 precursor_ratios）
    precursors_raw = final_protocol.get("protocol", {}).get("precursors", [])
    if isinstance(precursors_raw, list):
        for p in precursors_raw:
            if isinstance(p, dict):
                formula = p.get("chemical", "")
                ratio   = p.get("molar_ratio", 1.0)
            else:
                formula = str(p)
                ratio   = 1.0
            # Skip only multi-element compounds that are clearly NOT precursors
            # Allow: (1) single-element formulas (Ce, Al, Ge), (2) common oxides (Ce2O3, etc.)
            # block: multi-element non-oxides that are clearly products/byproducts
            is_single_elem = len(formula) <= 3 and formula[0].isupper() and formula.isalpha()
            is_oxide = "O" in formula and formula[0].isupper()
            if formula and (is_single_elem or is_oxide) and formula != material:
                precursor_formulas.append(formula)
                precursor_ratios[formula] = ratio

    # 如果 protocol 没有 precursor_ratios，从 wiki 尝试补充
    # 使用大小写不敏感匹配
    if not precursor_ratios and material.lower() in ("erta7o19",):
        # ErTa7O19 默认前驱体（正确的化学计量）
        precursor_formulas = ["Er2O3", "Ta2O5"]
        precursor_ratios  = {"Er2O3": 1.0, "Ta2O5": 3.5}
    # Ensure precursor_ratios is in final_protocol (needed by _run_rxn_ca_simulation)
    final_protocol["precursor_ratios"] = precursor_ratios
    print(f"\n[Phase 3a] rxn-ca Cellular Automaton simulation")
    # Inject size_override to reduce CA simulation time
    # size=10 → 1000 sites, ~34min per temperature step
    # size=5  → 125 sites,  ~3min per temperature step
    if size_override is not None:
        final_protocol["simulation_size"] = size_override
        print(f"  [size_override={size_override} → {size_override**3} sites]")
    _route = final_protocol.get("route") or final_protocol.get("protocol", {}).get("route")
    if _route == "arc_melt":
        # Intermetallics: arc melting is non-equilibrium, CA is not applicable.
        phase3_result = {"status": "skipped", "reason": "arc_melt route: CA not applicable"}
        print("  [Phase 3a] skipped — arc-melt route (non-equilibrium)")
        mp_key = None
    else:
        mp_key = mp_api_key()
    if mp_key:
        os.environ["MPRESTER_API_KEY"] = mp_key  # get_entries() uses MPRESTER_API_KEY
        try:
            phase_set, entry_set = _build_solid_phase_set(
                material, precursor_formulas, precursor_ratios
            )
            phase3_result = _build_reaction_recipe(
                material, precursor_formulas, final_protocol, phase_set, entry_set
            )
            print(f"  rxn-ca: {phase3_result.get('status', 'unknown')}")
            if phase3_result.get("status") == "ok":
                fracs = phase3_result.get("phase_fractions", {})
                top_phases = sorted(fracs.items(), key=lambda x: -x[1])[:5]
                for ph, wf in top_phases:
                    print(f"    {ph:<20s}  {wf:.3f}  (weight fraction)")
                print(f"  Dominant pathway: {phase3_result.get('dominant_pathways', ['N/A'])[0]}")
        except Exception as ex:
            import traceback
            print(f"  rxn-ca failed: {ex}")
            traceback.print_exc()
            phase3_result = {"status": "error", "error": str(ex)}
    else:
        print(f"  MP_API_KEY not set — skipping CA simulation")

    report["phase3_simulation"] = phase3_result
    
    # ── Phase 3b: Execution Plan ────────────────────────────────────────────
    print(f"\n[Phase 3b] Generating execution plan")
    exec_plan = generate_execution_plan(final_protocol)
    report["phase3_execution"] = exec_plan
    print(f"  ✅ {len(exec_plan['execution_plan'])} steps generated")

    # ── Hindsight: Phase 3b ──────────────────────────────────────────────
    hindsight_write(material, "phase3b",
        {"n_steps": len(exec_plan["execution_plan"]),
         "steps": [s.get("step_type") for s in exec_plan["execution_plan"]]},
        tags=["phase3b", "execution"])

    # ── Phase 4: XRD Experimental Validation ─────────────────────────────────
    phase4_xrd = {}
    if xrd_file:
        from pathlib import Path
        from xrd_analyzer import parse_dat, find_peaks, identify_phase, init_db

        print(f"\n[Phase 4] XRD validation: {xrd_file}")
        try:
            from xrd_analyzer import get_conn, scan_dat
            conn = get_conn()
            identified = scan_dat(Path(xrd_file), tolerance=0.35, required_match_frac=0.55)
            conn.close()

            # Normalize scan_dat output (uses 'phase' key) to Phase 4 schema (uses 'formula')
            for r in (identified or []):
                r["formula"] = r.pop("phase", None)
                r["total_ref_peaks"] = r.pop("n_ref_peaks", None)
                r["matched_peaks"]    = r.pop("n_matched", None)

            # Filter: exclude experimental self-references, require ≥4 matched peaks
            MIN_MATCHED = 4
            MIN_CONFIDENCE = 0.15   # weighted confidence threshold
            filtered = [
                r for r in (identified or [])
                if r.get("source", "") != "experimental"
                and r.get("n_matched", 0) >= MIN_MATCHED
                and r.get("confidence", 0) >= MIN_CONFIDENCE
            ]

            # Sort by confidence then by n_matched as tiebreaker
            filtered.sort(key=lambda x: (x.get("confidence", 0), x.get("n_matched", 0)), reverse=True)

            main_phase = filtered[0] if filtered else None
            impurities = [r for r in filtered[1:] if r.get("n_matched", 0) >= MIN_MATCHED]

            # Determine experimental ground truth
            expected_formula = material.replace(" ", "")
            main_phase_correct = (
                main_phase and
                main_phase.get("confidence", 0) >= MIN_CONFIDENCE and
                main_phase.get("formula", "").replace(" ", "") == expected_formula
            )

            # ── Derive weight fractions from matched-peak counts ──────────────────
            # scan_dat does not return explicit wfs; approximate from matched peaks
            # so audit_result can compute PHASE_INTERCEPTION labels
            total_matched = sum(int(r.get("n_matched", 0)) for r in (identified or []))
            xrd_phases = {}
            if total_matched > 0 and identified:
                for r in identified:
                    wf = int(r.get("n_matched", 0)) / total_matched
                    formula = r.get("phase", "unknown")
                    xrd_phases[formula] = round(wf, 4)

            # ── Phase 4 QPA: use create_estimator if system is configured ─────────
            # Only available for systems defined in chem.xrd.SYSTEM_CONFIGS
            qpa_result = None
            # Identification confidence and matched-peak counts are not
            # quantitative physical yields.
            yield_pct = None
            if total_matched > 0:
                # Map material formula → system key (lowercase no-spaces)
                system_key = material.lower().replace(" ", "")
                try:
                    from .xrd import create_estimator, SYSTEM_CONFIGS
                    if system_key in SYSTEM_CONFIGS:
                        est = create_estimator(system_key)
                        qpa_result = est.estimate(xrd_file)
                        yield_pct = qpa_result.yield_pct
                        # Override xrd_phases with proper Pawlik/Rietveld wt%
                        if qpa_result.phase_fractions:
                            xrd_phases = {
                                ph: round(wf, 4)
                                for ph, wf in qpa_result.phase_fractions.items()
                            }
                        print(f"  [QPA] {qpa_result}")
                    else:
                        print(f"  [QPA] system '{system_key}' not configured — "
                              f"quantitative yield remains unknown")
                except Exception as qpa_ex:
                    print(f"  [QPA] estimator failed ({qpa_ex}) — "
                          f"quantitative yield remains unknown")

            # ── Call audit_result for structured semantic priors ─────────────────
            ar_result = None
            try:
                if yield_pct is not None:
                    ar_result = _run_audit_result(xrd_phases, yield_pct, final_protocol)
            except Exception as ex:
                print(f"  ⚠️  audit_result failed: {ex}")

            if ar_result:
                for sp in ar_result.get("semantic_priors", []):
                    tag   = sp.get("tag", "UNKNOWN")
                    conf  = sp.get("confidence", 0.0)
                    detail = sp.get("detail", "")
                    # Write each semantic prior tag to Hindsight for cross-session recall
                    hindsight_write(
                        material, "phase4",
                        {"tag": tag, "confidence": conf, "detail": detail},
                        tags=["phase4", tag, "semantic_prior"],
                        confidence=conf,
                    )
                advice = ar_result.get("actionable_advice", [])
                if advice:
                    print(f"\n  📋 Structured diagnostic (audit_result):")
                    for a in advice:
                        print(f"     → {a}")

            phase4_xrd = {
                "xrd_file":               xrd_file,
                "main_phase":             main_phase,
                "impurities":             impurities,
                "main_phase_correct":     main_phase_correct,
                "confidence":             main_phase.get("confidence") if main_phase else None,
                "ground_truth":           "experimental",
                "matched_peaks":          main_phase.get("matched_peaks") if main_phase else 0,
                "total_ref_peaks":        main_phase.get("total_ref_peaks") if main_phase else 0,
                "all_results":            identified,
                "xrd_phases_approx":      xrd_phases,
                "yield_pct_approx":       round(yield_pct, 1) if yield_pct is not None else None,
                "use_for_bo":            yield_pct is not None and bool((qpa_result.phase_peaks_used or {}).get("use_for_bo", True)),
                "audit_result_tags":      [sp["tag"] for sp in ((ar_result or {}).get("semantic_priors") or [])],
                "actionable_advice":      ar_result.get("actionable_advice") if ar_result else [],
            }

            report["phase4_xrd"] = phase4_xrd

            if main_phase_correct:
                status = "✅ MAIN PHASE CONFIRMED"
                confidence_label = "HIGH"
            else:
                status = "⚠️  MAIN PHASE NOT CONFIRMED"
                confidence_label = "GROUND_TRUTH_EXPERIMENTAL"

            print(f"  {status}")
            if main_phase:
                print(f"     Formula:     {main_phase['formula']}  (confidence={main_phase['confidence']:.1%})")
                print(f"     Matched:     {int(main_phase.get('matched_peaks', 0))}/{main_phase.get('total_ref_peaks', 0)} peaks")
            if impurities:
                print(f"  Impurities detected ({len(impurities)}):")
                for imp in impurities:
                    print(f"     - {imp['formula']}  (confidence={imp['confidence']:.1%})")

            # ── Persist experimental ground truth ──────────────────────────────
            # Update Hindsight with experimental result via HTTP API
            try:
                from datetime import date
                import urllib.request as _urllib
                hcontent = (
                    f"XRD experiment for {material}: "
                    f"main_phase={main_phase['formula'] if main_phase else 'UNKNOWN'}, "
                    f"confidence={main_phase.get('confidence','N/A') if main_phase else 'N/A'}, "
                    f"impurities={[imp['formula'] for imp in (impurities or [])]}, "
                    f"xrd_file={Path(xrd_file).name}, "
                    f"date={date.today().isoformat()}"
                )
                payload = json.dumps({
                    "items": [{"content": hcontent,
                               "context": f"experimental_validation: {material}",
                               "tags": ["experimental","XRD",material,
                                        "verified" if main_phase_correct else "unverified"]}],
                    "async": False
                }).encode()
                req = _urllib.Request(
                    "http://127.0.0.1:8888/v1/default/banks/hermes/memories",
                    data=payload, headers={"Content-Type": "application/json"}
                )
                with _urllib.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 202):
                        print(f"  ✅ Hindsight updated")
                    else:
                        print(f"  ⚠️  Hindsight HTTP {resp.status}")
            except Exception as ex:
                print(f"  ⚠️  Hindsight write failed: {ex}")

            # Update wiki entity with XRD result
            try:
                entity_path = Path(WIKI_PATH) / "entities" / f"{expected_formula.lower().replace(' ', '')}.md"
                if entity_path.exists():
                    content = entity_path.read_text()
                    xrd_tag = f"\n\n## XRD Validation\n- **Date**: {date.today().isoformat()}\n- **Result**: {'PASS' if main_phase_correct else 'FAIL'}\n- **Main phase**: {main_phase['formula'] if main_phase else 'Not detected'}\n- **Confidence**: {main_phase.get('confidence','N/A') if main_phase else 'N/A'}\n- **Impurities**: {[imp['formula'] for imp in impurities]}\n"
                    content += xrd_tag
                    entity_path.write_text(content)
                    print(f"  📝 Wiki updated: {entity_path.name}")
            except Exception as ex:
                print(f"  ⚠️  Wiki update failed: {ex}")

        except Exception as ex:
            phase4_xrd = {"error": str(ex)}
            report["phase4_xrd"] = phase4_xrd
            print(f"  ❌ XRD analysis failed: {ex}")
    else:
        report["phase4_xrd"] = None
        print(f"\n[Phase 4] Skipped (no XRD file provided)")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    report["elapsed_seconds"] = round(elapsed, 1)
    report["ended_at"] = datetime.now().isoformat()
    
    print(f"\n{'='*60}")
    print(f"  Loop complete — {elapsed:.1f}s — success={report['success']}")
    print(f"{'='*60}")
    
    return report


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from chem.cli import main as _cli_main
    raise SystemExit(_cli_main(["loop"] + sys.argv[1:]))
