"""
★ ★ ★ 化学合成家 ★ 通用 step_helpers ★ ★ ★
★ iter_agent_helpers.py
★ ★ 完全通用 ★ 不依赖任何特定化学体系
"""
from typing import Dict, List, Optional, Tuple, Any


def step_bg(proj, hist, n: int = 8):
    """Background fitting (Chebyshev polynomial, n terms)"""
    try:
        hist.data['Background'][0] = list(range(1, n+1))
        hist.data['Background'][1] = n + 1
        hist.data['Background'][2] = True
        hist.set_refinements({'Background': True})
    except Exception:
        pass


def step_scale(proj, phases, hist):
    """Open Scale refinement on all phases"""
    for n, ph in phases.items():
        try:
            # ★ 使用 set_refinements (★ GSAS-II 在 proj 上下文的正确方法)
            ph.set_refinements({'Scale': True})
        except Exception:
            pass


def step_cell(proj, phases, hist):
    """Cell refinement on all phases"""
    for n, ph in phases.items():
        try:
            ph.set_refinements({'Cell': True})
        except Exception:
            pass


def step_size(proj, phases, hist, target_phase: str):
    """Size (crystallite) refinement on target phase"""
    if target_phase in phases:
        try:
            phases[target_phase].set_refinements({'Size': True})
        except Exception:
            pass


def step_U(proj, phases, hist):
    """Isotropic Uiso refinement"""
    for n, ph in phases.items():
        try:
            ph.set_refinements({'U': True})
        except Exception:
            pass


def step_mustrain(proj, phases, hist, target_phase: str):
    """Microstrain refinement"""
    if target_phase in phases:
        try:
            phases[target_phase].set_refinements({'Mustrain': True})
        except Exception:
            pass


def step_PO(proj, phases, hist, target_phase: str, axis: Tuple[int, int, int] = (0, 0, 1)):
    """Preferred orientation (March-Dollase)"""
    if target_phase in phases:
        try:
            phases[target_phase].set_refinements({'PO': True})
        except Exception:
            pass


def parse_lst(lst_path: str, phase_names: List[str]) -> Tuple[Optional[float], Optional[float], Dict]:
    """Parse GSAS-II LST file to extract Rwp, GoF, phase fractions"""
    import re
    import os
    
    rwp = None
    gof = None
    fractions = {}
    
    if not os.path.exists(lst_path):
        return rwp, gof, fractions
    
    with open(lst_path, 'r') as f:
        content = f.read()
    
    # ★ GSAS-II 实际格式: "wR = 98.60%" (★ 不是 Rwp)
    m = re.search(r'wR\s*=\s*([\d.]+)\s*%', content)
    if m:
        rwp = float(m.group(1))
    
    # ★ GoF: "GOF =  27.75" (★ 不是 gof)
    m = re.search(r'GOF\s*=\s*([\d.]+)', content)
    if m:
        gof = float(m.group(1))
    
    # ★ Phase weight: "Phase fraction: 1 Weight fraction:  0.23483, sig  0.00000"
    # ★ ★ 关键: Phase name 和 Weight fraction 距离可能 > 38000 字符
    # ★ 全局搜索模式
    phase_pattern = re.compile(r'Phase name:\s*(\S+)', re.MULTILINE)
    weight_pattern = re.compile(r'Weight fraction\s*:\s*([\d.]+),\s*sig\s+([\d.]+)')
    
    # ★ 找所有 Phase name (按出现顺序)
    all_phases = list(phase_pattern.finditer(content))
    # ★ 找所有 Weight fraction (按出现顺序)
    all_weights = list(weight_pattern.finditer(content))
    
    # ★ 按出现顺序配对 (★ 假设 Weight fraction #i 对应 Phase #i)
    for i, phase_match in enumerate(all_phases):
        if i < len(all_weights):
            name = phase_match.group(1)
            weight_match = all_weights[i]
            wt = float(weight_match.group(1))
            sig = float(weight_match.group(2))
            if name in phase_names:
                fractions[name] = (wt, sig)
    
    return rwp, gof, fractions