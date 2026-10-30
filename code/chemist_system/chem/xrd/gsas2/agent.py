"""
★ ★ ★ ★ 化学合成家 ★ 通用迭代精修 Agent ★ ★ ★ ★

★ 完全通用 ★ 不依赖任何特定化学体系 (Ce-Al-Ge, Gd-P-O, Mn-Cr-Fe-O 等)
★ 输入: chemistry_system + target_phase + raw_xrd_path
★ 自动: 候选相构建 + Rietveld 迭代 + 智能化决策

Version: 1.0 (2026-06-11)
Author: chemical synthesis automation
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ★ 化学合成家核心库
from ... import paths, config

def _mprester():
    """Import MPRester lazily: importing mp_api at module scope drags in a
    libstdc++ that collides with scipy's compiled extensions on some hosts."""
    from mp_api.client import MPRester
    return MPRester


import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares


# ============================================================
# ★ 通用化 ★ 配置
# ============================================================
DEFAULT_ARC_ROOT = os.environ.get('CHEMIST_ARC_ROOT', str(paths.ARC_RAW))
DEFAULT_CIF_DIR = os.environ.get('CHEMIST_CIF_DIR', str(paths.RIETVELD_CIFS))
DEFAULT_RAW_DIR = os.environ.get('CHEMIST_RAW_DIR', DEFAULT_ARC_ROOT)
MP_API_KEY = config.mp_api_key() or ''

# ★ 通用氧化物化学计量
COMMON_OXIDES = ['O', '2O3', 'O2', 'O3', 'O4', '2O', 'O5']

# ★ 通用候选相结构类型 (★ 任何体系都可能形成)
COMMON_STRUCTURE_TYPES = {
    'spinel': 'AB2O4',  # 立方尖晶石, Fd-3m
    'perovskite': 'ABO3',  # 钙钛矿, Pm-3m
    'rock_salt': 'MO',  # 岩盐, Fm-3m
    'corundum': 'M2O3',  # 刚玉, R-3c
    'spinel_solid': '(A,B)3O4',  # 尖晶石固溶体
}


# ============================================================
# ★ 通用化 ★ ★ 候选相生成器 (★ ★ 删除 ★ ★ Ce-Al-Ge 特定代码)
# ============================================================
def auto_build_candidates(chemistry_system: List[str], cif_dir: str = DEFAULT_CIF_DIR) -> Dict[str, str]:
    """
    ★ 通用 ★ 自动构建化学体系候选相字典
    
    Args:
        chemistry_system: 化学元素列表, e.g. ['Mn', 'Cr', 'Fe', 'O']
        cif_dir: CIF 文件目录
    
    Returns:
        {phase_name: cif_path} 字典
    """
    candidates = {}
    cations = [el for el in chemistry_system if el != 'O']
    
    # 1. 单质
    for el in cations:
        cif_path = os.path.join(cif_dir, f'{el}.cif')
        if os.path.exists(cif_path):
            candidates[el] = cif_path
    
    # 2. 简单氧化物 (★ 任何金属)
    for el in cations:
        for stoich in COMMON_OXIDES:
            name = f'{el}{stoich}'
            cif_path = os.path.join(cif_dir, f'{name}.cif')
            if os.path.exists(cif_path):
                candidates[name] = cif_path
    
    # 3. 复杂相 (★ spinel AB2O4, 任何 3+ 元素体系)
    if len(cations) >= 2:
        for el1 in cations:
            for el2 in cations:
                if el2 == el1:
                    continue
                # 尖晶石 AB2O4
                name = f'{el1}{el2}2O4'
                cif_path = os.path.join(cif_dir, f'{name}.cif')
                if os.path.exists(cif_path):
                    candidates[name] = cif_path
                # 钙钛矿 ABO3
                name = f'{el1}{el2}O3'
                cif_path = os.path.join(cif_dir, f'{name}.cif')
                if os.path.exists(cif_path):
                    candidates[name] = cif_path
    
    # 4. ★ 4 元体系 → ★ ★ 复杂尖晶石固溶体
    if len(cations) == 3:
        for el1 in cations:
            for el2 in cations:
                if el2 == el1:
                    continue
                for el3 in cations:
                    if el3 == el1 or el3 == el2:
                        continue
                    # 4 元尖晶石
                    name = f'{el1}{el2}2{el3}O8'  # A B2 C O8
                    cif_path = os.path.join(cif_dir, f'{name}.cif')
                    if os.path.exists(cif_path):
                        candidates[name] = cif_path
    
    return candidates


def init_phases_for_target(target_phase: str, chemistry_system: List[str], 
                           cif_dir: str = DEFAULT_CIF_DIR) -> List[str]:
    """
    ★ 通用 ★ 根据 ★ target_phase ★ 和 ★ chemistry_system ★ 生成初始相
    
    Args:
        target_phase: 目标相, e.g. 'Mn2Cr3FeO8'
        chemistry_system: 化学体系, e.g. ['Mn', 'Cr', 'Fe', 'O']
    
    Returns:
        初始相列表
    """
    init = [target_phase]
    
    # ★ 自动推断 ★ 起始物氧化物
    cations = [el for el in chemistry_system if el != 'O']
    
    for el in cations:
        # 默认化学计量: 3+ 价用 2O3, 4+ 价用 O2, 2+ 价用 O
        # ★ 这里用 ★ ★ 通用 ★ ★ 默认: ★ 2O3 (★ ★ 多数稳定)
        candidates_stoich = [f'{el}2O3', f'{el}O2', f'{el}O', f'{el}3O4']
        for stoich in candidates_stoich:
            cif_path = os.path.join(cif_dir, f'{stoich}.cif')
            if os.path.exists(cif_path):
                init.append(stoich)
                break
    
    return init


def extra_candidates_for_system(chemistry_system: List[str], 
                                cif_dir: str = DEFAULT_CIF_DIR) -> List[str]:
    """
    ★ 通用 ★ 根据 ★ chemistry_system ★ 生成额外候选相列表
    
    Args:
        chemistry_system: 化学元素列表
    
    Returns:
        额外候选相列表
    """
    extras = []
    cations = [el for el in chemistry_system if el != 'O']
    
    for el in cations:
        for stoich in COMMON_OXIDES:
            name = f'{el}{stoich}'
            if name not in extras:
                extras.append(name)
    
    # ★ 复杂相
    if len(cations) >= 2:
        for el1 in cations:
            for el2 in cations:
                if el2 == el1:
                    continue
                for formula in [
                    f'{el1}{el2}2O4',   # AB2O4 spinel
                    f'{el1}2{el2}O4',   # A2BO4 inverse/ordered spinel variants
                    f'{el1}{el2}O3',
                ]:
                    if formula not in extras:
                        extras.append(formula)
    
    return extras


def download_missing_cifs(chemistry_system: List[str], cif_dir: str = DEFAULT_CIF_DIR) -> Dict[str, str]:
    """
    ★ 通用 ★ 从 MP 自动下载 ★ 化学体系 ★ 缺失 CIF
    
    ★ 这是 ★ ★ 化学合成家 ★ ★ 最关键的 ★ 通用化 ★ 功能
    ★ 不依赖 ★ 任何硬编码 ★ 相列表
    ★ 自动从 MP 下载 ★ 同体系相
    """
    if not config.has_mp_key():
        return {}
    
    os.makedirs(cif_dir, exist_ok=True)
    downloaded = {}
    
    # 自动构建要下载的相列表
    candidate_names = extra_candidates_for_system(chemistry_system, cif_dir)
    
    MPRester = _mprester()
    with MPRester(api_key=MP_API_KEY) as mpr:
        for name in candidate_names:
            cif_path = os.path.join(cif_dir, f'{name}.cif')
            if os.path.exists(cif_path):
                downloaded[name] = cif_path
                continue
            
            # ★ 解析化学式, ★ ★ 自动从 MP 搜索
            try:
                # 简单解析: 提取元素和化学计量
                docs = mpr.materials.summary.search(
                    formula=name,
                    fields=["material_id", "formula_pretty", "structure"]
                )
                if docs:
                    # 选 E_hull=0 的最稳定相
                    best = None
                    best_ehull = float('inf')
                    for doc in docs:
                        ehull = doc.energy_above_hull if doc.energy_above_hull is not None else 0
                        if ehull < best_ehull:
                            best_ehull = ehull
                            best = doc
                    
                    if best:
                        best.structure.to(filename=cif_path, fmt='cif')
                        downloaded[name] = cif_path
            except Exception:
                continue
    
    return downloaded


# ============================================================
# ★ 通用化 ★ ★ 迭代精修 Agent (★ 完全通用, ★ 不依赖任何体系)
# ============================================================
class GenericIterAgent:
    """
    ★ 化学合成家 ★ 通用迭代精修 Agent
    
    ★ 完全通用: 支持任何 ★ 4 元 / 3 元 / 2 元 氧化物体系
    ★ 自动: 
       - 从 MP 下载候选 CIF
       - 多轮 Rietveld 精修 (bg, scale, cell, size, U, mustrain)
       - 智能分析 (kill 微弱相, fix 高误差相)
       - 动态加相 (当 Rwp 差时)
    """
    
    def __init__(self, sample: str, target_phase: str, raw_xrd_path: str, 
                 chemistry_system: List[str], output_dir: Optional[str] = None,
                 target_rwp: float = 15.0, conda_env: str = 'gsas2',
                 auto_download: bool = True, limits: Tuple[float, float] = (10.0, 90.0),
                 extra_phases: Optional[List[str]] = None):
        """
        初始化通用迭代精修 Agent
        
        Args:
            sample: 样本名 (用于工作目录命名)
            target_phase: 目标相, e.g. 'Mn2Cr3FeO8'
            raw_xrd_path: 实验 XRD 文件路径
            chemistry_system: 化学体系, e.g. ['Mn', 'Cr', 'Fe', 'O']
            output_dir: 输出目录 (★ 默认 ★ ARC_ROOT/rietveld_engine/runs/iter/{sample})
            target_rwp: 目标 Rwp (默认 15%)
            conda_env: GSAS-II conda 环境名
            auto_download: ★ ★ 是否自动从 MP 下载缺失 CIF
        """
        self.sample = sample
        self.target_phase = target_phase
        self.chemistry_system = chemistry_system
        self.target_rwp = target_rwp
        self.conda_env = conda_env
        self.auto_download = auto_download
        self.limits = limits
        self.user_extra_phases = extra_phases or []
        
        # ★ 目录设置 (★ 完全通用)
        arc_root = DEFAULT_ARC_ROOT
        self.cif_dir = DEFAULT_CIF_DIR
        if output_dir is None:
            self.output_dir = f'{arc_root}/rietveld_engine/runs/iter/{sample}'
        else:
            self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # ★ 1. ★ 复制 XRD 到 ★ ARC_ROOT/{sample}.raw
        # ★ 实际: 使用 ★ .txt 扩展名 (★ ★ GSAS-II 正确识别为 Rigaku text 格式)
        os.makedirs(DEFAULT_RAW_DIR, exist_ok=True)
        arc_raw = f'{DEFAULT_RAW_DIR}/{sample}.txt'
        if not os.path.exists(arc_raw) or os.path.getmtime(arc_raw) < os.path.getmtime(raw_xrd_path):
            try:
                shutil.copy(raw_xrd_path, arc_raw)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(arc_raw), exist_ok=True)
                shutil.copy(raw_xrd_path, arc_raw)
        self.raw_xrd_path = arc_raw
        
        # ★ 2. ★ 自动下载 CIF (★ 通用 ★ ★ MP API)
        if auto_download:
            download_missing_cifs(chemistry_system, self.cif_dir)
        
        # ★ 3. ★ 构建 ★ 候选相字典 (★ ★ 通用, ★ ★ 不硬编码)
        self.candidates = auto_build_candidates(chemistry_system, self.cif_dir)
        
        # ★ 4. ★ 自动生成 ★ 初始相
        self.init_phases = init_phases_for_target(target_phase, chemistry_system, self.cif_dir)
        
        # ★ 5. ★ 自动生成 ★ 额外候选
        self.extra_candidates = extra_candidates_for_system(chemistry_system, self.cif_dir)
        for phase in self.user_extra_phases:
            if phase and phase not in self.extra_candidates:
                self.extra_candidates.append(phase)
        
        # ★ 6. ★ GSAS-II 项目
        os.environ['GSAS2_REFINEMENT_SUBPROCESS'] = '1'
        
        # ★ log
        self.log = []
    
    def setup_project(self):
        """★ 通用 ★ 项目设置 (★ ★ 不依赖任何体系)"""
        from .refinement.gsas2_runner import GSAS2Runner
        from .refinement.cif_manager import CifManager
        
        cm = CifManager()
        recs = [cm.resolve_cif(n, preferred_path=self.candidates.get(n), allow_download=False) 
                for n in self.init_phases]
        
        runner = GSAS2Runner(output_dir=self.output_dir)
        self.proj, self.hist, self.phases, self.gpx = runner.build_project(
            self.raw_xrd_path, recs, limits=self.limits
        )
        self.proj['Controls']['data']['max cyc'] = 50
        self.hist.data['Sample Parameters']['Scale'][1] = False
        
        # ★ ★ 关键: 打开所有相的 Scale 精修 (★ ★ 不然 LST 标 Refine? False)
        for n, ph in self.phases.items():
            try:
                ph.set_refinements({'Scale': True})
            except Exception:
                pass
        
        self.fixed = set()
        self.killed = set()
        self.lst_path = f'{self.output_dir}/project_initial.lst'

    def reliability_report(self, rwp: Optional[float], gof: Optional[float], fr: Dict) -> Dict[str, Any]:
        """Classify whether phase fractions can be used quantitatively."""
        warnings = []
        quantification_allowed = True
        if rwp is None or rwp > 35.0:
            quantification_allowed = False
            warnings.append(f'Rwp too high for quantitative phase analysis: {rwp}')
        if gof is None or gof > 5.0:
            quantification_allowed = False
            warnings.append(f'GoF too high for quantitative phase analysis: {gof}')
        if not fr:
            quantification_allowed = False
            warnings.append('No parsable phase fractions in LST output')
        total = sum(v[0] for v in fr.values()) if fr else 0.0
        if fr and not (0.98 <= total <= 1.02):
            quantification_allowed = False
            warnings.append(f'Phase fractions do not sum to ~1: {total:.4f}')
        phase_quality = {}
        for name, (wt, sig) in fr.items():
            rel = sig / wt if wt > 0 else float('inf')
            phase_quality[name] = {
                'wt_percent': round(wt * 100, 4),
                'sigma_percent': round(sig * 100, 4),
                'relative_sigma': round(rel, 4) if rel != float('inf') else None,
                'quantitative': bool(wt > 0 and rel <= 0.35),
            }
            if wt > 0 and rel > 0.35:
                warnings.append(f'High relative uncertainty for {name}: {rel:.1%}')
        label = 'publication_ready'
        if not quantification_allowed:
            label = 'qualitative_only'
        elif rwp and rwp > 20.0:
            label = 'screening_only'
        elif rwp and rwp > 12.0:
            label = 'usable_with_caution'
        return {
            'label': label,
            'quantification_allowed': quantification_allowed,
            'phase_quality': phase_quality,
            'warnings': warnings,
            'thresholds': {
                'max_rwp_quantitative': 35.0,
                'max_gof_quantitative': 5.0,
                'max_rwp_publication_ready': 12.0,
            },
        }

    def _filter_rejected_fractions(self, fr: Dict) -> Dict:
        """Remove killed/rejected phases from parsed fractions and renormalize."""
        if not fr:
            return fr
        filtered = {k: v for k, v in fr.items() if k not in self.killed}
        total = sum(v[0] for v in filtered.values())
        if total <= 0:
            return filtered
        return {k: (v[0] / total, v[1] / total) for k, v in filtered.items()}
    
    def refine_step(self, label: str = 'Step') -> Tuple[Optional[float], Optional[float], Dict]:
        """★ 通用 ★ 精修一步"""
        from .refinement.iter_agent_helpers import parse_lst
        
        err = None
        try:
            self.proj.do_refinements([{}])
            self.proj.save()
        except Exception as e:
            err = str(e)[:200]
        
        rwp, gof, fr = parse_lst(self.lst_path, list(self.phases))
        self.log.append({
            'step': label, 'rwp': rwp, 'gof': gof,
            'wt': {k: round(v[0]*100, 2) for k, v in fr.items()},
            'sig': {k: round(v[1]*100, 4) for k, v in fr.items()},
            'fixed': sorted(self.fixed),
            'err': err
        })
        return rwp, gof, fr
    
    def analyse(self, fr: Dict, allow_kill: bool = True, min_round: int = 3):
        """★ 通用 ★ 智能分析 (★ ★ 自动 kill 微弱相, ★ ★ fix 高误差相)
        ★ ★ ★ 2026-06-11 Hermes 改进 (★ 不死板, ★ ★ 智能 3 层策略):
          1. ★ 起始物 (★ 氧化物) 永 ★ 不删 (★ 物理必保护)
          2. ★ 主要相 (★ wt > 5%) 永 ★ 不删 (★ 真实产物, ★ 不是拟合假象)
          3. ★ 目标相 + 痕量相 (★ wt < 5%) ★ 智能删 (★ ★ ★ 目标相低 ★ 也删)
        """
        for n, (wt, sg) in fr.items():
            if n in self.fixed:
                continue
            rel = sg / wt if wt > 0 else float('inf')
            
            # sigma 爆炸 → fix
            if sg > 1.0 or rel > 5.0:
                self.fix_phase(n, f'sigma blow-up sg={sg:.2e} rel={rel:.1f}')
                continue
            
            # 高相对误差 → fix
            if rel > 0.30:
                self.fix_phase(n, f'rel_err={rel:.1%}')
                continue
            
            # ★ ★ 3 层智能策略 ★ ★
            if allow_kill and wt < 0.05:  # ★ ★ 5% 阈值 (★ 不死板)
                # ★ ★ 保护 1: 起始物 (★ 氧化物) 永 ★ 不删
                if n.endswith('O3') or n.endswith('O2') or n.endswith('O4') or n == 'O' or n == '2O':
                    continue
                # ★ ★ 保护 2: 主要相 (★ wt > 5%) 永 ★ 不删
                if wt > 0.05:
                    continue
                # ★ ★ 保护 3: 目标相 ★ ★ 弱保护: ★ ★ 仅保护 wt > 1% (★ 但 ★ ★ 仍可删)
                # ★ 如果目标相 < 1%, ★ 删除 (★ 真实地反映 ★ 失败)
                if n == self.target_phase and wt > 0.01:
                    continue
                # ★ ★ 否则 ★ kill (★ ★ 目标相低也删 ★ ★)
                self.kill_phase(n, f'wt={wt:.4f} negligible')
    
    def fix_phase(self, n: str, reason: str = ''):
        """★ 通用 ★ 固定某相 (★ ★ scale 不变)"""
        if n not in self.phases or n in self.fixed:
            return
        try:
            # ★ 使用 set_refinements (★ GSAS-II 正确 API)
            self.phases[n].set_refinements({'Scale': False})
            # 同时尝试 HAP (★ 兼容性)
            try:
                self.phases[n].set_HAP_refinements({'Scale': False})
            except Exception:
                pass
            self.phases[n].data['Histograms'][self.hist.name]['Scale'][1] = False
        except Exception:
            pass
        self.fixed.add(n)
        self.log.append({'decision': f'fix {n}: {reason}'})
    
    def kill_phase(self, n: str, reason: str = ''):
        """★ 通用 ★ 杀死某相 (★ scale → 0)"""
        if n not in self.phases or n in self.killed:
            return
        try:
            self.phases[n].data['Histograms'][self.hist.name]['Scale'][0] = 1e-12
            self.phases[n].set_refinements({'Scale': False})
            try:
                self.phases[n].set_HAP_refinements({'Scale': False})
            except Exception:
                pass
            self.phases[n].data['Histograms'][self.hist.name]['Scale'][1] = False
        except Exception:
            pass
        self.killed.add(n)
        self.fixed.add(n)
        self.log.append({'decision': f'kill {n}: {reason}'})
    
    def try_add(self, name: str) -> bool:
        """★ 通用 ★ 动态加相 (★ ★ ★ 通用 ★ ★)"""
        if name in self.phases:
            return False
        from gsas2_refinement.cif_manager import CifManager
        
        cif_path = self.candidates.get(name)
        if not cif_path or not os.path.exists(cif_path):
            return False
        
        cm = CifManager()
        rec = cm.resolve_cif(name, preferred_path=cif_path, allow_download=False)
        try:
            phobj = self.proj.add_phase(rec.file_path, phasename=name, histograms=[self.hist])
            self.phases[name] = phobj
            self.log.append({'decision': f'add phase {name}'})
            return True
        except Exception as e:
            self.log.append({'decision': f'add {name} FAILED: {str(e)[:120]}'})
            return False
    
    def run(self, max_rounds: int = 5) -> Dict[str, Any]:
        """★ 通用 ★ 迭代精修主循环"""
        print(f"\n=== {self.sample} ({self.target_phase}) ===")
        print(f"  Chemistry: {self.chemistry_system}")
        print(f"  Initial phases: {self.init_phases}")
        print(f"  Target Rwp: {self.target_rwp}%")
        
        self.setup_project()
        
        best_rwp = 1e9
        best_gof = None
        best_fr = {}
        
        for round_i in range(max_rounds):
            print(f"\n--- Round {round_i+1} ---")
            
            # A. Background + Scale (★ Step 1+2 组合) - ★ 关键: 先开 Scale
            if round_i == 0:
                # ★ 简化: ★ ★ 开启 Scale + Background, ★ 一次精修
                from gsas2_refinement.iter_agent_helpers import step_scale
                try:
                    self.hist.set_refinements({'Background': {'no. coeffs': 8, 'refine': True}})
                except Exception:
                    try:
                        self.hist.set_refinements({'Background': True})
                    except Exception:
                        pass
                step_scale(self.proj, self.phases, self.hist)
                rwp, gof, fr = self.refine_step('BG+Scale')
                self.analyse(fr)
            
            # C. Cell
            try:
                from gsas2_refinement.iter_agent_helpers import step_cell
                step_cell(self.proj, self.phases, self.hist)
                rwp, gof, fr = self.refine_step('Cell')
            except (ImportError, Exception) as e:
                # ★ 回退: 简单 GSAS-II 直接精修
                rwp, gof, fr = self.refine_step(f'Cell-fallback')
            
            # D. Size
            try:
                from gsas2_refinement.iter_agent_helpers import step_size
                dom = self._get_dominant_phase(fr)
                if dom:
                    step_size(self.proj, self.phases, self.hist, dom)
                    rwp, gof, fr = self.refine_step('Size')
            except (ImportError, Exception) as e:
                rwp, gof, fr = self.refine_step(f'Size-fallback')
            
            # E. Try add candidate
            # Always challenge the initial model in round 1. A flexible
            # background can make a target+precursors model look acceptable
            # while known competing phases are still absent.
            if rwp and (rwp > self.target_rwp + 3 or round_i == 0):
                for cand in self.extra_candidates:
                    if not self.try_add(cand):
                        continue
                    prev_rwp, prev_gof, prev_fr = rwp, gof, fr
                    try:
                        self.phases[cand].set_HAP_refinements({'Scale': True})
                    except Exception:
                        pass
                    rwp_t, _, fr_t = self.refine_step(f'+{cand}')
                    if rwp_t and (rwp - rwp_t) > 0.2:
                        rwp, gof, fr = rwp_t, _, fr_t
                        self.analyse(fr)
                    else:
                        self.kill_phase(cand, f'no improvement')
                        # Keep the accepted model state for reporting. The
                        # killed phase may still appear in an old LST parse
                        # until another refinement is run, so do not let the
                        # rejected trial fractions leak into the final answer.
                        rwp, gof, fr = prev_rwp, prev_gof, prev_fr
            
            # F. Uiso
            try:
                from gsas2_refinement.iter_agent_helpers import step_U
                step_U(self.proj, self.phases, self.hist)
                rwp, gof, fr = self.refine_step('U')
            except (ImportError, Exception) as e:
                pass
            
            # Track best
            fr = self._filter_rejected_fractions(fr)
            if rwp is not None and rwp < best_rwp:
                best_rwp, best_gof = rwp, gof
                if fr:
                    best_fr = fr
            
            print(f"  Round {round_i+1} best Rwp: {best_rwp}")
            
            # ★ Early stop ★ 当 Rwp 已达目标
            if rwp and rwp < self.target_rwp:
                print(f"  ✓ Rwp < target ({self.target_rwp}%), stop")
                break
        
        reliability = self.reliability_report(best_rwp, best_gof, best_fr)
        return {
            'sample': self.sample,
            'target_phase': self.target_phase,
            'chemistry_system': self.chemistry_system,
            'limits_2theta': self.limits,
            'best_rwp': best_rwp,
            'best_gof': best_gof,
            'best_fractions': best_fr,
            'reliability': reliability,
            'log': self.log,
            'output_dir': self.output_dir
        }
    
    def _get_dominant_phase(self, fr: Dict) -> Optional[str]:
        """★ 通用 ★ 获取主要相 (★ ★ 按 weight fraction)"""
        if not fr:
            return None
        return max(fr, key=lambda k: fr[k][0])


# ============================================================
# ★ 通用化 ★ ★ CLI 入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Chemical Synthesis Universal Iterative Rietveld Agent')
    parser.add_argument('--sample', required=True, help='Sample name')
    parser.add_argument('--target', required=True, help='Target phase, e.g. Mn2Cr3FeO8')
    parser.add_argument('--xrd', required=True, help='Path to raw XRD file')
    parser.add_argument('--system', required=True, help='Chemistry system, e.g. Mn-Cr-Fe-O')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--target-rwp', type=float, default=15.0, help='Target Rwp (%)')
    parser.add_argument('--max-rounds', type=int, default=5, help='Max rounds')
    parser.add_argument('--no-auto-download', action='store_true', help='Disable auto CIF download from MP')
    parser.add_argument('--limits', default='10,90', help='2theta limits, e.g. 20,80')
    parser.add_argument('--extra-phase', action='append', default=[], help='Additional phase name to try; can be repeated')
    args = parser.parse_args()
    
    chemistry_system = args.system.replace(',', '-').split('-')
    limits = tuple(float(x) for x in args.limits.replace(':', ',').split(',')[:2])
    
    agent = GenericIterAgent(
        sample=args.sample,
        target_phase=args.target,
        raw_xrd_path=args.xrd,
        chemistry_system=chemistry_system,
        output_dir=args.output,
        target_rwp=args.target_rwp,
        auto_download=not args.no_auto_download,
        limits=limits,
        extra_phases=args.extra_phase,
    )
    
    result = agent.run(max_rounds=args.max_rounds)
    
    print("\n" + "="*70)
    print("【★ 通用 Iterative Agent 精修结果】")
    print("="*70)
    print(f"  Sample:         {result['sample']}")
    print(f"  Target phase:   {result['target_phase']}")
    print(f"  Chemistry:      {result['chemistry_system']}")
    print(f"  Best Rwp:       {result['best_rwp']}")
    print(f"  Best GoF:       {result['best_gof']}")
    print(f"  Reliability:    {result['reliability']['label']}")
    print(f"  Quantitative:   {result['reliability']['quantification_allowed']}")
    print(f"  Output dir:     {result['output_dir']}")
    print(f"\n  ★ Phase fractions:")
    for phase, (wt, sg) in result['best_fractions'].items():
        print(f"    {phase:15s}: wt={wt*100:6.2f}%  sig={sg*100:.4f}%")
    
    # 保存
    out_path = os.path.join(result['output_dir'], 'report.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ★ 报告: {out_path}")
    
    return result


if __name__ == '__main__':
    main()
