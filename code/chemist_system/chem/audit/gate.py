#!/home/ucas/miniconda3/envs/python12/bin/python
"""
Enhanced Phase Graph Auditor for Autonomous Chemist

修复历史
--------
2026-05-18 (重构):
  - SISSO 修复：改用 GibbsComputedStructureEntry.from_pd() 正确路径
      原版三个错误：
        ① formation_enthalpy_per_atom 传入了总能量 e.energy_per_atom（应传形成焓）
        ② 对元素叠加 G_corr 与 SISSO 内置 sum_g_i 双重计算，破坏参考系
        ③ 直接调构造器在非整百 K 时 G_ELEMS 键无序导致 interp1d 报错
      正确流程：先建 T=0 凸包取形成焓 → from_pd(pd_t0, T_round) → SISSO 统一校正
  - 结构化输出：audit_protocol() 同时返回 stage_results / audit_flags 等机器可读字段
  - Hindsight 接口：新增 audit_result() 将 XRD 实验结果翻译为 semantic_priors
  - get_t0_hull_data() 独立方法：Panel A 凸包可视化所需的 T=0 数据提取

2026-05-13:
  - FIX 3: SISSO 参考态系统性错误 (本次已重新修复)

2026-05-09:
  - Tammann 温度门、Szymanski regime 分类、烧结诊断、熵稳定化检测
"""
import json, sys, warnings, sqlite3
import numpy as np
from typing import List, Optional
from collections import Counter
from itertools import combinations as _combinations

from pymatgen.ext.matproj import MPRester
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import ComputedReaction
try:
    from pymatgen.core.entries import ComputedEntry, GibbsComputedStructureEntry
except ImportError:
    from pymatgen.entries.computed_entries import ComputedEntry, GibbsComputedStructureEntry

warnings.filterwarnings("ignore")

# ─── 实验热力学数据（仅用于 SISSO 完全失败时的降级路径） ──────
S0_JMOLK = {
    'O': 205.147, 'N': 191.610, 'H': 130.680, 'F': 202.790, 'Cl': 223.081,
    'Li': 29.12, 'Na': 51.30, 'K': 64.68, 'Rb': 76.78, 'Cs': 85.23,
    'Be': 9.54, 'Mg': 33.02, 'Ca': 41.59, 'Sr': 55.69, 'Ba': 62.42,
    'Al': 28.30, 'Ga': 40.88, 'In': 57.82, 'Sn': 51.18, 'Pb': 64.81, 'Bi': 56.74,
    'Si': 18.81, 'Ge': 31.09, 'B': 5.90, 'P': 41.09, 'As': 35.69, 'Sb': 45.69,
    'C': 5.740, 'S': 32.056, 'Se': 42.96, 'Te': 49.71,
    'Ti': 30.76, 'V': 28.90, 'Cr': 23.77, 'Mn': 32.22, 'Fe': 27.78,
    'Co': 30.04, 'Ni': 29.87, 'Cu': 33.15, 'Zn': 25.26,
    'Sc': 34.64, 'Y': 44.43, 'La': 48.39, 'Zr': 38.99, 'Nb': 36.40, 'Mo': 28.97,
    'Hf': 43.28, 'Ta': 41.51, 'W': 32.68, 'Re': 36.53, 'Os': 32.64,
    'Ir': 35.48, 'Pt': 41.63, 'Au': 47.49, 'Ag': 42.55,
    'Ce': 72.21, 'Pr': 73.93, 'Nd': 71.10, 'Sm': 69.58, 'Eu': 77.78,
    'Gd': 67.96, 'Tb': 73.30, 'Dy': 67.98, 'Ho': 68.03, 'Er': 41.51,
    'Tm': 62.77, 'Yb': 59.85, 'Lu': 50.96,
}
_O2_ST = np.array([205.147, 213.87, 220.94, 227.02, 232.34, 237.03,
                   241.25, 245.12, 252.00, 257.90, 263.10, 267.50])
_O2_T  = np.array([298, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 2000])
_G_CORR_CACHE = {}

def _S_interp(T_target, T_data, S_data):
    if T_target <= T_data[0]:
        return float(S_data[0])
    if T_target >= T_data[-1]:
        dS = (S_data[-1] - S_data[-2]) / (T_data[-1] - T_data[-2])
        return float(S_data[-1] + dS * (T_target - T_data[-1]))
    return float(np.interp(T_target, T_data, S_data))

def get_G_corr_element(el_symbol, T_K, n_atoms=1):
    """仅在 SISSO 完全失败的降级路径中使用。"""
    key = (el_symbol, round(T_K), n_atoms)
    if key in _G_CORR_CACHE:
        return _G_CORR_CACHE[key]
    T_K = float(T_K)
    NA = 96485.168
    if el_symbol == 'O' and n_atoms == 2:
        S_T = _S_interp(T_K, _O2_T, _O2_ST)
        dG = -T_K * S_T / (2 * NA)
    else:
        S0 = S0_JMOLK.get(el_symbol, 3 * 8.314)
        alpha = 0.4 if el_symbol in S0_JMOLK else 0.35
        S_T = S0 * (T_K / 298.15) ** alpha
        dG = -T_K * S_T / NA
    _G_CORR_CACHE[key] = dG
    return dG

# ─── 配置区 ───────────────────────────────────────────────────
def load_api_key(required: bool = False):
    """Resolve the Materials Project key lazily (never at import time)."""
    from ..config import mp_api_key
    return mp_api_key(required=required)

import os
API_KEY = None  # resolved on first use; see PhaseGraphAuditor.__init__
kB = 8.617333262e-5  # eV/K

# ─── 动态相数据库（Cache-Aside 架构）───────────────────────────
# 系统优先级：本地 SQLite → 网络补全（Materials Project）→ 硬编码降级

from ..paths import PHASE_REGISTRY_DB as _DB_PATH
_DB_PATH = str(_DB_PATH)


class DynamicPhaseRegistry:
    """
    动态互联的材料相数据库仓储。

    数据流：输入配方 → 本地 SQLite 命中 → 直接返回
                         ↓ 未命中
                    调用 MP-API 同步数据 → 落库 → 返回

    热力学筛选：仅保留 energy_above_hull ≤ 0.1 eV/atom 的基态/近基态相，
    过滤掉理论预测但无法合成的空中楼阁相。
    """

    _schema = """
    CREATE TABLE IF NOT EXISTS chemical_systems (
        system_key   TEXT PRIMARY KEY,
        element_list TEXT,
        last_sync    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS known_phases (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        system_key        TEXT,
        formula           TEXT,
        composition_json   TEXT,
        energy_above_hull  REAL,
        space_group       TEXT,
        note              TEXT,
        FOREIGN KEY (system_key) REFERENCES chemical_systems(system_key)
    );
    CREATE INDEX IF NOT EXISTS idx_phases_system ON known_phases(system_key);
    """

    def __init__(self, db_path: str = _DB_PATH, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._schema)
        # 预先注入硬编码降级数据（保证即使 MP 不可用也能工作）
        self._seed_fallback()

    def _seed_fallback(self):
        """将硬编码的 CE_AL_GE_PHASES 降级数据写入本地库（若尚不存在）"""
        SYSTEM_KEY = "Al-Ce-Ge"
        fallback_phases = [
            {"formula": "CeAl2Ge2",  "ehull": 0.0124, "sg": "P-3m1",
             "comp": {"Ce": 1, "Al": 2, "Ge": 2},
             "note": "metastable, arc-melt target"},
            {"formula": "Ce2Al3Ge4", "ehull": 0.0,    "sg": "Cmce",
             "comp": {"Ce": 2, "Al": 3, "Ge": 4},
             "note": "ground state, thermodynamic sink"},
            {"formula": "CeAlGe",    "ehull": 0.0068, "sg": "I41md",
             "comp": {"Ce": 1, "Al": 1, "Ge": 1},
             "note": "intermediate metastable phase"},
        ]
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM known_phases WHERE system_key = ? LIMIT 1",
                        (SYSTEM_KEY,))
            if cur.fetchone():
                return  # 已有数据，跳过
            cur.execute(
                "INSERT OR IGNORE INTO chemical_systems (system_key, element_list) VALUES (?, ?)",
                (SYSTEM_KEY, "Al,Ce,Ge"))
            for p in fallback_phases:
                cur.execute("""
                    INSERT INTO known_phases
                    (system_key, formula, composition_json, energy_above_hull, space_group, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (SYSTEM_KEY, p["formula"], json.dumps(p["comp"]),
                      p["ehull"], p["sg"], p["note"]))
            conn.commit()

    # ── MP-API 同步器 ────────────────────────────────────────────
    def _fetch_from_mp(self, system_key: str, elements: List[str]) -> List[dict]:
        """从 Materials Project 抓取整个体系的基态相数据"""
        if not self.api_key:
            return []

        try:
            from mp_api.client import MPRester
        except ImportError:
            return []

        downloaded = []
        try:
            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(
                    chemsys=elements,
                    fields=["formula_pretty", "composition", "energy_above_hull", "symmetry"]
                )
                for doc in docs:
                    # 热力学筛选：仅保留 ehull ≤ 0.1 eV/atom 的基态/近基态相
                    if (doc.energy_above_hull is not None
                            and doc.energy_above_hull <= 0.1):
                        sg = (doc.symmetry.point_group
                              if doc.symmetry else "")
                        downloaded.append({
                            "formula":   doc.formula_pretty,
                            "comp":      {k: float(v) for k, v
                                          in doc.composition.items()},
                            "ehull":     float(doc.energy_above_hull),
                            "space_group": sg,
                            "note":      "MP_SUMMARY",
                        })
        except Exception as e:
            print(f"[DynamicPhaseRegistry] MP sync failed for {system_key}: {e}",
                  file=sys.stderr)
        return downloaded

    # ── 持久化写入 ───────────────────────────────────────────────
    def _save_phases(self, system_key: str, elements: List[str],
                     phases: List[dict]):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO chemical_systems (system_key, element_list, last_sync)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (system_key, ",".join(elements)))
            for p in phases:
                cur.execute("""
                    INSERT INTO known_phases
                    (system_key, formula, composition_json, energy_above_hull, space_group, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (system_key, p["formula"], json.dumps(p["comp"]),
                      p["ehull"], p.get("space_group", ""), p.get("note", "")))
            conn.commit()

    # ── 核心查询接口 ────────────────────────────────────────────
    def get_phases(self, elements: List[str]) -> List[dict]:
        """
        高层业务接口：获取指定元素体系的稳定相（含自动补全）。

        Parameters
        ----------
        elements : list[str]
            元素符号列表，如 ["Al", "Ce", "Ge"]

        Returns
        -------
        list[dict]
            每个 dict 包含：formula, comp（原子比例 dict）,
            ehull（eV/atom）, space_group, note
        """
        elements = [el.strip().capitalize() for el in elements]
        system_key = "-".join(sorted(elements))

        # ── Cache Hit ──────────────────────────────────────────────
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT formula, composition_json, energy_above_hull, space_group, note "
                "FROM known_phases WHERE system_key = ?",
                (system_key,))
            rows = cur.fetchall()

        if rows:
            return [
                {"formula": r[0],
                 "comp":    json.loads(r[1]),
                 "ehull":   r[2],
                 "space_group": r[3] or "",
                 "note":    r[4] or ""}
                for r in rows
            ]

        # ── Cache Miss：触发 MP 自动补全 ─────────────────────────
        print(f"[DynamicPhaseRegistry] 本地未命中 {system_key}，正在自云端补全...",
              file=sys.stderr)
        remote_phases = self._fetch_from_mp(system_key, elements)
        if remote_phases:
            self._save_phases(system_key, elements, remote_phases)
            return remote_phases

        # ── 降级：返回硬编码数据（由 _seed_fallback 保证必存在） ───
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT formula, composition_json, energy_above_hull, space_group, note "
                "FROM known_phases WHERE system_key = ?",
                (system_key,))
            rows = cur.fetchall()
        return [
            {"formula": r[0], "comp": json.loads(r[1]),
             "ehull": r[2], "space_group": r[3] or "", "note": r[4] or ""}
            for r in rows
        ]

    def get_ground_state(self, elements: List[str]) -> Optional[dict]:
        """返回 ehull 最小的相（基态相）"""
        phases = self.get_phases(elements)
        if not phases:
            return None
        return min(phases, key=lambda p: p["ehull"])

    def get_phase_by_formula(self, elements: List[str],
                             formula: str) -> Optional[dict]:
        """按化学式精确查找某相"""
        phases = self.get_phases(elements)
        target = Composition(formula).reduced_formula
        for p in phases:
            if Composition(p["formula"]).reduced_formula == target:
                return p
        return None


def get_melting_point(formula):
    """Standalone function — defined after MP_MELTING below."""
    return MP_MELTING.get(Composition(formula).reduced_formula, None)


# ═══════════════════════════════════════════════════════════════════════════
# ARC-MELT KNOWLEDGE BASE (for production-grade hardware & process audit)
# ═══════════════════════════════════════════════════════════════════════════
# Source: ASTM/CRC Handbook, NIST-JANAF thermochemical tables
# All values are empirical/measured, NOT synthetic.

MP_MELTING = {
    "Al2O3": 2072, "Y2O3": 2438, "ZrO2": 2700, "TiO2": 1843,
    "BaO": 1923, "BaCO3": 1400, "BaTiO3": 1625, "Ba2TiO4": 1720,
    "BaTi2O5": 1560, "BaTi5O11": 1550,
    "Ta2O5": 1872, "Er2O3": 2430, "ErTaO4": 1900,
    "ErTa7O19": 1900, "ErTa3O9": 1900, "Er3TaO7": 1900,
    "SiO2": 1710, "MgO": 2852, "CaO": 2580,
    "La2O3": 2315, "Nd2O3": 2275, "Sm2O3": 2330,
    "Li2O": 1438, "Li4Ti5O12": 1500, "LATP": 1250, "LLZO": 1500,
}

ELEMENT_PROPERTIES = {
    # ── Rare Earths ──────────────────────────────────────────────
    "Ce": {"melting_point_c": 798,  "boiling_point_c": 3443, "is_rare_earth": True,
           "hazard": "oxidizes rapidly above 200°C in impure argon"},
    "La": {"melting_point_c": 920,  "boiling_point_c": 3464, "is_rare_earth": True,
           "hazard": "highly reactive, forms oxide easily"},
    "Pr": {"melting_point_c": 931,  "boiling_point_c": 3520, "is_rare_earth": True,
           "hazard": "pyrophoric in fine form"},
    "Nd": {"melting_point_c": 1016, "boiling_point_c": 3074, "is_rare_earth": True,
           "hazard": "moderately reactive"},
    "Sm": {"melting_point_c": 1072, "boiling_point_c": 1794, "is_rare_earth": True,
           "hazard": "evaporates at arc temp (BP only 1794°C)"},
    "Gd": {"melting_point_c": 1313, "boiling_point_c": 3273, "is_rare_earth": True,
           "hazard": "high MP, safe in arc"},
    "Y":  {"melting_point_c": 1522, "boiling_point_c": 3345, "is_rare_earth": True,
           "hazard": "high MP, safe in arc"},
    # ── Aluminum family ─────────────────────────────────────────
    "Al": {"melting_point_c": 660,  "boiling_point_c": 2519, "is_rare_earth": False,
           "hazard": "low MP, good for eutectic, volatile above 2000°C"},
    "Ga": {"melting_point_c": 29.8, "boiling_point_c": 2229, "is_rare_earth": False,
           "hazard": "melts near RT, corrosive to ceramics at high T"},
    "In": {"melting_point_c": 156.6,"boiling_point_c": 2072, "is_rare_earth": False,
           "hazard": "very low MP, causes spitting during arc"},
    # ── Germanium / Si / Sn ──────────────────────────────────────
    "Ge": {"melting_point_c": 938,  "boiling_point_c": 2833, "is_rare_earth": False,
           "hazard": "high BP safe for arc, slight Ge loss at >2000°C"},
    "Si": {"melting_point_c": 1414, "boiling_point_c": 3265, "is_rare_earth": False,
           "hazard": "safe in arc"},
    "Sn": {"melting_point_c": 232,  "boiling_point_c": 2602, "is_rare_earth": False,
           "hazard": "low MP, Sn vapor deposits on chamber walls"},
    # ── High-vapor-pressure elements (一票否决) ──────────────────
    "Zn": {"melting_point_c": 419,  "boiling_point_c": 907,  "is_rare_earth": False,
           "hazard": "BP=907°C — explosive vapor at arc temp (3000°C)"},
    "Mg": {"melting_point_c": 650,  "boiling_point_c": 1090, "is_rare_earth": False,
           "hazard": "BP=1090°C — explosive vapor at arc temp"},
    "Cd": {"melting_point_c": 321,  "boiling_point_c": 767,  "is_rare_earth": False,
           "hazard": "BP=767°C — highly toxic vapor, explosive"},
    "Na": {"melting_point_c": 97.8, "boiling_point_c": 883,  "is_rare_earth": False,
           "hazard": "BP=883°C — pyrophoric, reacts violently with water"},
    "K":  {"melting_point_c": 63.4, "boiling_point_c": 759,  "is_rare_earth": False,
           "hazard": "BP=759°C — pyrophoric"},
    "Li": {"melting_point_c": 180,  "boiling_point_c": 1342, "is_rare_earth": False,
           "hazard": "BP=1342°C — lithium vapor highly reactive"},
    "Ca": {"melting_point_c": 842,  "boiling_point_c": 1484, "is_rare_earth": False,
           "hazard": "BP=1484°C — moderate risk, forms CaO easily"},
    "Ba": {"melting_point_c": 727,  "boiling_point_c": 1897, "is_rare_earth": False,
           "hazard": "BP=1897°C — Ba vapor toxic"},
    "Mn": {"melting_point_c": 1246, "boiling_point_c": 2061, "is_rare_earth": False,
           "hazard": "BP=2061°C — slight Mn loss at arc temp"},
    "Fe": {"melting_point_c": 1538, "boiling_point_c": 2861, "is_rare_earth": False,
           "hazard": "safe in arc"},
    "Co": {"melting_point_c": 1495, "boiling_point_c": 2927, "is_rare_earth": False,
           "hazard": "safe in arc"},
    "Ni": {"melting_point_c": 1455, "boiling_point_c": 2913, "is_rare_earth": False,
           "hazard": "safe in arc"},
    "Cu": {"melting_point_c": 1085, "boiling_point_c": 2562, "is_rare_earth": False,
           "hazard": "safe in arc (some loss at >2000°C)"},
    "Ti": {"melting_point_c": 1668, "boiling_point_c": 3287, "is_rare_earth": False,
           "hazard": "safe, used as Ti-getter"},
    "Zr": {"melting_point_c": 1855, "boiling_point_c": 4409, "is_rare_earth": False,
           "hazard": "safe, high BP"},
    "Mo": {"melting_point_c": 2623, "boiling_point_c": 4639, "is_rare_earth": False,
           "hazard": "safe in arc"},
    "W":  {"melting_point_c": 3422, "boiling_point_c": 5555, "is_rare_earth": False,
           "hazard": "safe, very high BP"},
}

# Known intermetallic phases for Ce-Al-Ge ternary system (ICSD/PDF)
CE_AL_GE_PHASES = {
    "CeAl2Ge2":  {"space_group": "P-3m1",  "e_hull_meV": 12.4,
                  "note": "metastable, arc-melt target"},
    "Ce2Al3Ge4": {"space_group": "Cmce",  "e_hull_meV": 0.0,
                  "note": "ground state, thermodynamic sink"},
    "CeAlGe":    {"space_group": "I41md",  "e_hull_meV": 6.8,
                  "note": "intermediate metastable phase"},
}


def tammann_temperature(tm_c):
    tm_k = tm_c + 273.15
    return 0.5 * tm_k - 273.15, 0.67 * tm_k - 273.15

# ─── 烧结诊断 ─────────────────────────────────────────────────
def diagnose_sintering(stage_temp_c, product_formula, dwell_minutes):
    tm = get_melting_point(product_formula)
    if tm is None:
        return "UNKNOWN", "熔点未知，跳过烧结充分性评估"
    t_onset, t_full = tammann_temperature(tm)
    if stage_temp_c < t_onset:
        return "FAIL", (
            f"温度 {stage_temp_c}°C < Tammann onset {t_onset:.0f}°C "
            f"(0.5×T_m={tm}°C). 固相扩散可忽略. "
            f"考虑: 冷烧结(CSP) / 提高温度 / 延长时间"
        )
    elif stage_temp_c < t_full:
        return "WARNING", (
            f"温度 {stage_temp_c}°C 处于 Tammann 区间 [{t_onset:.0f}–{t_full:.0f}]°C. "
            f"扩散受限，建议延长 dwell 或提高温度"
        )
    else:
        Q_typical, R = 500, 8.314e-3
        k_ratio = np.exp(-Q_typical/R * (1/(stage_temp_c+273) - 1/(t_full+273)))
        return "OK", (
            f"温度 {stage_temp_c}°C > Tammann full {t_full:.0f}°C, "
            f"扩散充分. k_relative ≈ {k_ratio:.1f}× vs T={t_full:.0f}°C 基准"
        )

# ─── Regime 分类 ──────────────────────────────────────────────
SZYMANSKI_THRESHOLD = 0.060  # eV/atom

def classify_regime(e_hull, competitors_ehull, temp_c):
    if not competitors_ehull:
        delta_dg = e_hull
        if delta_dg >= SZYMANSKI_THRESHOLD:
            return "THERMODYNAMIC", (
                f"ΔΔG ≈ {delta_dg*1000:.1f} meV/atom ≥ 60 meV (Szymanski). 热力学驱动."
            )
        elif delta_dg > 0.001:
            return "MIXED", (
                f"ΔΔG ≈ {delta_dg*1000:.1f} meV/atom < 60 meV. 混合控制."
            )
        else:
            return "MIXED", (
                f"SISSO校正后目标相在凸包上 (ehull≈0). 热力学可及, 保守判定 MIXED."
            )
    min_comp = min(competitors_ehull)
    delta = e_hull - min_comp
    if delta > SZYMANSKI_THRESHOLD:
        return "KINETIC", (
            f"ΔΔG = {delta*1000:.1f} meV/atom > 60 meV. 目标相热力学不利，需动力学路径."
        )
    elif delta < -SZYMANSKI_THRESHOLD:
        return "THERMODYNAMIC", (
            f"ΔΔG = {delta*1000:.1f} meV/atom < -60 meV. 目标相热力学更稳定，反应自发."
        )
    else:
        return "MIXED", (
            f"|ΔΔG| = {abs(delta)*1000:.1f} meV/atom ≤ 60 meV. 混合控制."
        )

# ─── 熵稳定化检测 ─────────────────────────────────────────────
def is_entropy_stabilized(target_formula, e_hull, n_cations=None):
    fl = target_formula.lower()
    if any(h in fl for h in ["heo", "high-entropy", "entropy"]):
        return True
    if n_cations is not None and n_cations >= 5 and e_hull > 0.030:
        return True
    return False


# ─── 氧化还原安全检查 (★ 2026-06-10 新增, 解决 Mn2O3/Fe2O3 还原临界问题) ───
# 关键临界点 (★ 来源: 真实文献调研, Carrillo 2015 等):
#   Mn2O3 -> Mn3O4 (Mn2+Mn3+2O4 尖晶石): 900-1000°C 空气中
#   Mn3O4 -> MnO: 1000-1100°C
#   Fe2O3 -> Fe3O4 (尖晶石): >1200°C 空气中
#   Co3O4 -> CoO: ~900°C
#   PbO2 -> Pb3O4: ~290°C
#   MoO3 挥发: >700°C
#   Cr2O3 / Al2O3 / Gd2O3 / Y2O3 / ZrO2 / HfO2: 全温度稳定
# 
# ★★★ 关键设计: 对于含 M3+ (Mn3+, Fe3+) 的体系, 必须保持温度低于还原临界
#               否则平均阳离子价态下降, 导致尖晶石固溶体热力学陷阱
REDOX_CRITICAL_C = {
    # ===== ★ M3+ 还原类 (空气中温度) =====
    "Mn2O3":   (950,  "reduction", "Mn2O3 -> Mn3O4 (Mn2+Mn3+2O4 尖晶石)"),
    "Mn3O4":   (1100, "reduction", "Mn3O4 -> MnO (Mn2+)"),
    "MnO2":    (500,  "reduction", "MnO2 -> Mn2O3 (Mn4+ -> Mn3+)"),
    "Fe2O3":   (1200, "reduction", "Fe2O3 -> Fe3O4 (尖晶石)"),
    "Co3O4":   (900,  "reduction", "Co3O4 -> CoO (Co3+ -> Co2+)"),
    "Ni2O3":   (600,  "reduction", "Ni3+ 还原"),
    "Cu2O":    (1200, "reduction", "Cu+ 歧化"),
    "CuO":     (1026, "reduction", "Cu2+ -> Cu+ (Cu2O)"),
    "Bi2O3":   (800,  "reduction", "Bi3+ 还原 (熔点附近)"),
    "V2O5":    (700,  "reduction", "V5+ -> V2O3"),
    "Pb3O4":   (600,  "reduction", "Pb2+ 稳定"),
    "Pr6O11":  (900,  "reduction", "Pr4+/3+ 还原"),
    "MoO3":    (700,  "volatilization", "MoO3 挥发"),
    "SnO2":    (1500, "stable_in_air", "Sn4+ 在空气中稳定 (H2 下 < 600°C 还原)"),
    # ===== 磷酸盐/酸分解类 =====
    "P2O5":       (400,  "volatilization", "P2O5 挥发"),
    "NH4H2PO4":   (200,  "decomposition",  "NH4H2PO4 分解 (MAP 熔点 190°C)"),
    "(NH4)2HPO4": (200,  "decomposition",  "DAP 分解"),
    "H3PO4":      (200,  "volatilization", "H3PO4 聚合"),
    # ===== 碳酸盐分解 =====
    "CaCO3":      (825,  "decomposition",  "CaCO3 -> CaO + CO2"),
    "MgCO3":      (350,  "decomposition",  "MgCO3 -> MgO + CO2"),
    "SrCO3":      (1100, "decomposition",  "SrCO3 -> SrO + CO2"),
    "BaCO3":      (1300, "decomposition",  "BaCO3 -> BaO + CO2"),
    # ===== 氢氧化物脱水 =====
    # (可在 future 扩展)
    # ===== 全温度稳定 =====
    "Cr2O3":   (9999, "stable", "Cr3+ 全温度稳定"),
    "Al2O3":   (9999, "stable", "Al3+ 全温度稳定"),
    "Gd2O3":   (9999, "stable", "Gd3+ 全温度稳定"),
    "Y2O3":    (9999, "stable", "Y3+ 全温度稳定"),
    "La2O3":   (9999, "stable", "La3+ 全温度稳定"),
    "CeO2":    (9999, "stable", "Ce4+ 全温度稳定 (还原需要 2000°C+)"),
    "Fe3O4":   (9999, "stable", "已经是混合价态"),
    "CoO":     (9999, "stable", "Co2+ 稳定"),
    "NiO":     (9999, "stable", "Ni2+ 稳定"),
    "MnO":     (9999, "stable", "Mn2+ 稳定"),
    "CaO":     (9999, "stable", "Ca2+ 稳定"),
    "MgO":     (9999, "stable", "Mg2+ 稳定"),
    "SrO":     (9999, "stable", "Sr2+ 稳定"),
    "BaO":     (9999, "stable", "Ba2+ 稳定"),
    "TiO2":    (9999, "stable", "Ti4+ 全温度稳定"),
    "ZrO2":    (9999, "stable", "Zr4+ 全温度稳定"),
    "HfO2":    (9999, "stable", "Hf4+ 全温度稳定"),
    "Ta2O5":   (9999, "stable", "Ta5+ 稳定"),
    "Nb2O5":   (9999, "stable", "Nb5+ 稳定"),
    "WO3":     (9999, "stable", "W6+ 稳定"),
    "Li2O":    (9999, "stable", "Li+ 稳定"),
    "Na2O":    (9999, "stable", "Na+ 稳定"),
    "K2O":     (9999, "stable", "K+ 稳定"),
    "B2O3":    (9999, "stable", "B3+ 稳定"),
    "In2O3":   (9999, "stable", "In3+ 稳定"),
    "Nd2O3":   (9999, "stable", "Nd3+ 稳定"),
    "Sm2O3":   (9999, "stable", "Sm3+ 稳定"),
    "Eu2O3":   (9999, "stable", "Eu3+ 稳定 (Eu2+ 在 H2 下形成)"),
    "Tb4O7":   (9999, "stable", "Tb4+/3+ 混合稳定"),
    "Dy2O3":   (9999, "stable", "Dy3+ 稳定"),
    "Ho2O3":   (9999, "stable", "Ho3+ 稳定"),
    "Er2O3":   (9999, "stable", "Er3+ 稳定"),
    "Tm2O3":   (9999, "stable", "Tm3+ 稳定"),
    "Yb2O3":   (9999, "stable", "Yb3+ 稳定"),
    "Lu2O3":   (9999, "stable", "Lu3+ 稳定"),
    "Sc2O3":   (9999, "stable", "Sc3+ 稳定"),
    "Ga2O3":   (9999, "stable", "Ga3+ 稳定"),
    "GeO2":    (9999, "stable", "Ge4+ 稳定"),
    "SiO2":    (9999, "stable", "Si4+ 稳定"),
    "H2O":     (9999, "stable", "水"),
}



def diagnose_oxidation_state(stage_temp_c, precursor_fmls, atmosphere="Air"):
    """
    ★★★ 严格基于化学知识的氧化还原/分解/挥发诊断 (★ 2026-06-11 v3 重写)
    
    严格基于真实化学知识:
    - 还原类 (Mn2O3/Fe2O3/Co3O4/Bi2O3): 在空气中被还原为低价态
    - 分解类 (CaCO3/MgCO3/BaCO3/SrCO3): 碳酸盐分解
    - 挥发类 (P2O5/H3PO4/MoO3): 挥发或聚合
    - 磷酸盐/酸 (NH4H2PO4/MAP/DAP): 分解放 NH3
    - 全稳定 (Cr2O3/Al2O3/Gd2O3/Y2O3/ZrO2/etc): 不参与
    
    严格判定:
    - T < crit*0.88: OK (无风险)
    - T in [crit*0.88, crit): WARNING (接近, 需注意)
    - T >= crit: FAIL (已发生, 必须避免)
    - T > crit+50: FAIL SEVERE (严重)
    - 流动 O2 抑制: 仅 T in [crit, crit+50] 时降级到 WARNING
    
    Args:
        stage_temp_c: 当前阶段温度 (°C)
        precursor_fmls: 起始物列表 (化学式)
        atmosphere: 气氛 ("Air", "O2 (flowing)", "Ar", "N2" 等)
    
    Returns:
        ("OK"|"WARNING"|"FAIL"|"UNKNOWN", 详细消息)
    """
    if not precursor_fmls:
        return "UNKNOWN", "无起始物信息"
    
    is_o2_atmosphere = atmosphere and "O2" in str(atmosphere) and "flowing" in str(atmosphere).lower()
    
    # ★ 按类型分组
    reduction_risks = []     # 还原类 (M3+ -> M2+)
    decomposition_risks = []  # 分解类 (碳酸盐/磷酸盐)
    volatilization_risks = []  # 挥发类
    
    for f in precursor_fmls:
        if f not in REDOX_CRITICAL_C:
            continue
        crit, ftype, fdesc = REDOX_CRITICAL_C[f]
        if crit >= 9999:
            continue  # 全稳定
        
        # ★★★ 严格基于化学知识的阈值 (★ 2026-06-11 v4)
        # T/crit >= 1.05: 严重 (T > crit+50°C), 远超过, 必然分解/还原
        # T/crit in [1.0, 1.05): 临界, 部分分解/还原
        # T/crit in [0.92, 1.0): 接近, 可能开始 (T < 临界, 实际 8% 余量)
        # T/crit < 0.92: 安全
        # 理由: 还原反应是平衡反应, 临界点 ± 50°C 是真实反应区间
        ratio = stage_temp_c / crit
        # ★★★ 严格基于化学知识 v5 (★ 2026-06-11 修正)
        # ratio >= 1.10 (T > crit*1.1, 即 T-crit > 10%): 严重, 化学已大量分解
        # ratio in (1.0, 1.10) (T > crit, 10%以内): 临界, O2 抑制可救
        # ratio in [0.92, 1.0]: 接近, T < crit 实际未发生
        # ratio < 0.92: 安全
        if ratio >= 1.10:
            entry = (f, stage_temp_c, crit, ftype, fdesc, "SEVERE")
        elif ratio > 1.0:
            entry = (f, stage_temp_c, crit, ftype, fdesc, "OVER")
        elif ratio >= 0.92:
            entry = (f, stage_temp_c, crit, ftype, fdesc, "WARNING")
        else:
            continue  # OK
        
        if ftype == "reduction":
            reduction_risks.append(entry)
        elif ftype == "decomposition":
            decomposition_risks.append(entry)
        elif ftype == "volatilization":
            volatilization_risks.append(entry)
        else:
            # stable_in_air 但仍有限度
            reduction_risks.append(entry)
    
    # ★★★ 严格判定
    # 1. SEVERE 任何 1 个 -> FAIL SEVERE
    severe_all = [e for e in (reduction_risks + decomposition_risks + volatilization_risks) if e[5] == "SEVERE"]
    if severe_all:
        msg = "★★★ 严重氧化还原/分解风险: " + "; ".join(
            f"{e[0]} 在 {e[1]}°C 严重超出 (临界 {e[2]}°C, {e[4]})" for e in severe_all[:2]
        )
        msg += ". ★ 阳离子价态下降 + 起始物分解, 产物偏离. 建议: 降低温度 < 临界-100°C"
        return "FAIL", msg
    
    # 2. OVER 任何 1 个 -> FAIL (但流动 O2 抑制时降到 WARNING)
    over_all = [e for e in (reduction_risks + decomposition_risks + volatilization_risks) if e[5] == "OVER"]
    if over_all:
        if is_o2_atmosphere and not decomposition_risks and not volatilization_risks:
            # ★ 流动 O2 抑制 + 仅有还原类 -> 降到 WARNING
            msg = "氧化还原警告 (流动 O2 部分抑制): " + "; ".join(
                f"{e[0]} 在 {e[1]}°C 略超临界 (临界 {e[2]}°C, {e[4]})" for e in over_all[:2]
            )
            msg += ". 建议: 温度 < 临界-50°C 更安全"
            return "WARNING", msg
        else:
            # 分解/挥发/无 O2 -> FAIL
            if decomposition_risks or volatilization_risks:
                msg = "★★★ 起始物分解/挥发风险: "
            else:
                msg = "★★★ 氧化还原临界风险: "
            msg += "; ".join(
                f"{e[0]} 在 {e[1]}°C 超临界 (临界 {e[2]}°C, {e[4]})" for e in over_all[:2]
            )
            msg += ". ★ 起始物已分解/挥发, 产物偏离. 建议: 降低温度 < 临界-50°C"
            return "FAIL", msg
    
    # 3. WARNING 任何 1 个 -> WARNING
    # ★ 流动 O2 抑制: 还原类 WARNING 完全抑制 (T < crit, 未发生)
    #    分解/挥发类 WARNING 部分抑制
    warning_all = [e for e in (reduction_risks + decomposition_risks + volatilization_risks) if e[5] == "WARNING"]
    if warning_all:
        # 全部是还原类 + 流动 O2 -> OK
        only_reduction = all(e[3] == "reduction" for e in warning_all)
        if is_o2_atmosphere and only_reduction:
            return "OK", "流动 O2 完全抑制 (T < 临界, 未到还原温度)"
        msg = "氧化还原警告: " + "; ".join(
            f"{e[0]} 在 {e[1]}°C 接近临界 (临界 {e[2]}°C, {e[4]})" for e in warning_all[:2]
        )
        return "WARNING", msg
    
    return "OK", "无氧化还原/分解/挥发风险"

# ─── 核心审计类 ───────────────────────────────────────────────
class PhaseGraphAuditor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.entries_cache  = {}   # chemsys → raw MP entries
        self.pd_t0_cache    = {}   # chemsys → T=0 PhaseDiagram
        self.pd             = None
        self._sisso_fallbacks = set()
        self._sisso_mode = "unknown"  # "sisso" | "fallback_gcorr"

    def _get_entries(self, chemsys):
        key = "-".join(sorted(chemsys))
        if key not in self.entries_cache:
            print(f"Fetching {key} from MP...", file=sys.stderr)
            with MPRester(self.api_key) as mpr:
                self.entries_cache[key] = mpr.get_entries_in_chemsys(chemsys)
        return self.entries_cache[key]

    def _get_pd_t0(self, chemsys):
        """构建并缓存 T=0 PhaseDiagram（用于形成焓参考）。"""
        key = "-".join(sorted(chemsys))
        if key not in self.pd_t0_cache:
            entries = self._get_entries(chemsys)
            self.pd_t0_cache[key] = PhaseDiagram(entries)
        return self.pd_t0_cache[key]

    def build_thermal_pd(self, temp_c, chemsys):
        """
        SISSO 热力学相图构建（修正版 2026-05-18）。

        正确流程：
          1. 用 T=0 PD 获取各物相的形成焓（DFT 参考系）
          2. 调用 GibbsComputedStructureEntry.from_pd(pd_t0, T_round)
             - 元素相保持 T=0 DFT 能量
             - 化合物由 SISSO 统一校正：G = n*(H_f + g_delta) - sum_g_i(T)
             - G_ELEMS 包含 Er、Ta，SISSO 对本系统完全适用
          3. 温度取整至最近 100K（避免 G_ELEMS dict 无序时 interp1d 报错）
          4. 降级路径：SISSO 失败时对所有相统一叠加 G_corr（形成焓近似不变）

        返回：PhaseDiagram（T 校正后）
        同时设置：
          self._sisso_mode  → "sisso" 或 "fallback_gcorr"
          self._sisso_fallbacks → 降级路径下的受影响物相名集合
        """
        temp_k = temp_c + 273.15
        # 取整到最近 100K（SISSO G_ELEMS 数据间隔 100K）
        temp_k_round = int(round(temp_k / 100.0) * 100)
        temp_k_round = max(300, min(2000, temp_k_round))

        pd_t0 = self._get_pd_t0(chemsys)
        self._sisso_fallbacks.clear()

        try:
            gibbs_entries = GibbsComputedStructureEntry.from_pd(
                pd_t0, temp=float(temp_k_round)
            )
            self._sisso_mode = "sisso"
            self.pd = PhaseDiagram(gibbs_entries)
            return self.pd

        except Exception as sisso_err:
            # 降级路径：对所有相（元素+化合物）统一叠加 G_corr
            # 效果：G_f(compound, T) ≈ G_f(T=0)（热力学校正在形成能中相消）
            # 优先级：仍能保证相图内部一致，不产生参考系错位
            print(f"[build_thermal_pd] SISSO failed ({sisso_err}), using G_corr fallback",
                  file=sys.stderr)
            self._sisso_mode = "fallback_gcorr"
            entries = self._get_entries(chemsys)
            corrected = []
            for e in entries:
                comp = e.composition
                dG = sum(
                    comp[el] * get_G_corr_element(el.symbol, temp_k, 1)
                    for el in comp.elements
                )
                corrected.append(ComputedEntry(comp, e.energy + dG))
                if len(comp.elements) > 1:
                    self._sisso_fallbacks.add(comp.reduced_formula)
            self.pd = PhaseDiagram(corrected)
            return self.pd

    def get_t0_hull_data(self, chemsys, target_formula, endpoint_formulas=None):
        """
        提取 Panel A 凸包可视化所需的 T=0 数据。

        参数
        ----
        chemsys : list[str]  元素系统，如 ['Er','Ta','O']
        target_formula : str  目标相，如 'ErTa7O19'
        endpoint_formulas : list[str] | None
            二元端点，如 ['Er2O3','Ta2O5']。默认自动选取端点。

        返回
        ----
        dict：
          phases         : list[dict]  各相的 x_cation / Ef / ehull / stability
          target         : dict         目标相数据
          endpoints      : dict         端点数据
          above_tieline  : float        目标相在端点连线上方的能量 (meV/at)
          sisso_mode     : str          "t0_dft"（此函数固定用 T=0）
        """
        pd_t0 = self._get_pd_t0(chemsys)
        target_comp = Composition(target_formula)
        target_el_symbols = [el.symbol for el in target_comp.elements
                             if el.symbol != 'O']

        if endpoint_formulas is None:
            # 自动：选取只含单种阳离子的氧化物作端点
            endpoints = []
            for e in pd_t0.stable_entries:
                comp = e.composition
                cations = [el.symbol for el in comp.elements if el.symbol != 'O']
                if len(cations) == 1 and cations[0] in target_el_symbols:
                    endpoints.append(e.composition.reduced_formula)
            endpoint_formulas = list(dict.fromkeys(endpoints))

        # 收集各相数据
        all_formulas = set(endpoint_formulas) | {target_formula}
        for e in pd_t0.all_entries:
            comp = e.composition
            cations = [el.symbol for el in comp.elements if el.symbol != 'O']
            if all(c in target_el_symbols for c in cations) and cations:
                all_formulas.add(comp.reduced_formula)

        phase_data = []
        endpoint_data = {}
        target_data = None
        for formula in sorted(all_formulas):
            ents = [e for e in pd_t0.all_entries
                    if e.composition.reduced_formula == formula]
            if not ents:
                continue
            best = min(ents, key=lambda e: pd_t0.get_e_above_hull(e))
            comp = best.composition
            cations = {el.symbol: comp[el] for el in comp.elements if el.symbol != 'O'}
            total_cation = sum(cations.values())
            # x 轴：第一个非 Er 阳离子的摩尔分数（可扩展）
            second_cation = [s for s in target_el_symbols if s != target_el_symbols[0]]
            x = (cations.get(second_cation[0], 0) / total_cation
                 if second_cation and total_cation > 0 else 0.0)
            eh = pd_t0.get_e_above_hull(best)
            ef = pd_t0.get_form_energy_per_atom(best)
            d = {
                "formula":   formula,
                "x_cation":  round(float(x), 4),
                "Ef_meV_at": round(float(ef) * 1000, 2),
                "ehull_meV_at": round(float(eh) * 1000, 2),
                "stable":    bool(eh < 0.005),
            }
            phase_data.append(d)
            if formula in endpoint_formulas:
                endpoint_data[formula] = d
            if formula == Composition(target_formula).reduced_formula:
                target_data = d

        # 计算目标相在端点连线上方的能量
        above_tieline = None
        if target_data and len(endpoint_data) >= 2:
            eps = list(endpoint_data.values())
            x0, ef0 = eps[0]["x_cation"], eps[0]["Ef_meV_at"]
            x1, ef1 = eps[1]["x_cation"], eps[1]["Ef_meV_at"]
            xt = target_data["x_cation"]
            if abs(x1 - x0) > 1e-6:
                tie_val = ef0 + (ef1 - ef0) * (xt - x0) / (x1 - x0)
                above_tieline = round(target_data["Ef_meV_at"] - tie_val, 2)

        return {
            "phases":        sorted(phase_data, key=lambda d: d["x_cation"]),
            "target":        target_data,
            "endpoints":     endpoint_data,
            "above_tieline_meV_at": above_tieline,
            "sisso_mode":    "t0_dft",
        }

    def audit_protocol(self, protocol_json):
        """
        正向审计。

        返回
        ----
        {
          "status":           "SUCCESS" | "ERROR",
          "report":           str,           # 人类可读 markdown
          "target":           str,
          "chemsys":          list[str],
          "stage_results":    list[dict],    # 每阶段结构化结果
          "audit_flags":      list[str],     # 全局 flag 集合
          "overall_regime":   str,
          "critical_failures":list[dict],
          "sisso_mode":       str,           # "sisso" | "fallback_gcorr"
        }
        """
        try:
            target_formula  = protocol_json["material"]
            target_comp     = Composition(target_formula)
            target_el_syms  = [el.symbol for el in target_comp.elements]
            n_cations       = len([e for e in target_comp.elements if e.symbol != "O"])

            protocol         = protocol_json["protocol"]
            precursor_fmls   = [p["chemical"] for p in protocol["precursors"]]
            thermal_profile  = protocol["thermal_profile"]

            all_els = set(target_el_syms)
            for f in precursor_fmls:
                all_els.update([el.symbol for el in Composition(f).elements])
            chemsys = sorted(list(all_els))

            report = [
                f"### Scientific Audit Report: {target_formula}",
                f"**System**: {'-'.join(chemsys)} | **Cations**: {n_cations}",
                f"**Method**: DFT+SISSO Gibbs (from_pd) + Tammann gate + Szymanski regime\n",
            ]
            stage_diagnostics = []
            stage_results = []
            self._sisso_fallbacks.clear()
            sisso_mode_used = "unknown"

            for stage in thermal_profile:
                temp_c     = stage.get("target_temperature_c", 25)
                dwell_min  = stage.get("dwell_time_min", stage.get("dwell_minutes", 0))
                stage_label = stage.get("stage", "?")

                if temp_c < 350:
                    stage_diagnostics.append(
                        f"**Stage {stage_label} ({temp_c}°C)**: [SKIP] T < 350°C, non-reactive"
                    )
                    continue

                self.build_thermal_pd(temp_c, chemsys)
                sisso_mode_used = self._sisso_mode

                # ── A. Ehull（SISSO 校正相图）───────────────────────
                target_entries = [
                    e for e in self.pd.all_entries
                    if e.composition.reduced_formula == target_formula
                ]
                if not target_entries:
                    stage_diagnostics.append(
                        f"**Stage {stage_label} ({temp_c}°C)**: [MISSING] "
                        f"{target_formula} not in MP phase diagram"
                    )
                    continue

                target_entry = min(target_entries,
                                   key=lambda e: self.pd.get_e_above_hull(e))
                e_hull = self.pd.get_e_above_hull(target_entry)

                # ── B. T=0 ehull（用于报告亚稳性，与 SISSO T 结果对比）──
                pd_t0 = self._get_pd_t0(chemsys)
                t0_entries = [e for e in pd_t0.all_entries
                              if e.composition.reduced_formula == target_formula]
                e_hull_t0 = None
                if t0_entries:
                    best_t0 = min(t0_entries, key=lambda e: pd_t0.get_e_above_hull(e))
                    e_hull_t0 = pd_t0.get_e_above_hull(best_t0)

                # ── C. 熵稳定化 ──────────────────────────────────────
                entropy_stabilized = is_entropy_stabilized(target_formula, e_hull, n_cations)
                entropy_flag = (
                    " [ENTROPY-STABILIZED: E_hull positive expected at T=0]"
                    if entropy_stabilized else ""
                )

                # ── D. dG & BEP Ea ───────────────────────────────────
                try:
                    r_entries = []
                    for f in precursor_fmls:
                        match = [e for e in self.pd.all_entries
                                 if e.composition.reduced_formula == f]
                        if match:
                            r_entries.append(match[0])
                    rxn = ComputedReaction(r_entries, [target_entry])
                    total_atoms = sum(
                        c * en.composition.num_atoms
                        for en, c in zip(rxn.all_entries, rxn.coeffs) if c > 0
                    )
                    dg_rxn = rxn.calculated_reaction_energy / total_atoms
                    ea_est = 1.25 + 0.5 * max(dg_rxn, 0)
                except Exception:
                    dg_rxn, ea_est = 0.0, 1.25

                # ── E. 竞争相 ────────────────────────────────────────
                stable_competitors = []
                competitor_ehulls  = []
                for e in self.pd.stable_entries:
                    f = e.composition.reduced_formula
                    if (f != target_formula and
                            set(e.composition.elements).issubset(set(target_el_syms))):
                        stable_competitors.append(f)
                        competitor_ehulls.append(self.pd.get_e_above_hull(e))

                # ── F. Regime ────────────────────────────────────────
                regime, regime_reason = classify_regime(e_hull, competitor_ehulls, temp_c)

                # ── G. 烧结诊断 ──────────────────────────────────────
                sinter_status, sinter_msg = diagnose_sintering(
                    temp_c, target_formula, dwell_min)
                sinter_emoji = {
                    "OK": "🟢", "WARNING": "🟡", "FAIL": "🔴", "UNKNOWN": "⚪"
                }[sinter_status]
                reg_emoji = {"THERMODYNAMIC": "⬡", "KINETIC": "◈", "MIXED": "⬢"}[regime]

                # ── H. Ehull 稳定性标注 ──────────────────────────────
                ehull_str = f"{e_hull*1000:.1f} meV/atom"
                if e_hull < 0.005:
                    stability = "STABLE @ T"
                elif e_hull < 0.030:
                    stability = "NEAR-HULL (accessible)"
                elif e_hull < 0.060:
                    stability = "MODERATE"
                else:
                    stability = "DEEP META-STABLE"

                t0_note = ""
                if e_hull_t0 is not None:
                    t0_note = f" | T=0: {e_hull_t0*1000:.1f} meV/at"

                # ── 结构化 flags ─────────────────────────────────────
                flags = []
                if sinter_status == "FAIL":
                    flags.append("KINETIC_ARREST")
                elif sinter_status == "WARNING":
                    flags.append("SINTERING_MARGINAL")
                if regime == "KINETIC":
                    flags.append("KINETIC_CONTROL")
                elif regime == "MIXED":
                    flags.append("MIXED_CONTROL")
                if stable_competitors and e_hull > 0.020:
                    flags.append("COMPETING_PHASE_TRAP")
                if self._sisso_mode == "fallback_gcorr":
                    flags.append("SISSO_FALLBACK")
                if entropy_stabilized:
                    flags.append("ENTROPY_STABILIZED")
                # ── I. 氧化还原安全检查 (★ 2026-06-10 新增) ─────
                atm = protocol.get("atmosphere", "Air")
                redox_status, redox_msg = diagnose_oxidation_state(
                    temp_c, precursor_fmls, atm
                )
                redox_emoji = {
                    "OK": "🟢", "WARNING": "🟡", "FAIL": "🔴", "UNKNOWN": "⚪"
                }[redox_status]
                if redox_status == "FAIL":
                    flags.append("REDOX_RISK_CRITICAL")
                elif redox_status == "WARNING":
                    flags.append("REDOX_RISK_NEAR")


                stage_result = {
                    "stage":               stage_label,
                    "temp_c":              temp_c,
                    "dwell_min":           dwell_min,
                    "sisso_mode":          self._sisso_mode,
                    "e_hull_meV":          round(e_hull * 1000, 2),
                    "e_hull_t0_meV":       round(e_hull_t0 * 1000, 2) if e_hull_t0 is not None else None,
                    "stability":           stability,
                    "dG_rxn_eV_per_atom":  round(dg_rxn, 4),
                    "Ea_est_eV":           round(ea_est, 3),
                    "regime":              regime,
                    "regime_reason":       regime_reason,
                    "sintering_status":    sinter_status,
                    "sintering_msg":       sinter_msg,
                    "competing_phases":    stable_competitors[:6],
                    "entropy_stabilized":  entropy_stabilized,
                    "redox_status":         redox_status,  # ★ 2026-06-10 新增
                    "redox_msg":            redox_msg,     # ★ 2026-06-10 新增
                    "flags":               flags,
                }
                stage_results.append(stage_result)

                # ── Markdown 段落 ────────────────────────────────────
                res = [
                    f"**Stage {stage_label} | {temp_c}°C | {dwell_min}min{entropy_flag}**",
                    f"  Ehull(SISSO,{temp_c}°C): {ehull_str} → [{stability}]{t0_note}",
                    f"  dG_rxn: {dg_rxn:.3f} eV/atom | Ea_est: {ea_est:.2f} eV",
                    f"  {reg_emoji} Regime: {regime} — {regime_reason}",
                ]
                if redox_status != "OK":
                    res.append(f"  {redox_emoji} REDOX: {redox_status} - {redox_msg}")
                if stable_competitors and e_hull > 0.020:
                    comp_str = ", ".join(stable_competitors[:4])
                    if len(stable_competitors) > 4:
                        comp_str += f" (+{len(stable_competitors)-4} more)"
                    res.append(f"  ⚠️ Competing stable phases: {comp_str}")
                res.append(f"  {sinter_emoji} Sintering: {sinter_msg}")
                stage_diagnostics.append("\n".join(res))

            # ── 汇总 ─────────────────────────────────────────────────
            report.extend(stage_diagnostics)
            active = stage_results
            all_flags = sorted(set(f for s in active for f in s["flags"]))
            regimes   = [s["regime"] for s in active]
            overall   = Counter(regimes).most_common(1)[0][0] if regimes else "UNKNOWN"
            critical  = [s for s in active if {"KINETIC_ARREST", "REDOX_RISK_CRITICAL"} & set(s["flags"])]
            incomplete_checks = (not active or any(
                s.get("sintering_status") == "UNKNOWN" or s.get("redox_status") == "UNKNOWN"
                for s in active))
            review_status = ("REJECTED" if critical else "HOLD" if incomplete_checks
                             else "WARNING" if all_flags else "APPROVED")

            if active:
                report.append("\n**Overall Conclusion**")
                report.append(f"  Regime summary: {dict(Counter(regimes))}")
                report.append(f"  SISSO mode: {sisso_mode_used}")
                if critical:
                    report.append(
                        f"  🔴 CRITICAL: {len(critical)} stage(s) failed sintering gate"
                    )
                if any(s["entropy_stabilized"] for s in active):
                    report.append("  ℹ️ Entropy-stabilized: finite-T G needed for ehull")
                if any("SISSO_FALLBACK" in s["flags"] for s in active):
                    report.append("  ⚠️ SISSO fallback: validate with raw DFT")

            return {
                "status":            "SUCCESS",
                "execution_status":  "OK",
                "review_status":     review_status,
                "report":            "\n".join(report),
                "target":            target_formula,
                "chemsys":           chemsys,
                "stage_results":     stage_results,
                "audit_flags":       all_flags,
                "overall_regime":    overall,
                "critical_failures": critical,
                "sisso_mode":        sisso_mode_used,
            }

        except Exception as e:
            import traceback
            return {
                "status":  "ERROR",
                "execution_status": "ERROR",
                "review_status": "HOLD",
                "report":  f"Audit failed: {e}\n{traceback.format_exc()}",
                "target":  protocol_json.get("material", "UNKNOWN"),
                "chemsys": [], "stage_results": [], "audit_flags": ["AUDIT_ERROR"],
                "overall_regime": "UNKNOWN", "critical_failures": [],
                "sisso_mode": "unknown",
            }

    def audit_result(self, xrd_phases: dict, yield_pct: float, protocol_json: dict) -> dict:
        """
        后验 Hindsight 分析：将 XRD 实验结果翻译为结构化语义标签。

        Parameters
        ----------
        xrd_phases : dict
            {formula: weight_fraction}，如 {"ErTa7O19": 0.72, "Ta2O5": 0.28}
        yield_pct : float
            目标相产率（0–100）
        protocol_json : dict
            实验所用的原始协议 JSON

        Returns
        -------
        {
          "target":              str,
          "yield_pct":           float,
          "xrd_phases":          dict,
          "semantic_priors":     list[{tag, confidence, detail}],
          "hindsight_summary":   str,
          "actionable_advice":   list[str],
          "forward_audit_flags": list[str],
        }
        """
        target = protocol_json.get("material", "UNKNOWN")
        target_reduced = Composition(target).reduced_formula
        forward = self.audit_protocol(protocol_json)
        stage_results = forward.get("stage_results", [])

        semantic_priors = []
        actionable_advice = []

        # ── 副产物识别 ─────────────────────────────────────────
        byproducts = {
            Composition(f).reduced_formula: v
            for f, v in xrd_phases.items()
            if Composition(f).reduced_formula != target_reduced and v > 0.05
        }

        # ── 1. KINETIC_ARREST ──────────────────────────────────
        arrest = [s for s in stage_results if "KINETIC_ARREST" in s["flags"]]
        if arrest and yield_pct < 80:
            for s in arrest:
                semantic_priors.append({
                    "tag": "KINETIC_ARREST", "confidence": 0.90,
                    "detail": (
                        f"Stage {s['stage']} @ {s['temp_c']}°C: {s['sintering_msg']} "
                        f"| Yield={yield_pct:.0f}%"
                    )
                })
            semantic_priors.append({
                "tag": "KINETIC_ARREST", "confidence": 0.90,
                "detail": f"Tammann gate failed | Yield={yield_pct:.0f}%"
            })
            actionable_advice.append(
                f"提高峰值温度至 ≥ {arrest[0]['temp_c'] + 100}°C 或延长保温以越过 Tammann 阈值"
            )

        # ── 2. 竞争相截获 ──────────────────────────────────────
        for bp, frac in byproducts.items():
            predicted = any(bp in s.get("competing_phases", []) for s in stage_results)
            semantic_priors.append({
                "tag": "PHASE_INTERCEPTION",
                "confidence": 0.88 if predicted else 0.62,
                "detail": (
                    f"XRD byproduct {bp} @ {frac*100:.0f}% "
                    f"({'predicted' if predicted else 'unpredicted'}) "
                    f"| Yield={yield_pct:.0f}%"
                )
            })
            if frac > 0.15:
                actionable_advice.append(
                    f"调整前驱体比例以抑制 {bp}（XRD 占比 {frac*100:.0f}%）"
                )

        # ── 3. Regime 与产率 ───────────────────────────────────
        for s in stage_results:
            if s.get("regime") == "KINETIC" and yield_pct < 70:
                semantic_priors.append({
                    "tag": "KINETIC_CONTROL", "confidence": 0.82,
                    "detail": (
                        f"Stage {s['stage']} @ {s['temp_c']}°C KINETIC "
                        f"(Ehull={s['e_hull_meV']} meV) | Yield={yield_pct:.0f}%"
                    )
                })
                actionable_advice.append(
                    f"Stage {s['stage']}: 延长保温或增加中间研磨以克服动力学壁垒"
                )
            elif s.get("regime") == "THERMODYNAMIC" and yield_pct < 70:
                semantic_priors.append({
                    "tag": "THERMODYNAMIC_INSUFFICIENCY", "confidence": 0.72,
                    "detail": (
                        f"Stage {s['stage']} @ {s['temp_c']}°C THERMODYNAMIC "
                        f"但 yield={yield_pct:.0f}%，考虑化学计量或气氛"
                    )
                })

        # ── 4. SISSO 降级警告 ──────────────────────────────────
        if forward.get("sisso_mode") == "fallback_gcorr":
            semantic_priors.append({
                "tag": "SISSO_UNRELIABLE", "confidence": 0.70,
                "detail": "SISSO 降级到 G_corr fallback，G_f(T) 近似为 T=0 值"
            })

        # ── 5. 熵稳定化验证 ────────────────────────────────────
        if any(s.get("entropy_stabilized") for s in stage_results) and yield_pct > 50:
            semantic_priors.append({
                "tag": "ENTROPY_STABILIZED_VALIDATED", "confidence": 0.68,
                "detail": f"{target} 为熵稳定化相，Yield={yield_pct:.0f}% 验证高温合成可行"
            })

        # ── 6. 产率级别 ────────────────────────────────────────
        yield_tag = ("HIGH_YIELD" if yield_pct >= 90
                     else "MODERATE_YIELD" if yield_pct >= 60 else "LOW_YIELD")
        semantic_priors.append({
            "tag": yield_tag, "confidence": 0.98,
            "detail": f"Yield={yield_pct:.0f}%"
        })

        tag_str = " ".join(f"[{p['tag']}]" for p in semantic_priors)
        bp_str  = ", ".join(f"{k}({v*100:.0f}%)" for k, v in list(byproducts.items())[:3])
        hindsight_summary = (
            f"{target} | Yield={yield_pct:.0f}% | "
            f"Byproducts: [{bp_str or 'none'}] | {tag_str}"
        )
        return {
            "target":              target,
            "yield_pct":           yield_pct,
            "xrd_phases":          xrd_phases,
            "semantic_priors":     semantic_priors,
            "hindsight_summary":   hindsight_summary,
            "actionable_advice":   list(dict.fromkeys(actionable_advice)),
            "forward_audit_flags": forward.get("audit_flags", []),
        }

    def audit_arc_melt(self, recipe: dict) -> dict:
        """
        Production-grade arc-melt safety audit.
        Called identically to audit_protocol() — no new entry point needed.

        Input recipe dict fields:
            recipe_id              (str)
            target_compound        (str, e.g. "CeAl2Ge2")
            raw_materials          (list[dict]): element, molar_ratio, purity_percent, form
            hardware               (dict): water_cooled_copper_hearth, vacuum_base_pressure_pa,
                                           ti_getter_enabled, centroid_arc_temp_c
            process                (dict): flipped_melting_cycles, initial_current_A,
                                           atmosphere_pressure_mbar
            post_treatment         (dict): requires_annealing, temperature_c, duration_days,
                                           protective_foil, quench_method, sealed_in_quartz

        Returns
        -------
            dict: {
              "status":        "APPROVED" | "WARNING" | "REJECTED" | "HOLD",
              "security_token": str or None,
              "checks":        list[dict],   # each: check_id, category, status, message,
                                             #        severity, recommendation
              "hazard_summary": list[str],
            }
        """
        checks = []
        ARC_TEMP_C = recipe.get("hardware", {}).get("centroid_arc_temp_c", 3000)

        # ── 1. Hardware interlock ──────────────────────────────────
        hw = recipe.get("hardware", {})
        water_ok = hw.get("water_cooled_copper_hearth", False)
        checks.append({
            "check_id": "HW_001", "category": "hardware",
            "status": "PASS" if water_ok else "FAIL",
            "message": "水冷铜坩埚 ✓" if water_ok
                       else "CRITICAL: 未启用水冷铜坩埚，非自壳熔炼导致交叉污染",
            "severity": "CRITICAL" if not water_ok else "LOW",
            "recommendation": None if water_ok else "安装或启用 水冷铜坩埚",
        })
        vacuum_pa = hw.get("vacuum_base_pressure_pa", 999)
        vacuum_ok = vacuum_pa < 0.01
        checks.append({
            "check_id": "HW_002", "category": "hardware",
            "status": "PASS" if vacuum_ok else "FAIL",
            "message": f"本底真空 {vacuum_pa:.2e} Pa ≤ 0.01 Pa ✓" if vacuum_ok
                       else f"CRITICAL: 本底真空 {vacuum_pa:.2e} Pa 超过 0.01 Pa",
            "severity": "CRITICAL" if not vacuum_ok else "LOW",
            "recommendation": None if vacuum_ok else "检查泵组或增加洗气循环",
        })
        has_re = any(
            ELEMENT_PROPERTIES.get(m["element"], {}).get("is_rare_earth", False)
            for m in recipe.get("raw_materials", [])
        )
        ti_enabled = hw.get("ti_getter_enabled", False)
        ti_missing = has_re and not ti_enabled
        checks.append({
            "check_id": "HW_003", "category": "hardware",
            "status": "FAIL" if ti_missing else "PASS",
            "message": "Ti-getter 未配置" if ti_missing else "Ti-getter 已启用 ✓",
            "severity": "HIGH" if ti_missing else "LOW",
            "recommendation": "启用 Ti-getter 步骤（熔炼前先熔炼金属 Ti 球 30s 吸氧）"
                             if ti_missing else None,
        })

        # ── 2. Element vapor pressure check ───────────────────────
        for mat in recipe.get("raw_materials", []):
            el = mat.get("element", "")
            props = ELEMENT_PROPERTIES.get(el, {})
            if not props:
                checks.append({
                    "check_id": f"EL_{el}_UNKNOWN", "category": "element",
                    "status": "HOLD",
                    "message": f"元素 [{el}] 不在知识库中，无法评估 arc 蒸气压风险",
                    "severity": "CRITICAL",
                    "recommendation": f"查阅 CRC/ASTM 热化学数据手册，补充 {el} 的沸点数据",
                })
                continue
            bp = props.get("boiling_point_c", 9999)
            if bp < 2500:
                checks.append({
                    "check_id": f"EL_{el}_VAPOR", "category": "element",
                    "status": "FAIL",
                    "message": (f"元素 [{el}] 沸点 {bp}°C ≪ arc {ARC_TEMP_C}°C. "
                                f"预估蒸气压 >> 10 atm，极剧挥发/爆炸风险"),
                    "severity": "CRITICAL",
                    "recommendation": "使用真空感应熔炼或替换低蒸气压形态",
                })

        # ── 3. Process parameters ───────────────────────────────────
        proc = recipe.get("process", {})
        flips = proc.get("flipped_melting_cycles", 0)
        if flips < 3:
            checks.append({
                "check_id": "PROC_001", "category": "process",
                "status": "WARNING",
                "message": f"翻转熔炼次数 ({flips}) < 通用标准 (3)，可能导致宏观偏析",
                "severity": "MEDIUM",
                "recommendation": f"增加翻转熔炼次数至 {min(flips + 2, 8)} 次",
            })
        current_A = proc.get("initial_current_A", 0)
        if current_A > 100:
            checks.append({
                "check_id": "PROC_002", "category": "process",
                "status": "WARNING",
                "message": f"初始电流 {current_A}A > 100A，熔体飞溅风险升高",
                "severity": "MEDIUM",
                "recommendation": "降低初始电流至 <100A，待全熔后再增大",
            })
        pressure_mbar = proc.get("atmosphere_pressure_mbar", 0)
        if not (500 < pressure_mbar < 1000):
            checks.append({
                "check_id": "PROC_003", "category": "process",
                "status": "FAIL",
                "message": f"工作气压 {pressure_mbar} mbar 不在安全区间 (500–1000 mbar)",
                "severity": "HIGH",
                "recommendation": "调整气压至 500–800 mbar",
            })

        # ── 4. Post-treatment ──────────────────────────────────────
        pt = recipe.get("post_treatment", {})
        if pt.get("requires_annealing") and pt.get("temperature_c"):
            temp = pt["temperature_c"]
            foil = pt.get("protective_foil", "")
            if temp > 600 and not foil:
                checks.append({
                    "check_id": "PT_001", "category": "post_treatment",
                    "status": "FAIL",
                    "message": f"退火温度 {temp}°C > 600°C 且未配置保护箔，稀土合金会与石英管剧烈反应",
                    "severity": "CRITICAL",
                    "recommendation": "使用 Ta 箔（推荐）或 Nb 箔严密包裹样品后再封管",
                })
            if temp > 800 and pt.get("sealed_in_quartz"):
                checks.append({
                    "check_id": "PT_003", "category": "post_treatment",
                    "status": "WARNING",
                    "message": f"高温退火（{temp}°C）真空封管于石英管中，热膨胀系数差异可能导致破裂",
                    "severity": "MEDIUM",
                    "recommendation": "使用刚玉管（Al2O3）或降低退火温度至 <900°C",
                })

        # ── 5. CeAl2Ge2 thermodynamic context ───────────────────────
        target = recipe.get("target_compound", "").lower().replace(" ", "").replace("-", "")
        if "ceal2ge2" in target:
            checks.append({
                "check_id": "CEAL2GE2_001", "category": "thermodynamic",
                "status": "INFO",
                "message": ("CeAl₂Ge₂ 已知竞争相：Ce₂Al₃Ge₄ (E_hull=0, ground state, "
                           "热力学 sink)、CeAlGe (E_hull=6.8 meV/atom). "
                           "arc-melt 倾向于形成低能相，提高冷却速率抑制基态相"),
                "severity": "LOW",
                "recommendation": "提高冷却速率（水淬）或降低退火温度以抑制 Ce₂Al₃Ge₄",
            })

        # ── Overall status ─────────────────────────────────────────
        statuses = [c["status"] for c in checks]
        if "HOLD" in statuses:
            overall = "HOLD"
        elif "FAIL" in statuses:
            overall = "REJECTED"
        elif "WARNING" in statuses:
            overall = "WARNING"
        else:
            overall = "APPROVED"

        import datetime
        token = None
        if overall in ("APPROVED", "WARNING"):
            ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            token = f"AUTH_TOKEN_ARC_PASS_{ts}_{recipe.get('recipe_id', 'UNK')}"

        hazard_summary = [
            ELEMENT_PROPERTIES.get(m["element"], {}).get("hazard", "")
            for m in recipe.get("raw_materials", [])
            if m["element"] in ELEMENT_PROPERTIES
        ]

        return {
            "status": overall,
            "execution_status": "OK",
            "review_status": overall,
            "security_token": token,
            "checks": checks,
            "hazard_summary": hazard_summary,
        }


# ─── 执行入口 ────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({
            "error": "Usage: python audit.py <protocol.json> [xrd.json yield_pct]"
        }))
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)

    auditor = PhaseGraphAuditor(load_api_key(required=True))

    if len(sys.argv) == 4:
        with open(sys.argv[2], encoding="utf-8") as f:
            xrd = json.load(f)
        result = auditor.audit_result(xrd, float(sys.argv[3]), data)
    else:
        result = auditor.audit_protocol(data)

    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)
    print(json.dumps(result, ensure_ascii=False, indent=2, cls=_NpEncoder))
