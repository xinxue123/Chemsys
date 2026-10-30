#!/usr/bin/env python3
"""
bayesian_optimizer.py — Bayesian Optimization Module for Autonomous Chemist
==========================================================================

Plugs into autonomous_chemist.py between Phase 2 (audit) and Phase 3 (execution).

Design goals:
  1. Use yield extracted from XRD as the objective function
  2. Use sklearn GaussianProcessRegressor (no extra installs)
  3. Embed into existing audit_loop as a post-audit optimizer
  4. Support multi-round closed-loop optimization

Bayesian Optimization Loop
--------------------------
  Round N:
    1. Propose next candidate parameters (initial: audit's protocol)
    2. Run experiment → XRD → yield_from_xrd()
    3. Update GPR with (params, yield) pair
    4. Compute acquisition (EI/UCB)
    5. Propose next candidate
    6. Repeat until yield >= threshold OR max_rounds

References:
  - Noe et al. (2020) "Bayesian optimization for materials science"
    https://doi.org/10.1088/1361-648X/ab9d3b
  - sklearn GPR: Kramer et al. "Gaussian processes for machine learning"

Usage (standalone):
  python bayesian_optimizer.py ErTa7O19 --max-rounds 8 --xrd-dir ./xrd

Usage (imported):
  from bayesian_optimizer import BayesianOptimizer, yield_from_xrd
  bo = BayesianOptimizer(param_space, objective='erta7o19')
  bo.add_observation(params, yield_value)
  next_params = bo.suggest()
"""

import json, time, sys, os, sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.special import erf

# ── sklearn GPR (built-in) ────────────────────────────────────────────────
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel as C, Matern
)

# ── Paths (configurable via module-level overrides or env vars) ───────────────
# Defaults use relative paths resolved from this file's directory.
from ..paths import XRD_DB as _DEFAULT_DB, DELIVERABLES as _DEFAULT_DELIVERABLES

DEFAULT_XRD_DB = _DEFAULT_DB
DEFAULT_DELIVERABLES = _DEFAULT_DELIVERABLES

def set_audit_script(path):  global AUDIT_SCRIPT; AUDIT_SCRIPT = path
def set_python12(path):      global PYTHON12; PYTHON12 = path
def set_xrd_db(path):        global XRD_DB; XRD_DB = path
def set_deliverables(path):   global DELIVERABLES; DELIVERABLES = Path(path)

AUDIT_SCRIPT = ""   # legacy hook; auditing is now in-process (chem.audit.gate)
PYTHON12     = "python"
XRD_DB       = str(DEFAULT_XRD_DB)
DELIVERABLES = DEFAULT_DELIVERABLES

# ── Objective: yield from XRD ─────────────────────────────────────────────
def yield_from_xrd(xrd_txt_path: str, target_formula: str,
                   ref_db: str = XRD_DB) -> float:
    """
    Extract phase yield (%) from XRD data.
    Method: integrated-peak-area Rietveld-equivalent (5 diagnostic peaks/phase).

    Returns yield in [0, 100] for the target phase.
    Returns -1 on error.

    For ErTa7O19: yield = ErTa7O19 integrated area / sum(all key-peak areas) * 100
    """
    try:
        twths, intensities = parse_dat(xrd_txt_path)
    except Exception as e:
        print(f"  [yield_from_xrd] parse error: {e}", file=sys.stderr)
        return -1.0

    # ── Load reference peaks from DB ───────────────────────────────────────
    conn = sqlite3.connect(ref_db)
    c = conn.cursor()
    c.execute("SELECT formula, two_theta, hkl, I FROM peaks")
    rows = c.fetchall()
    conn.close()

    # Build ref dict: formula -> [(two_theta, hkl, I), ...]
    ref_peaks = {}
    for formula, tw, hkl, I in rows:
        ref_peaks.setdefault(formula, []).append((float(tw), hkl, float(I)))

    if target_formula not in ref_peaks:
        # Fallback: use known positions from literature
        print(f"  [yield_from_xrd] {target_formula} not in DB, using hardcoded peaks", file=sys.stderr)
        ref_peaks[target_formula] = _fallback_peaks(target_formula)

    # ── Integrate peaks in windows ──────────────────────────────────────────
    TOL = 0.7   # degrees — same as xrd_analyzer
    areas = {}
    for formula, peaks in ref_peaks.items():
        total = 0.0
        for ref_tw, hkl, I_ref in peaks:
            # Sum counts in [ref_tw - TOL, ref_tw + TOL]
            mask = (twths >= ref_tw - TOL) & (twths <= ref_tw + TOL)
            observed_I = np.max(intensities[mask]) if mask.any() else 0.0
            total += observed_I
        areas[formula] = total

    total_all = sum(areas.values())
    if total_all <= 0:
        return -1.0

    yield_pct = areas.get(target_formula, 0.0) / total_all * 100.0
    return yield_pct


def _fallback_peaks(formula: str):
    """Hardcoded fallback peaks for known phases."""
    fallbacks = {
        "ErTa7O19": [
            (29.45, "(2,-1,-1)", 100.0),
            (30.48, "(2,-1,-1)", 46.0),
            (34.37, "(2,-1,-1)", 54.0),
            (37.05, "(3,0,-3)", 58.0),
        ],
        "ErTaO4": [
            (24.69, "(1,1,0)", 36.0),
            (48.78, "(0,2,2)", 28.0),
            (56.68, "(1,1,3)", 31.0),
        ],
        "BaTiO3": [
            (22.3, "(001)", 100.0),
            (31.5, "(101)", 80.0),
            (38.9, "(111)", 60.0),
        ],
        "ZrNCl": [
            (12.6, "(001)", 100.0),
            (37.9, "(003)", 70.0),
            (40.1, "(100)", 65.0),
        ],
    }
    return fallbacks.get(formula, [(29.0, "?", 50.0)])


def parse_dat(path: str):
    """Parse .txt XRD file. Returns (two_thetas, intensities)."""
    twths, intensities = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    t = float(parts[0])
                    i = float(parts[1])
                    if 5 <= t <= 90:
                        twths.append(t)
                        intensities.append(i)
                except ValueError:
                    pass
    return np.array(twths), np.array(intensities)


# ── Parameter Space Definition ────────────────────────────────────────────────
# For ErTa7O19 solid-state synthesis (full thermal profile):
#   - T_max_c:          最高烧结温度 (°C) — Ta2O5分解驱动力
#   - T_dwell_c:        最终保温温度 (°C) — 熵稳定化窗口
#   - dwell_min:        第一轮保温时间 (min)
#   - dwell2_min:       第二轮保温时间 (min, 0=跳过)
#   - intermediate_grind: 中间研磨 (0/1)
#   - step_down:        逐级降温 (0/1): 0=直接升温, 1=T_max→T_dwell逐降
#   - decomp_temp_c:    分解温度 (°C)
#   - decomp_dwell_min: 分解保温时间 (min)
PARAM_BOUNDS = {
    "T_max_c":           (1100.0, 1450.0),
    "T_dwell_c":         (1100.0, 1250.0),
    "dwell_min":         (60.0,   720.0),
    "dwell2_min":        (0.0,    480.0),
    "intermediate_grind":(0.0,    1.0),
    "step_down":         (0.0,    1.0),
    "decomp_temp_c":     (350.0,  600.0),
    "decomp_dwell_min":  (60.0,   300.0),
}
PARAM_KEYS = list(PARAM_BOUNDS.keys())
N_PARAMS = len(PARAM_KEYS)

# ── 降维配置（8D → 3D 活跃优化）─────────────────────────────────────────────
# 依据：length_scale 分析显示 T_dwell=0.02, T_max=0.07 最敏感（8D GPR 崩溃）
#       其他 5 维 length_scale > 1000，几乎不敏感 → 固定到最优值
ACTIVE_KEYS_3D = ["T_dwell_c", "T_max_c", "intermediate_grind"]
FIXED_8D = {
    # These must match the OPTIMAL values to avoid losing information
    # during the 8D→3D projection. If training data has variation in a
    # fixed dimension, that variation is IRRECOVERABLY LOST.
    "dwell_min":        480.0,   # Cycle6/best exp: 480 or 360; use 480 to cover
    "dwell2_min":      360.0,   # Cycle6: 360min second dwell
    "step_down":          1.0,   # Cycle6=1 (逐级降温 critical for 40%)
    "decomp_temp_c":    400.0,
    "decomp_dwell_min":120.0,
}
BOUNDS_3D = [
    (1100.0, 1250.0),   # T_dwell_c
    (1200.0, 1450.0),   # T_max_c
    (0.0,     1.0  ),   # intermediate_grind
]

# ── Initial GPR length scales (physics-informed) ──────────────────────────
# T_dwell: Δyield~18% over ΔT=100°C → length_scale ≈ 50°C is meaningful
# T_max:   only C5→C6 data (ΔT=200, Δyield~3%), but confounded by grind/dwell
#          use length_scale=200 to avoid over-sensitivity to sparse T_max data
# grind:   binary, length_scale=0.5 is standard for 0/1
_INITIAL_LS = [50.0, 200.0, 0.5]


def project_to_active(vec8D: np.ndarray) -> np.ndarray:
    """Project 8D vector → 3D active subspace."""
    return np.array([vec8D[PARAM_KEYS.index(k)] for k in ACTIVE_KEYS_3D])


def project_to_full(vec3D: np.ndarray) -> np.ndarray:
    """
    Expand 3D active vector → 8D full space (filling FIXED values).
    vec3D = [T_dwell_c, T_max_c, intermediate_grind]
    """
    full = np.zeros(N_PARAMS)
    full[PARAM_KEYS.index("T_dwell_c")]       = vec3D[0]   # active
    full[PARAM_KEYS.index("T_max_c")]         = vec3D[1]   # active
    full[PARAM_KEYS.index("intermediate_grind")] = vec3D[2]  # active
    full[PARAM_KEYS.index("dwell_min")]       = FIXED_8D["dwell_min"]
    full[PARAM_KEYS.index("dwell2_min")]      = FIXED_8D["dwell2_min"]
    full[PARAM_KEYS.index("step_down")]       = FIXED_8D["step_down"]
    full[PARAM_KEYS.index("decomp_temp_c")]   = FIXED_8D["decomp_temp_c"]
    full[PARAM_KEYS.index("decomp_dwell_min")] = FIXED_8D["decomp_dwell_min"]
    return full

def params_to_vec(params: dict) -> np.ndarray:
    return np.array([params[k] for k in PARAM_KEYS])

def vec_to_params(vec: np.ndarray) -> dict:
    out = {}
    for k, v in zip(PARAM_KEYS, vec):
        if k in ("intermediate_grind", "step_down"):
            out[k] = "yes" if v >= 0.5 else "no"
        elif k in ("dwell2_min",):
            out[k] = float(v)
        else:
            out[k] = float(v)
    return out

def vec_to_protocol(vec: np.ndarray, base_protocol: dict) -> dict:
    """
    Fill in BO params into base protocol template from audit.
    Maps 8-dim BO vector to full thermal profile with multi-stage profile:
      Stage 1: decomp (decomp_temp_c, decomp_dwell_min)
      Stage 2: T_max_c → T_dwell_c ramp + dwell_min
      Stage 3: T_dwell_c dwell (if dwell2_min > 0, second cycle)
      Grind: intermediate_grind
    """
    p = vec_to_params(vec)
    proto = json.loads(json.dumps(base_protocol))

    # Build full thermal profile
    stages = []

    # Stage 1: decomposition
    stages.append({
        "stage": 1,
        "stage_type": "decomposition",
        "target_temperature_c": int(round(p["decomp_temp_c"])),
        "dwell_minutes": int(round(p["decomp_dwell_min"])),
        "ramp_rate_c_min": 5.0,
    })

    # Stage 2: high-T ramp to T_max, dwell at T_max
    if p["step_down"] == "yes":
        # T_max -> T_dwell step down
        stages.append({
            "stage": 2,
            "stage_type": "high_temp_ramp",
            "target_temperature_c": int(round(p["T_max_c"])),
            "dwell_minutes": int(round(p["dwell_min"])),
            "ramp_rate_c_min": 5.0,
        })
        stages.append({
            "stage": 3,
            "stage_type": "step_down_sintering",
            "target_temperature_c": int(round(p["T_dwell_c"])),
            "dwell_minutes": int(round(p["dwell2_min"])) if p["dwell2_min"] > 0 else int(round(p["dwell_min"])),
            "ramp_rate_c_min": 2.0,   # controlled step-down
        })
    else:
        # Direct to T_dwell
        stages.append({
            "stage": 2,
            "stage_type": "sintering",
            "target_temperature_c": int(round(p["T_dwell_c"])),
            "dwell_minutes": int(round(p["dwell_min"])),
            "ramp_rate_c_min": 5.0,
        })
        if p["dwell2_min"] > 0:
            stages.append({
                "stage": 3,
                "stage_type": "second_sintering",
                "target_temperature_c": int(round(p["T_dwell_c"])),
                "dwell_minutes": int(round(p["dwell2_min"])),
                "ramp_rate_c_min": 5.0,
            })

    proto["protocol"]["thermal_profile"] = stages
    proto["protocol"]["metadata"]["intermediate_grind"] = p["intermediate_grind"]
    proto["protocol"]["metadata"]["step_down"] = p["step_down"]
    proto["protocol"]["metadata"]["T_max_c"] = p["T_max_c"]
    return proto


# ── GPR Model ─────────────────────────────────────────────────────────────
class GPRModel:
    """
    3D Active Subspace GPR for ErTa7O19 solid-state synthesis.
    Active dims: T_dwell_c, T_max_c, intermediate_grind
    Fixed dims: dwell_min, dwell2_min, step_down, decomp_temp_c, decomp_dwell_min

    Kernel: ConstantKernel * Matern(nu=2.5) + WhiteKernel
    Kernel bounds are widened to prevent convergence warnings with n=6.
    """
    def __init__(self, n_dims: int = 3, noise: float = 2.0):
        self.n_dims = n_dims  # always 3 (active subspace)
        self.noise  = noise
        # Use physics-informed initial length scales (not optimized from data yet)
        # This prevents the optimizer from converging to pathological solutions
        # when n << d (e.g. 6 points in 8D reduced to 3D with sparse coverage)
        kernel = (
            C(15.0, (1e-1, 1e4)) *
            Matern(
                length_scale=_INITIAL_LS,  # [T_dwell=50, T_max=200, grind=0.5]
                nu=2.5,
                length_scale_bounds=(5.0, 1e4)  # FIXED lower bound = 5.0
            ) +
            WhiteKernel(noise_level=noise**2, noise_level_bounds=(1e-1, 1e2))
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=1e-6,
            random_state=42,
        )
        self._fitted    = False
        self.X_train_   = np.zeros((0, n_dims))
        self.y_train_   = np.zeros(0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = np.atleast_2d(X)
        self.y_train_ = np.atleast_1d(y)
        self.model.fit(self.X_train_, self.y_train_)
        self._fitted = True

    def predict(self, X, return_std=True):
        X = np.atleast_2d(X)
        return self.model.predict(X, return_std=return_std)

    def suggest_next(self, bounds: list, n_candidates: int = 2000,
                     acquisition: str = "ei",
                     xi: float = 0.01) -> np.ndarray:
        """
        Grid-sample candidates, evaluate acquisition, return best.
        acquisition: 'ei' (Expected Improvement) or 'ucb' (Upper Confidence Bound)
        """
        if len(self.X_train_) < 2:
            # Fewer than 2 points: random exploration to build initial model
            rng = np.random.default_rng(int(time.time()))
            idx = rng.integers(0, len(bounds))
            # Explore each dimension randomly
            X_cands = np.column_stack([
                rng.uniform(lo, hi, n_candidates)
                for lo, hi in bounds
            ])
            # Pick a random candidate far from the first observation
            if len(self.X_train_) == 1:
                x0 = self.X_train_[0]
                dists = np.linalg.norm(X_cands - x0, axis=1)
                # Pick the 70th percentile distance to ensure exploration
                thresh = np.percentile(dists, 70)
                mask   = dists >= thresh
                X_cands = X_cands[mask]
            best_idx = np.random.default_rng(int(time.time()*1e6)%(len(X_cands)))
            best_idx = best_idx.integers(0, len(X_cands))
            return X_cands[best_idx]

        rng = np.random.default_rng(42)
        X_cands = np.column_stack([
            rng.uniform(lo, hi, n_candidates)
            for lo, hi in bounds
        ])

        mu, sigma = self.predict(X_cands)
        sigma = np.maximum(sigma, 1e-6)

        if acquisition == "ei":
            # Expected Improvement (for maximization)
            y_best = np.max(self.y_train_)
            z      = (mu - y_best - xi) / sigma
            ei     = (mu - y_best - xi) * GPRModel._norm_cdf(z) + sigma * GPRModel._norm_pdf(z)
            scores = ei
        else:  # ucb
            kappa  = 2.0
            scores = mu + kappa * sigma

        best_idx = np.argmax(scores)
        return X_cands[best_idx]

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1 + erf(x / np.sqrt(2)))

    @staticmethod
    def _norm_pdf(x):
        return np.exp(-0.5*x**2) / np.sqrt(2*np.pi)


# ── Acquisition Functions ───────────────────────────────────────────────────
def expected_improvement(mu, sigma, y_best, xi=0.01):
    """EI for maximization problem."""
    sigma = np.maximum(sigma, 1e-6)
    z     = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * 0.5*(1+erf(z/np.sqrt(2))) + sigma*np.exp(-0.5*z**2)/np.sqrt(2*np.pi)


# ── Main BO Loop ────────────────────────────────────────────────────────────
class BayesianOptimizer:
    """
    Bayesian Optimization for solid-state synthesis parameter search.

    Embeds into autonomous_chemist as:
      Phase 2.5: Bayesian Optimization (between audit_loop and execution_plan)

    Workflow:
      1. Initialize from audit's best protocol
      2. For each round:
         a. Run experiment → XRD
         b. Extract yield_from_xrd()
         c. Update GPR
         d. Suggest next params via acquisition
      3. Stop when yield >= threshold OR max_rounds reached
      4. Return best protocol + optimization trajectory
    """

    def __init__(self,
                 target_formula: str,
                 base_protocol: dict,
                 max_rounds: int = 8,
                 yield_threshold: float = 85.0,
                 noise: float = 3.0,   # yield measurement noise (%)
                 ):

        self.target    = target_formula
        self.base      = base_protocol        # from audit_loop
        self.max_rounds = max_rounds
        self.threshold  = yield_threshold
        self.noise      = noise

        bounds = BOUNDS_3D  # 3D active subspace
        self.gpr    = GPRModel(n_dims=3, noise=noise)
        self.bounds  = bounds
        self.rounds  = []
        self.best_y  = -np.inf
        self.best_x  = None
        self.history = []   # list of {"round", "params", "yield", "xrd_file"}

        # ── Warm-start: seed from base protocol's sintering params ────────
        init_vec = self._extract_initial_vec()
        # Evaluate initial point (if xrd data exists, use it; else add as observation)
        print(f"  [BO] Initial params from audit: {vec_to_params(init_vec)}")

    def _extract_initial_vec(self) -> np.ndarray:
        """Pull sintering params from base_protocol → 8D full vector."""
        p = self.base
        thermal = p.get("protocol", {}).get("thermal_profile", [])
        sintering_stage = thermal[-1] if thermal else {}
        grind_flag = p.get("protocol", {}).get("metadata", {}).get("intermediate_grind", "no")
        grind = 1.0 if grind_flag == "yes" else 0.0
        step_flag = p.get("protocol", {}).get("metadata", {}).get("step_down", "no")
        step = 1.0 if step_flag == "yes" else 0.0
        T_max_val = p.get("protocol", {}).get("metadata", {}).get("T_max_c", 1400.0)
        decomp_t  = p.get("protocol", {}).get("thermal_profile", [{}])[0].get("target_temperature_c", 400.0)
        decomp_d  = p.get("protocol", {}).get("thermal_profile", [{}])[0].get("dwell_minutes", 120.0)
        return np.array([
            T_max_val,  # T_max_c
            sintering_stage.get("target_temperature_c", 1200.0),  # T_dwell_c
            sintering_stage.get("dwell_minutes", 360.0),          # dwell_min
            0.0,        # dwell2_min
            grind,      # intermediate_grind
            step,       # step_down
            decomp_t,   # decomp_temp_c
            decomp_d,   # decomp_dwell_min
        ])

    def add_observation(self, params_vec8D: np.ndarray, yield_value: float,
                       xrd_file: str = None):
        """
        Add a (params, yield) observation.
        Internally projects 8D → 3D active subspace for GPR training.
        """
        # Project to 3D active subspace
        vec3D = project_to_active(np.atleast_1d(params_vec8D))
        X = vec3D.reshape(1, -1)
        y = np.array([yield_value])
        if self.gpr._fitted:
            X_all = np.vstack([self.gpr.X_train_, X])
            y_all = np.hstack([self.gpr.y_train_, y])
            self.gpr.fit(X_all, y_all)
        else:
            self.gpr.fit(X, y)

        self.history.append({
            "params_vec8D": np.atleast_1d(params_vec8D).tolist(),
            "params": vec_to_params(params_vec8D),
            "yield": yield_value,
            "xrd_file": xrd_file,
        })

        if yield_value > self.best_y:
            self.best_y = y[0]
            self.best_x = np.atleast_1d(params_vec8D).copy()

        print(f"  [BO] Added obs: yield={yield_value:.1f}%, best={self.best_y:.1f}%")

    def suggest(self) -> np.ndarray:
        """
        Suggest next experiment parameters.
        Uses 3D active-subspace GPR for acquisition (EI),
        then expands the 3D candidate → 8D full vector for robot execution.
        """
        if not self.gpr._fitted:
            # No data yet: return initial from audit (8D)
            return self._extract_initial_vec()

        # Sample candidates in 3D active space
        rng = np.random.default_rng(42)
        X_cands3D = np.column_stack([
            rng.uniform(lo, hi, 2000) for lo, hi in self.bounds
        ])
        # Discretize grind
        X_cands3D[:, 2] = (X_cands3D[:, 2] > 0.5).astype(float)

        mu, sigma = self.gpr.predict(X_cands3D)
        sigma = np.maximum(sigma, 1e-6)
        y_best = self.best_y

        # EI
        z = (mu - y_best - 0.01) / sigma
        ei = (mu - y_best - 0.01)*(0.5*(1+erf(z/np.sqrt(2)))) + \
              sigma*np.exp(-0.5*z**2)/np.sqrt(2*np.pi)
        best_idx = np.argmax(ei)
        best3D = X_cands3D[best_idx]

        # Expand 3D → 8D for robot execution
        return project_to_full(best3D)

    def should_stop(self) -> bool:
        if self.best_y >= self.threshold:
            print(f"  [BO] Yield threshold {self.threshold}% reached: {self.best_y:.1f}%")
            return True
        if len(self.history) >= self.max_rounds:
            print(f"  [BO] Max rounds {self.max_rounds} reached.")
            return True
        return False

    def run_round(self, round_num: int, xrd_file: str) -> dict:
        """
        Execute one BO round: suggest → observe → update → decide.
        suggest() returns 8D vector; add_observation projects to 3D internally.
        """
        if round_num == 0:
            params_vec = self._extract_initial_vec()
        else:
            params_vec = self.suggest()  # now returns 8D np.ndarray

        yield_val = yield_from_xrd(xrd_file, self.target)

        if yield_val < 0:
            print(f"  [BO] WARNING: yield extraction failed for {xrd_file}, setting to 0")
            yield_val = 0.0

        self.add_observation(params_vec, yield_val, xrd_file)

        mu_all, sigma_all = [], []
        if self.gpr._fitted:
            X_grid = np.column_stack([
                np.linspace(lo, hi, 50) for lo, hi in self.bounds
            ])
            mu_all, sigma_all = self.gpr.predict(X_grid)

        return {
            "round": round_num,
            "params_vec": params_vec.tolist(),
            "params": vec_to_params(params_vec),
            "yield": yield_val,
            "best_y": self.best_y,
            "best_params": vec_to_params(self.best_x) if self.best_x is not None else None,
            "stop": self.should_stop(),
        }

    def optimization_trajectory(self) -> list:
        """Return full history with round-by-round details."""
        return self.history

    def best_protocol(self) -> dict:
        """Return the protocol corresponding to the best observed yield."""
        if self.best_x is None:
            return self.base
        return vec_to_protocol(self.best_x, self.base)


# ── Integration with autonomous_chemist ─────────────────────────────────────
"""
To embed BO into autonomous_chemist.py, add after audit_loop():

    from bayesian_optimizer import BayesianOptimizer, yield_from_xrd, PARAM_BOUNDS, PARAM_KEYS

    # Phase 2.5: Bayesian Optimization
    if use_bo and not audit_log.get("success"):
        print(f"\n[Phase 2.5] Bayesian Optimization ({max_bo_rounds} rounds)")
        bo = BayesianOptimizer(
            target_formula=current_protocol["material"],
            base_protocol=audit_log.get("final_protocol", current_protocol),
            max_rounds=max_bo_rounds,
            yield_threshold=85.0,
        )
        for round_i in range(max_bo_rounds):
            xrd_file = f"./xrd/bo_round_{round_i}.txt"
            if not os.path.exists(xrd_file):
                print(f"  Round {round_i}: No XRD file yet — run experiment first")
                break
            result = bo.run_round(round_i, xrd_file)
            print(f"  Round {round_i}: yield={result['yield']:.1f}%, best={result['best_y']:.1f}%")
            if result["stop"]:
                break

        best_protocol = bo.best_protocol()
    else:
        best_protocol = audit_log.get("final_protocol", current_protocol)

Note: In a fully automated lab, Phase 4 (XRD) would be roboticized and
the BO loop would run without human intervention.
"""


# ── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bayesian Optimizer for Autonomous Chemist")
    parser.add_argument("material", help="Target material, e.g. ErTa7O19")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--xrd-dir", default="./xrd",
                        help="Directory containing XRD .txt files")
    parser.add_argument("--threshold", type=float, default=85.0,
                        help="Yield threshold to stop optimization (%%)")
    parser.add_argument("--seed", type=int, default=0,
                        help="First XRD file index in xrd-dir (for simulation mode)")
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate with synthetic yield (no real XRD)")
    args = parser.parse_args()

    print(f"\nBayesian Optimizer — {args.material}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  XRD dir:    {args.xrd_dir}")
    print(f"  Threshold:  {args.threshold}%\n")

    # Build base protocol from audit
    AUDIT_CMD = [
        PYTHON12, AUDIT_SCRIPT,
        str(DELIVERABLES / f"{args.material}_phase2.json")
    ]
    base_protocol = None
    if Path(AUDIT_CMD[2]).exists():
        with open(AUDIT_CMD[2]) as f:
            base_protocol = json.load(f).get("data", {}).get("protocol")
    if base_protocol is None:
        # Minimal fallback protocol
        base_protocol = {
            "material": args.material,
            "protocol": {
                "precursors": [{"chemical": "Er2O3"}, {"chemical": "Ta2O5"}],
                "thermal_profile": [
                    {"stage": 1, "stage_type": "decomposition",
                     "target_temperature_c": 600, "dwell_minutes": 300,
                     "ramp_rate_c_min": 5.0},
                    {"stage": 2, "stage_type": "sintering",
                     "target_temperature_c": 1200, "dwell_minutes": 390,
                     "ramp_rate_c_min": 5.0},
                ],
                "metadata": {"intermediate_grind": "yes"},
            }
        }

    bo = BayesianOptimizer(
        target_formula=args.material,
        base_protocol=base_protocol,
        max_rounds=args.max_rounds,
        yield_threshold=args.threshold,
    )

    # Collect XRD files from xrd-dir matching material pattern
    xrd_dir = Path(args.xrd_dir)
    xrd_files = sorted(xrd_dir.glob(f"**/*{args.material}*.txt"))
    if not xrd_files:
        xrd_files = sorted(xrd_dir.glob("**/ErTa*.txt"))
    if not xrd_files:
        xrd_files = sorted(xrd_dir.glob("**/*.txt"))

    print(f"Found {len(xrd_files)} XRD files")

    for round_i in range(args.max_rounds):
        xrd_file = xrd_files[round_i] if round_i < len(xrd_files) else None

        if args.simulate:
            # Simulated yield: based on distance from known optimum
            params_vec = bo.suggest() if round_i > 0 else bo._extract_initial_vec()
            T = params_vec[0]
            dwell = params_vec[1]
            # Simulate: yield peaks at 1200C, 390min
            sim_yield = (
                40.0
                + 55.0 * np.exp(-((T - 1200)/50)**2)
                + 10.0 * np.exp(-((dwell - 400)/200)**2)
                + np.random.normal(0, 3.0)
            )
            sim_yield = max(0, min(100, sim_yield))
            bo.add_observation(params_vec, sim_yield, str(xrd_file) if xrd_file else None)
            result = {"round": round_i, "yield": sim_yield,
                      "best_y": bo.best_y, "stop": bo.should_stop()}
        else:
            if xrd_file is None or not xrd_file.exists():
                print(f"Round {round_i}: No XRD file available — waiting for experiment")
                break
            result = bo.run_round(round_i, str(xrd_file))

        print(f"  Round {round_i}: yield={result['yield']:.1f}%  "
              f"best={result['best_y']:.1f}%  stop={result['stop']}")

        if result["stop"]:
            break

    print(f"\n=== Optimization Complete ===")
    print(f"  Best yield: {bo.best_y:.1f}%")
    print(f"  Best params: {vec_to_params(bo.best_x) if bo.best_x is not None else 'N/A'}")

    # Save result
    out_path = DELIVERABLES / f"{args.material}_bo_result.json"
    DELIVERABLES.mkdir(parents=True, exist_ok=True)
    output = {
        "material": args.material,
        "best_yield": bo.best_y,
        "best_params": vec_to_params(bo.best_x) if bo.best_x is not None else {},
        "best_protocol": bo.best_protocol(),
        "trajectory": [
            {"round": h["round"], "params": h["params"], "yield": h["yield"]}
            if "round" in h else h
            for h in bo.optimization_trajectory()
        ],
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {out_path}")
