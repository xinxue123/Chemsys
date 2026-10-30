"""
LLM-Driven Bayesian Optimization -- Fully Generic, No Plugin Registry
=====================================================================
Replaces the deprecated SystemPlugin/register_plugin BO stack

Core idea: the LLM IS the "plugin".  Every optimization run is self-contained.
The LLM reasons over the system description + existing observations to:
  1. Infer param_space (what variables to vary, in what range)
  2. Generate mu_func(x) -> float   (physics-based mean yield)
  3. Generate sigma_func(x) -> float (total uncertainty)
  4. Select active_params (which subset to actively optimize)

Three model regimes:
  n < 3  -> ConservativePrior  (mu=50%, sigma 15→30% at boundaries)
  n < 10 -> LLM-Guided          (LLM generates physics model)
  n >= 10 -> GPR               (data-driven, sklearn Matern nu=2.5)

Usage (production with real LLM)
--------------------------------
    bo = LLMDrivenBO(
        system_description="Ce-Al-Ge intermetallic via arc-melt. "
                          "Variables: AlCe_ratio (1.5-20), anneal_temp (400-900°C). "
                          "Ground state: Ce2Al3Ge4 (ehull=0). Target: CeAl2Ge2 (ehull=12.4).",
        llm_reasoner=my_llm_api_wrapper,   # your LLM callable
    )
    bo.add_observation(AlCe_ratio=6.0, anneal_temp=850, yield_value=64.5)
    bo.add_observation(AlCe_ratio=3.5, anneal_temp=850, yield_value=28.0)
    sugg = bo.suggest()
    # run experiment, feed result back via bo.add_observation(...)

Usage (testing / fallback without LLM)
-------------------------------------
    from llm_driven_bo import LLMDrivenBO, heuristic_reasoner
    bo = LLMDrivenBO(system_description="...", llm_reasoner=heuristic_reasoner)
"""

import numpy as np
from scipy.special import erf, erfcx  # noqa: F401
from dataclasses import dataclass, field, replace
from typing import Optional, Callable, Dict, List, Tuple
from collections import defaultdict
from types import MappingProxyType

# --------------------------------------------------------------------------
# Acquisition helpers (pure numpy, system-agnostic)
# --------------------------------------------------------------------------

def _ei(mu: float, sigma: float, y_best: float, xi: float = 0.01) -> float:
    if sigma <= 0:
        return 0.0
    delta = mu - y_best - xi
    z = delta / sigma
    return delta * (0.5 + 0.5 * erf(z / np.sqrt(2))) + \
           sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _ucb(mu: float, sigma: float, kappa: float = 2.0) -> float:
    return mu + kappa * sigma


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _cast(val) -> float:
    """Convert numpy scalar to Python float."""
    if hasattr(val, 'item'):
        return val.item()
    return float(val)


# --------------------------------------------------------------------------
# Conservative Prior (n < 3 -- safe, no model reliance)
# --------------------------------------------------------------------------

@dataclass
class ConservativePrior:
    param_space: Dict[str, Tuple[float, float]]

    def mu(self, x: dict) -> float:
        return 50.0

    def sigma(self, x: dict) -> float:
        if not self.param_space:
            return 20.0
        dist = 0.0
        for k, (lo, hi) in self.param_space.items():
            center = (lo + hi) * 0.5
            half = (hi - lo) * 0.5
            if half > 0:
                val = _cast(x.get(k, center))
                dist += ((val - center) / half) ** 2
        dist = np.sqrt(dist / max(1, len(self.param_space)))
        return 15.0 + 15.0 * min(dist, 1.0)


# --------------------------------------------------------------------------
# GPR Model (n >= gpr_threshold -- data-driven Matern GPR)
# --------------------------------------------------------------------------

class GPRModel:
    """Shared GP with immutable full inputs and explicit observation noise.

    Research objectives are transformed before fitting and use
    ``normalize_y=False``. Legacy percentage-valued callers may request
    normalization, but missing coordinates are never imputed.
    """
    def __init__(self, param_space: Dict[str, Tuple[float, float]], *,
                 normalize_y: bool = True, random_state: int = 0,
                 n_restarts_optimizer: int = 5, alpha: float = 1e-6):
        space = {name: tuple(float(v) for v in bounds) for name, bounds in param_space.items()}
        if not space or any(len(v) != 2 or not np.isfinite(v).all() or v[0] >= v[1] for v in space.values()):
            raise ValueError("GPR requires a fixed nonempty domain with finite lower < upper bounds")
        if not np.isfinite(alpha) or alpha <= 0 or n_restarts_optimizer < 0:
            raise ValueError("GPR nugget must be positive and restart count nonnegative")
        self.param_space = MappingProxyType(space)
        self.param_names = tuple(space)
        self.normalize_y = bool(normalize_y)
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = float(alpha)
        self._fitted = False
        self._gpr = None
        self._X_buf: List[dict] = []
        self._y_buf: List[float] = []
        self._sigma_buf: Optional[list[float]] = None
        self.fit_count = 0

    def _to_scaled(self, X_raw: List[dict]) -> np.ndarray:
        X = np.zeros((len(X_raw), len(self.param_names)))
        for i, x in enumerate(X_raw):
            if set(x) != set(self.param_names):
                raise ValueError("GPR input must contain exactly the fixed full parameter vector")
            for j, k in enumerate(self.param_names):
                lo, hi = self.param_space[k]
                value = x[k]
                if isinstance(value, (str, bool)) or not np.isscalar(value) or not np.isfinite(value):
                    raise ValueError(f"GPR coordinate {k} must be a finite number")
                val = float(value)
                if not lo <= val <= hi:
                    raise ValueError(f"GPR coordinate {k} is outside the immutable hard domain")
                X[i, j] = (val - lo) / (hi - lo)
        return X

    def fit(self, X_raw: List[dict], y_raw: List[float], *, y_sigma=None) -> None:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

        X = self._to_scaled(X_raw)
        y = np.asarray(y_raw, dtype=float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError("GPR observations must align with finite objective values")
        sigma = None if y_sigma is None else np.asarray(y_sigma, dtype=float)
        if sigma is not None and (sigma.shape != y.shape or not np.isfinite(sigma).all() or np.any(sigma < 0)):
            raise ValueError("GPR uncertainties must be finite, nonnegative, and aligned")
        self._X_buf = [dict(row) for row in X_raw]
        self._y_buf = y.tolist()
        self._sigma_buf = None if sigma is None else sigma.tolist()
        self._fitted = False
        if len(X) < 2:
            self._fitted = False
            return

        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(len(self.param_names)),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        noise = self.alpha if sigma is None else sigma ** 2 + self.alpha
        if sigma is not None and self.normalize_y:
            scale = float(y.std()) or 1.0
            noise = noise / scale ** 2
        self._gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=noise, random_state=self.random_state,
            n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=self.normalize_y,
        )
        self._gpr.fit(X, y)
        self._fitted = True
        self.fit_count += 1

    def predict_many(self, X_raw: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._to_scaled(X_raw)
        if not self._fitted:
            raise RuntimeError("GPR prediction requires at least two fitted observations")
        if not len(X):
            return np.empty(0), np.empty(0)
        return self._gpr.predict(X, return_std=True)

    def predict(self, x_raw: dict) -> Tuple[float, float]:
        mu, sigma = self.predict_many([x_raw])
        return float(mu[0]), float(sigma[0])

    def add_observation(self, x_raw: dict, y_raw: float, *, y_sigma=None) -> None:
        if (self._sigma_buf is None) != (y_sigma is None) and self._X_buf:
            raise ValueError("uncertainty availability must be consistent across observations")
        sigma = None if y_sigma is None else [*(self._sigma_buf or []), y_sigma]
        self.fit([*self._X_buf, dict(x_raw)], [*self._y_buf, y_raw], y_sigma=sigma)


# --------------------------------------------------------------------------
# LLM Model container (returned by the LLM reasoner)
# --------------------------------------------------------------------------

@dataclass
class LLMModel:
    """Container for LLM-generated yield model."""
    mu_func: Callable[[dict], float]
    sigma_func: Callable[[dict], float]
    param_space: Dict[str, Tuple[float, float]]
    active_params: List[str]
    reasoning: str = ""
    model_code: str = ""


# --------------------------------------------------------------------------
# Main BO class
# --------------------------------------------------------------------------

@dataclass
class Suggestion:
    params: dict
    mu_pred: float
    sigma_pred: float
    acq_method: str
    acq_value: float
    ci_lo: float
    ci_hi: float

    def __str__(self) -> str:
        return (
            f"Suggestion [{self.acq_method}={self.acq_value:.3f}]\n"
            f"  mu={self.mu_pred:.1f}%  sigma={self.sigma_pred:.1f}%\n"
            f"  Approximate model interval (uncalibrated): [{self.ci_lo:.1f}%, {self.ci_hi:.1f}%]\n"
            f"  params: {self.params}"
        )


@dataclass
class Observation:
    yield_value: float
    raw: dict = field(default_factory=dict)
    yield_sigma: Optional[float] = None


class LLMDrivenBO:
    """
    Fully generic, plugin-free Bayesian optimizer driven by an LLM reasoner.

    No SystemPlugin registry.  No register_plugin() calls.
    Every optimization run is self-contained.

    Parameters
    ----------
    system_description : str
        Free-text description of the synthesis system.
        Example: "Ce-Al-Ge intermetallic via arc-melt. Optimize AlCe ratio (1.5-20),
        anneal temperature. Ground state Ce2Al3Ge4 (ehull=0). Target CeAl2Ge2."
    llm_reasoner : callable(system_description, observations) -> LLMModel
        Your LLM API wrapper.  Receives system description + observation history,
        returns an LLMModel with mu_func, sigma_func, param_space, active_params.
        If None: uses heuristic_reasoner (keyword-based, no LLM needed).
    gpr_threshold : int (default 10)
        Switch to GPR when n_obs >= gpr_threshold.
    exploration_weight : float (default 2.0)
        UCB kappa.  Higher = more exploration.
    ei_xi : float (default 0.01)
        EI minimum improvement threshold.
    """

    def __init__(
        self,
        system_description: str,
        llm_reasoner: Optional[Callable[[str, List[Observation]], LLMModel]] = None,
        gpr_threshold: int = 10,
        exploration_weight: float = 2.0,
        ei_xi: float = 0.01,
        full_param_space: Optional[Dict[str, Tuple[float, float]]] = None,
        max_llm_calls: int = 4,
        refresh_observation_counts: Tuple[int, ...] = (0, 4, 8, 12),
        candidate_filter: Optional[Callable[[dict], bool]] = None,
        binary_params: Tuple[str, ...] = ("grind", "step_down", "intermediate_grind"),
    ):
        self.system_description = system_description
        self.llm_reasoner = llm_reasoner if llm_reasoner is not None else heuristic_reasoner
        self.gpr_threshold = gpr_threshold
        self.kappa = exploration_weight
        self.xi = ei_xi
        if type(max_llm_calls) is not int or not 0 <= max_llm_calls <= 4:
            raise ValueError("LLM call budget must be an integer from zero to four")
        self.max_llm_calls = max_llm_calls
        self.refresh_observation_counts = tuple(refresh_observation_counts)
        self.llm_events = []
        self.candidate_filter = candidate_filter
        self.binary_params = frozenset(binary_params)

        self.observations: List[Observation] = []
        self._llm_model: Optional[LLMModel] = None
        self._gpr_model: Optional[GPRModel] = None
        self._param_space = MappingProxyType({key: tuple(value) for key, value in (full_param_space or {}).items()})
        self._active_params: List[str] = []
        self._mode = "conservative"

        # Initialize with first LLM call
        self._refresh_llm_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_observation(self, yield_value: float, *, yield_sigma=None, **params) -> None:
        if not np.isfinite(yield_value) or (yield_sigma is not None and (not np.isfinite(yield_sigma) or yield_sigma < 0)):
            raise ValueError("observation and uncertainty must be finite")
        obs = Observation(yield_value=yield_value, raw=dict(params), yield_sigma=yield_sigma)
        self.observations.append(obs)

        # Refresh LLM model (reasons over updated observation set)
        self._refresh_llm_model()

        # Init or update GPR
        if len(self.observations) >= self.gpr_threshold and self._gpr_model is None:
            self._gpr_model = GPRModel(self._param_space)
            xs = [o.raw for o in self.observations]
            ys = [o.yield_value for o in self.observations]
            sigmas = [o.yield_sigma for o in self.observations]
            if any(v is None for v in sigmas) and any(v is not None for v in sigmas):
                raise ValueError("all observations must use the same uncertainty protocol")
            self._gpr_model.fit(xs, ys, y_sigma=None if sigmas[0] is None else sigmas)

        elif self._gpr_model is not None:
            self._gpr_model.add_observation(params, yield_value, y_sigma=yield_sigma)

        self._update_mode()

    def suggest(self, method: str = "EI") -> Suggestion:
        if not self.observations:
            raise RuntimeError("Need at least one observation before suggesting.")
        if not self._param_space:
            raise RuntimeError("Cannot recommend without an explicit complete parameter domain")

        self._update_mode()

        ps = self._llm_model.param_space if self._llm_model else self._param_space
        # Activity hints may not delete coordinates from the GP's full input.
        ap = list(self._param_space)

        # Build mu/sigma function based on current mode
        if self._mode == "gpr" and self._gpr_model is not None and self._gpr_model._fitted:
            def mu_sigma(x: dict) -> Tuple[float, float]:
                return self._gpr_model.predict(x)
        elif self._llm_model is not None:
            def mu_sigma(x: dict) -> Tuple[float, float]:
                return self._llm_model.mu_func(x), self._llm_model.sigma_func(x)
        else:
            cp = ConservativePrior(ps if ps else {"_": (0, 1)})
            def mu_sigma(x: dict) -> Tuple[float, float]:
                return cp.mu(x), cp.sigma(x)

        y_best = self.y_best

        best_x, best_acq = _grid_search(
            param_space=ps,
            active_params=ap,
            mu_sigma_fn=mu_sigma,
            y_best=y_best,
            method=method,
            xi=self.xi,
            kappa=self.kappa,
            candidate_filter=self.candidate_filter,
            batch_predict_fn=(self._gpr_model.predict_many if self._mode == "gpr"
                              and self._gpr_model is not None and self._gpr_model._fitted else None),
        )

        # Round binary params
        for k, (lo, hi) in ps.items():
            if k in self.binary_params and k in best_x and lo == 0.0 and hi == 1.0:
                best_x[k] = int(round(best_x[k]))

        mu, sg = mu_sigma(best_x)
        acq_val = _ei(mu, sg, y_best, self.xi) if method == "EI" else _ucb(mu, sg, self.kappa)

        return Suggestion(
            params=best_x,
            mu_pred=float(mu),
            sigma_pred=float(sg),
            acq_method=method,
            acq_value=float(acq_val),
            ci_lo=float(mu - 1.96 * sg),
            ci_hi=float(mu + 1.96 * sg),
        )

    @property
    def y_best(self) -> float:
        if not self.observations:
            raise RuntimeError("No observations yet.")
        return max(o.yield_value for o in self.observations)

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    @property
    def model_mode(self) -> str:
        return self._mode

    def summary(self) -> dict:
        lines = [f"LLMDrivenBO -- {self.system_description[:60]}..."]
        lines.append(f"  Observations: {self.n_obs}")
        if self.n_obs > 0:
            lines.append(f"  y_best: {self.y_best:.2f}%")
            lines.append(f"  Model mode: {self._mode}")
            if self._llm_model is not None:
                lines.append(f"  Active params: {self._llm_model.active_params}")
                lines.append(f"  Reasoning: {self._llm_model.reasoning[:120]}")
        return {"text": "\n".join(lines), "llm_calls": len(self.llm_events)}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_llm_model(self) -> None:
        count = len(self.observations)
        if (count not in self.refresh_observation_counts or len(self.llm_events) >= self.max_llm_calls
                or any(event["observations"] == count for event in self.llm_events)):
            return
        event = {"call": len(self.llm_events) + 1, "observations": count}
        try:
            model = self.llm_reasoner(self.system_description, list(self.observations))
            if not isinstance(model, LLMModel):
                raise ValueError("legacy callback must return LLMModel")
            proposed = {key: tuple(value) for key, value in model.param_space.items()}
            if not proposed or any(len(v) != 2 or not np.isfinite(v).all() or v[0] >= v[1] for v in proposed.values()):
                raise ValueError("invalid proposed parameter bounds")
            if not self._param_space:
                self._param_space = MappingProxyType(dict(proposed))
            if set(proposed) - set(self._param_space):
                raise ValueError("LLM may not introduce coordinates outside the fixed full domain")
            clipped = {}
            for name, (hard_lo, hard_hi) in self._param_space.items():
                lo, hi = proposed.get(name, (hard_lo, hard_hi))
                lo, hi = max(lo, hard_lo), min(hi, hard_hi)
                if lo >= hi:
                    raise ValueError("proposed interval lies outside the full hard domain")
                clipped[name] = (lo, hi)
            self._llm_model = replace(model, param_space=clipped, active_params=list(self._param_space))
            self._active_params = list(self._param_space)
            event.update(status="accepted", proposed_bounds=proposed)
        except Exception as exc:
            event.update(status="failed_retaining_previous", error=f"{type(exc).__name__}: {exc}")
        self.llm_events.append(event)

    def _update_mode(self) -> None:
        n = len(self.observations)
        if n < 3:
            self._mode = "conservative"
        elif n < self.gpr_threshold:
            self._mode = "llm_guided"
        else:
            self._mode = "gpr"


# --------------------------------------------------------------------------
# Grid search acquisition (system-agnostic)
# --------------------------------------------------------------------------

def _grid_search(
    param_space: Dict[str, Tuple[float, float]],
    active_params: List[str],
    mu_sigma_fn: Callable[[dict], Tuple[float, float]],
    y_best: float,
    method: str = "EI",
    xi: float = 0.01,
    kappa: float = 2.0,
    n_grid_per_dim: int = 10,
    n_random: int = 20,
    candidate_filter=None,
    batch_predict_fn=None,
) -> Tuple[dict, float]:
    if not active_params or not param_space:
        return {}, 0.0

    rng = np.random.default_rng(42)

    grids = []
    for p in active_params:
        lo, hi = param_space[p]
        grids.append(np.linspace(lo, hi, n_grid_per_dim))

    candidates = []
    if n_grid_per_dim ** len(active_params) <= 8192:
        mesh = np.meshgrid(*grids, indexing="ij")
        flat_grids = [mg.ravel() for mg in mesh]
        for flat_idx in range(flat_grids[0].size):
            candidates.append({p: _cast(flat_grids[i][flat_idx]) for i, p in enumerate(active_params)})
    else:
        for _ in range(2048):
            candidates.append({p: float(rng.uniform(*param_space[p])) for p in active_params})

    for _ in range(n_random):
        cand = {p: rng.uniform(*param_space[p]) for p in active_params}
        candidates.append(cand)

    if candidate_filter is not None:
        candidates = [x for x in candidates if candidate_filter(x)]
    if not candidates:
        raise ValueError("no candidates satisfy the fixed hard-domain filter")
    if batch_predict_fn is not None:
        means, sigmas = batch_predict_fn(candidates)
        values = (np.array([_ei(mu, sigma, y_best, xi) for mu, sigma in zip(means, sigmas)])
                  if method == "EI" else means + kappa * sigmas)
        index = int(np.argmax(values))
        return candidates[index], float(values[index])

    best_acq = -np.inf
    best_x = None

    for x in candidates:
        mu, sigma = mu_sigma_fn(x)
        if method == "EI":
            acq = _ei(mu, sigma, y_best, xi)
        else:
            acq = _ucb(mu, sigma, kappa)

        if acq > best_acq:
            best_acq = acq
            best_x = x

    return (best_x or {}, best_acq)


# --------------------------------------------------------------------------
# Built-in heuristic reasoner (no LLM needed -- for testing/fallback)
# --------------------------------------------------------------------------

def heuristic_reasoner(
    system_description: str,
    observations: List[Observation],
) -> LLMModel:
    """
    Keyword-based reasoner.  No LLM API call needed.

    Detects system type from system_description and generates a simple
    yield model + param space.  Used as:
      1. Default when llm_reasoner=None
      2. Baseline to compare against real LLM-driven models

    For production: replace this with a real LLM API call that returns
    an LLMModel with actual physics-informed mu_func/sigma_func.
    """
    desc = system_description.lower()

    # ---- ErTa7O19 oxide system ----
    if "erta" in desc or "ertao" in desc or "er2o3" in desc:
        param_space = {
            "T_dwell": (1050, 1320),
            "T_max": (1100, 1450),
            "grind": (0, 1),
            "step_down": (0, 1),
            "dwell2_min": (0, 480),
            "dwell1_min": (360, 600),
        }
        active_params = ["T_dwell", "T_max", "grind", "step_down", "dwell1_min", "dwell2_min"]

        def mu_func(x):
            T = _cast(x.get("T_dwell", 1100))
            grind = int(round(_cast(x.get("grind", 0))))
            sd = int(round(_cast(x.get("step_down", 1))))
            d1 = _cast(x.get("dwell1_min", 480))
            d2 = _cast(x.get("dwell2_min", 300))
            base = 24.0
            grind_corr = -2.0 if grind == 1 else 0.0
            sd_corr = -1.0 if sd == 1 else 0.0
            temp_corr = (T - 1100) * 0.05
            time_corr = np.log1p(d1) * 0.5 + np.log1p(d2) * 0.3
            return max(0.0, base + temp_corr + time_corr + grind_corr + sd_corr)

        def sigma_func(x):
            T = _cast(x.get("T_dwell", 1100))
            s_extrap = 2.0 if T > 1300 else 1.5
            return (1.5**2 + s_extrap**2 + 2.0**2 + 1.0**2) ** 0.5

        reasoning = "ErTa7O19 oxide KB. Temp + time corrections."

    # ---- CeAl2Ge2 intermetallic ----
    elif "ceal2ge2" in desc or "ce-al-ge" in desc or "al-ce-ge" in desc or "ce2al3ge4" in desc:
        param_space = {
            "AlCe_ratio": (1.5, 20.0),
            "anneal_temp": (0, 900),
            "anneal_hours": (0, 168),
        }
        active_params = ["AlCe_ratio", "anneal_temp", "anneal_hours"]

        def mu_func(x):
            alce = _cast(x.get("AlCe_ratio", 6.0))
            ann_t = _cast(x.get("anneal_temp", 0))
            ann_h = _cast(x.get("anneal_hours", 0))
            # Parabolic around AlCe ~6 (observed optimum)
            alce_corr = -0.8 * (alce - 6.0) ** 2 + 15.0
            # Annealing suppresses metastable target (promotes ground state Ce2Al3Ge4)
            ann_corr = -0.015 * ann_t
            ann_corr += 3.0 if 0 < ann_h <= 24 else 0.0
            return max(0.0, min(100.0, 50.0 + alce_corr + ann_corr))

        def sigma_func(x):
            alce = _cast(x.get("AlCe_ratio", 6.0))
            s_alce = 5.0 if alce < 3.0 else 2.5
            s_extrap = 3.0 if _cast(x.get("anneal_temp", 0)) > 700 else 1.5
            return (s_alce**2 + s_extrap**2 + 3.0**2) ** 0.5

        reasoning = "CeAl2Ge2 intermetallic. AlCe parabolic ~6, annealing suppresses target."

    # ---- Generic oxide fallback ----
    else:
        param_space = {"T_sinter": (800, 1500), "hold_h": (1, 12)}
        active_params = ["T_sinter", "hold_h"]

        def mu_func(x):
            T = _cast(x.get("T_sinter", 1000))
            h = _cast(x.get("hold_h", 4))
            return max(0.0, 0.03 * (T - 800) + 3.0 * np.log(h + 1) + 40)

        def sigma_func(x):
            return 5.0

        reasoning = "Generic oxide. Arrhenius + log-time model."

    return LLMModel(
        mu_func=mu_func,
        sigma_func=sigma_func,
        param_space=param_space,
        active_params=active_params,
        reasoning=reasoning,
        model_code=f"# Heuristic reasoner (no LLM)\n# {reasoning}",
    )
