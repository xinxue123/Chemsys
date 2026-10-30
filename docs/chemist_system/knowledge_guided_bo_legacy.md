# Knowledge-Guided Bayesian Optimization — SKILL

## Metadata
- **name**: knowledge-guided-bo
- **category**: materials-science / autonomous-synthesis
- **trigger**: When user asks to run Bayesian optimization for ErTa7O19 synthesis, or any solid-state sintering BO task
- **trigger conditions**: ["bayesian", "BO", "yield prediction", "synthesis optimization", "ErTa7O19"]
- **description**: Replaces GPR-based BO with a physics/chemistry knowledge base for predicting μ (mean yield) and σ (uncertainty). Provides interpretable, robust acquisition even with n < 10 data points.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  knowledge_guided_bo (sub-agent)                              │
│                                                              │
│  ┌────────────────┐    ┌─────────────────────────────┐    │
│  │ knowledge_base  │    │ acquisition.py               │    │
│  │  • mu_physics()│    │  • ei_knowledge()            │    │
│  │  • sigma_phys()│    │  • ucb_knowledge()           │    │
│  │  • KnowledgeBase   │  │  • generate_candidates()      │    │
│  └───────┬────────┘    └──────────────┬──────────────────┘    │
│          │                            │                        │
│          └──────────┬─────────────────┘                        │
│                     ▼                                         │
│              ┌──────────────┐                                 │
│              │  core.py      │                                 │
│              │  Knowledge    │  ← orchestrates KB + acquisition│
│              │  GuidedBO     │                                 │
│              └──────┬───────┘                                 │
│                     │                                          │
│                     ▼  suggest() → Suggestion{                 │
│                        T_dwell, T_max, grind, yield_pred, σ   │
└──────────────────────────────────────────────────────────────┘
```

## Fixed Workflow (execute in order)

### Step 1 — Load knowledge base module
```python
import sys
sys.path.insert(0, '/home/ucas/.hermes/plugins_scripts')
from knowledge_guided_bo import KnowledgeGuidedBO, KnowledgeBase
```

### Step 2 — Initialize BO agent
```python
bo = KnowledgeGuidedBO(
    system="ErTa7O19",
    fixed_params={
        "step_down": 1,       # always use step-down protocol
        "dwell2_min": 360.0, # always use second dwell
        "dwell1_min": 360.0, # standard first dwell
    }
)
```

### Step 3 — Add observations
For each experimental cycle, add via `add_observation()`:
```python
bo.add_observation(
    T_dwell=1100,            # °C
    T_max=1100,              # °C (same as dwell = no high-T decomposition)
    grind=0,                 # 0 = no intermediate grind, 1 = grind
    yield_value=22.5,        # % from XRD RIR analysis
    step_down=0,             # for older cycles
    dwell2_min=0,            # no second dwell
    xrd_file="path/to/xrd.txt"
)
```

### Step 4 — Run suggest
```python
suggestion = bo.suggest(
    n_candidates=5000,   # candidate grid size
    method="EI",         # "EI" or "UCB"
    kappa=2.0,           # for UCB only
    return_top_k=3       # get top-3 alternatives
)
```

### Step 5 — Parse suggestion output
```python
s = suggestion  # Suggestion dataclass
print(f"T_dwell: {s.T_dwell:.0f}°C")
print(f"T_max:   {s.T_max:.0f}°C")
print(f"grind:   {s.grind}")
print(f"yield_pred: {s.mu_pred:.1f}% ± {s.sigma_pred:.1f}%")
print(f"EI:      {s.acq_value:.3f}")
print(f"95% CI:  [{s.mu_pred - 2*s.sigma_pred:.1f}%, {s.mu_pred + 2*s.sigma_pred:.1f}%]")
```

### Step 6 — Generate report
```python
print(bo.report())           # human-readable
summary = bo.summary_dict()  # machine-readable dict
```

## Output Schema (Suggestion dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `T_dwell` | float | 主保温温度 (°C) |
| `T_max` | float | 最高分解温度 (°C) |
| `grind` | int | 中间研磨 (0/1) |
| `step_down` | int | 逐级降温 (0/1) |
| `dwell1_min` | float | 主段保温时间 (min) |
| `dwell2_min` | float | 第二段保温时间 (min) |
| `mu_pred` | float | 预测产量 (%) |
| `sigma_pred` | float | 预测不确定性 (%) |
| `acq_value` | float | EI 或 UCB 分值 |
| `acq_method` | str | "EI" or "UCB" |
| `rank` | int | 在候选池中的排名 |

## Physics Model (mu_physics)

```
μ(T_dwell) = YIELD_MAX × sigmoid((T_dwell - 1032) / 80)
             × f_kinetic(T_dwell, dwell_hours)
             + Δgrind (+2.1%) + ΔTmax (>1350°C, +3-5%) + Δstep_down (+1.5%)

YIELD_MAX = 42.0%  (ErTaO4 competing phase ceiling, empirical from C6=40.4%)
f_kinetic = [1 - 0.234·exp(-2.32·ΔT/100)] × [1 - exp(-0.5·dwell_hours)]
```

## Uncertainty Model (sigma_physics)

```
σ = √(σ²_meas + σ²_extrap + σ²_model + σ²_Tmax + σ²_param)
  σ_meas    = 1.5%    (XRD RIR repeatability)
  σ_extrap  = 1.5-10% (T_dwell 外推惩罚)
  σ_model   = 2.0%    (sigmoid 参数不确定性)
  σ_Tmax    = 2-4%    (T_max 效应稀疏数据)
  σ_param   = 0.5-2% (grind/step/dwell2 校正项)
```

## When to Use

| Scenario | Use GPR BO | Use Knowledge-Guided BO |
|----------|-----------|------------------------|
| n > 20 data points | ✓ | possible |
| n < 10 data points | ✗ (sparse) | ✓ (designed for this) |
| Need interpretable μ/σ | ✗ (black box) | ✓ (physics-based) |
| Quick prediction | ✗ (slow) | ✓ (instant) |
| Need to combine with GPR | — | ✓ (hybrid mode available) |

## Key Advantages over GPR BO

1. **No length-scale pathology**: Knowledge-guided BO doesn't suffer from
   length_scale → 0 or → ∞ when n << d
2. **Interpretable σ**: σ comes from physical sources, not posterior variance
3. **Robust to sparse data**: 3 points or 30 points, behavior is predictable
4. **Composable**: Can be used as a prior and combined with GPR in hybrid mode
5. **Fast**: No GPR fitting needed, only μ/σ evaluation → < 1ms per candidate

## Example: ErTa7O19 with 6 cycles

```python
bo = KnowledgeGuidedBO("ErTa7O19")
bo.add_observation(T_dwell=1100, T_max=1100, grind=0, yield_value=22.5)
bo.add_observation(T_dwell=1100, T_max=1100, grind=0, yield_value=23.6)
bo.add_observation(T_dwell=1100, T_max=1100, grind=0, yield_value=22.3)
bo.add_observation(T_dwell=1100, T_max=1100, grind=1, dwell2_min=360, yield_value=25.7)
bo.add_observation(T_dwell=1200, T_max=1200, grind=0, yield_value=37.6)
bo.add_observation(T_dwell=1200, T_max=1400, grind=1, dwell2_min=360,
                   step_down=1, yield_value=40.4)

sugg = bo.suggest(n_candidates=10000, method="EI")
# → T_dwell=1300°C, T_max=1400°C, grind=1, μ=42.0%, σ=11.4%, EI=5.13
```

## Integration with Main Agent

The main agent (hermes) should:

1. **Delegate to this sub-agent** when user asks for synthesis BO
2. **Pass experimental results** from XRD analysis as observations
3. **Use the Suggestion output** to generate robot commands or synthesis JSON
4. **Pass data back** to bayesian_optimizer.py for hybrid GPR+KB mode

## Troubleshooting

**Problem**: `mu_pred` always returns the same value regardless of T_dwell
→ **Cause**: T_dwell is hitting the sigmoid ceiling at 42% (yield_max)
→ **Fix**: This is expected when T_dwell > 1250°C — physics saturates at 42%

**Problem**: σ is too small for extrapolation regions
→ **Cause**: sigma_extrap formula not aggressive enough
→ **Fix**: Adjust sigma_extrap thresholds in `knowledge_base.py`
