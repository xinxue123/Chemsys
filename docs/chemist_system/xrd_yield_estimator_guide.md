# Skill: XRD Yield Estimation + Knowledge-Guided BO

## Purpose

Complete closed-loop system: run synthesis → measure XRD → estimate phase yield → Bayesian Optimization → next experiment.

> **Note (CeAl2Ge2 arc-melt study).** For the published CeAl₂Ge₂ results the
> primary quantifier is the agent-driven GSAS-II dynamic Rietveld backend
> (`create_estimator(system, backend="gsas2")`, the `auto` default for `ceal2ge2`);
> the PeakBasedRI / RIR estimator below is retained as the zero-install,
> structure-derived **cross-check**, not the primary method.

## Architecture

```
烧制方案 ──→ [机械臂执行] ──→ 产物粉末
                                  │
                            [XRD表征]
                                  │ 2θ-I 数据
                                  ▼
                    ┌─────────────────────────┐
                    │  XRD Yield Estimator    │
                    │  • PeakBasedRI (当前)   │ ←── 默认，无需安装
                    │  • RietveldRefiner(GSAS-II) │ ←── 精度更高（可选）
                    └────────────┬────────────┘
                                 │ yield ± σ
                                 ▼
                    ┌─────────────────────────┐
                    │  KnowledgeGuided BO      │
                    │  add_observation(params, │
                    │    yield_value)          │
                    │  suggest() → 下一推荐     │
                    └─────────────────────────┘
```

## Quick Start

### Step 1 — 安装依赖

```bash
# 无需安装额外依赖（PeakBasedRI 使用 scipy + numpy）
pip install numpy scipy matplotlib
```

### Step 2 — 使用已有 6 cycle 数据测试

```bash
python xrd_bo_loop.py \
  --xrd /home/ucas/workspace/xrd/20260428-1400度1200度/ErTaO_20260428_162240.txt \
  --params '{"T_dwell":1200,"T_max":1400,"grind":1,"step_down":1,"dwell2_min":360}' \
  --system erta7o19
```

### Step 3 — 安装 GSAS-II（Rietveld 精修，~1-2% 精度）

```bash
# Download from https://www.aps.anl.gov/Actors-and-Parallelism/GSAS-II
wget https://github.com/AdvancedPhotonSource/GSAS-II/releases/download/v2025.2/GSASII.tgz
tar -xzf GSASII.tgz
export GSASIIPATH=/path/to/GSASII

# Use Rietveld backend
python xrd_bo_loop.py --xrd Cycle7.txt --use-rietveld
```

## 添加新材料系统

只需实现 `SystemPlugin` + 配置 XRD 峰位：

```python
from knowledge_guided_bo import KnowledgeGuidedBO, SystemPlugin, register_plugin
from xrd_yield_estimator import SYSTEM_CONFIGS

class BaTiO3Plugin(SystemPlugin):
    name = "batio3"
    param_space = {
        "T_sinter": (1200, 1350),
        "hold_h":   (1, 4),
        "grind":    (0, 1),
    }
    
    def yield_model(self, x):
        # 物理/经验公式
        T = x["T_sinter"]
        h = x["hold_h"]
        return min(60 + 0.04*(T-1200) + 3*h, 99.0)
    
    def sigma_sources(self, x):
        return {
            "meas":   1.5,
            "extrap": 3.0 if x["T_sinter"] > 1300 else 1.0,
            "model":  2.0,
        }

register_plugin("batio3", BaTiO3Plugin)

# 添加 XRD 配置（峰位，校准数据）
SYSTEM_CONFIGS["batio3"] = {
    "target_phase": "BaTiO3",
    "phases": {
        "BaTiO3": {"peaks_2theta": [45.3], "preferred": 0},
        "BaCO3":  {"peaks_2theta": [24.9], "preferred": 0},
    },
    "calibration": {
        "slope": 1.0, "intercept": 0.0,  # 替换为实际校准数据
        "cal_sigma": 3.0,
    },
}

# 运行 BO
bo = KnowledgeGuidedBO("batio3")
bo.add_observation(T_sinter=1250, hold_h=2, grind=1, yield_value=72.0)
s = bo.suggest()
```

## 关键参数说明

| 参数 | 含义 |
|------|------|
| `T_dwell` | 保温温度 °C |
| `T_max` | 最高烧结温度 °C |
| `grind` | 是否研磨（0/1） |
| `step_down` | 是否阶梯降温（0/1） |
| `dwell2_min` | 第二保温时间 min |

## PeakBasedRI 精度说明

| 情况 | 精度 | 说明 |
|------|------|------|
| 已有校准数据（≥3点） | ~2-3% MAE | 当前模式 |
| 无校准，纯 RIR 法 | 5-10% | 仅用于快速估计 |
| GSAS-II Rietveld | 1-2% | 需安装 GSAS-II |

**注意**：固相合成 XRD 峰比值法有系统性限制（结晶度、择优取向随温度变化）。最佳做法是用 GSAS-II Rietveld 建立校准，再将校准参数填入 `SYSTEM_CONFIGS`。

## 文件结构

```
/home/ucas/.hermes/plugins_scripts/
├── xrd_yield_estimator/
│   ├── __init__.py       ← 公共 API: create_estimator(), YieldResult, SYSTEM_CONFIGS
│   ├── peak_based.py     ← PeakBasedRI 实现（SNIP 背景 + 峰积分 + Chung RIR）
│   ├── rietveld.py       ← GSAS-II 后端模板（装好 GSAS-II 后替换）
│   ├── xrd_bo_loop.py   ← 闭环脚本
│   └── SKILL.md          ← 本文件
└── knowledge_guided_bo/
    ├── __init__.py       ← KnowledgeGuidedBO, SystemPlugin, get_plugin, register_plugin
    ├── knowledge_base.py  ← 物理模型 + generic_mu/sigma/ei_generic
    └── core.py           ← 调度器
```
