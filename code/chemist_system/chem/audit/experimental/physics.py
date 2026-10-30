#!/home/ucas/miniconda3/envs/python12/bin/python
import numpy as np
from pymatgen.core import Composition
import cantera as ct

# 初始化 Cantera 环境
gas_sol = ct.Solution('gri30.yaml')
kB = 8.617333e-5 
GAS_PHASES = ["CO2", "O2", "CO", "H2O", "N2", "SO2", "Ar"]

# 标准空气环境下的气体天然本底分压 (单位: atm)
AMBIENT_AIR_PRESSURES = {
    "N2": 0.7808,
    "O2": 0.2095,
    "H2O": 0.02,     # 实验室常规湿度约 2%
    "CO2": 0.0004,   # 400 ppm
    "Ar": 0.0093
}
# 对于空气中不存在的气体 (如 SO2, Cl2)，设定极低的默认“逃逸背压”
DEFAULT_TRACE_PRESSURE = 1e-5
# 当你把上面提取出的字典丢给 Phase 类时：
class Phase:
    def __init__(self, data_dict):
        self.name = data_dict['formula_pretty']
        
        # 处理组成和 Factor
        comp_obj = Composition(data_dict['composition'])
        _, factor = comp_obj.get_reduced_composition_and_factor()
        self.composition = {str(k): v / factor for k, v in comp_obj.as_dict().items()}
        self.n_atoms = sum(self.composition.values())
        
        # 【核心物理锚点】：生成能 (eV/FU)
        self.h_f_298 = data_dict['formation_energy_per_atom'] * self.n_atoms
        
        # 结构与质量参数 (供 SISSO 描述符使用)
        total_mass = sum([Composition(el).weight * amt for el, amt in self.composition.items()])
        self.m_avg = total_mass / self.n_atoms
        self.v_pa = (data_dict['volume'] / factor) / self.n_atoms if self.n_atoms > 0 else 15.0

def get_thermal_correction_gas(name, T):
    """
    计算气体从 298.15K 到 T 的自由能增量 [G(T) - G(298.15)]
    """
    try:
        gas_sol.X = {name: 1.0}
        
        # 目标温度下的 Gibbs 自由能
        gas_sol.TP = T, 101325
        g_t = gas_sol.gibbs_mole
        
        # 298.15K 基准的 Gibbs 自由能
        gas_sol.TP = 298.15, 101325
        g_298 = gas_sol.gibbs_mole
        
        return (g_t - g_298) / 1000.0 / 96485.3 # 转换为 eV/FU
    except Exception as e:
        # 兜底：如果 Cantera 找不到某种特殊气体，返回一个粗略的经验熵衰减
        return -0.0022 * (T - 298.15) 


# ==========================================
# 核心函数：高精度化学势计算
# ==========================================
def get_Phi_FU(phase, T, interventions=None):
    """
    计算特定物相在 T 温度下的绝对化学势 (基于 298.15K 生成能基准)
    
    :param phase: Phase 对象 (必须具有 name, h_f_298, n_atoms 属性)
    :param T: 当前温度 (K)
    :param interventions: 外部/智能体干预字典，可重写气压或施加能量偏置
    :return: 修正后的化学势 (eV/FU)
    """
    interventions = interventions or {}
    
    # 1. 物理锚点：298.15K 标准生成焓 (确保气固在同一基准线上)
    G = phase.h_f_298 

    # 2. 气体相动态处理
    if phase.name in GAS_PHASES:
        # (A) 热力学演化: G(T) - G(298)
        dg_thermal = get_thermal_correction_gas(phase.name, T)
        
        # (B) 物理驱动力: 勒夏特列排气效应 RT * ln(P_eff)
        # 优先级: 用户指定 > 空气本底字典 > 默认痕量分压
        if 'effective_pressure' in interventions:
            p_eff = interventions['effective_pressure']
        else:
            p_eff = AMBIENT_AIR_PRESSURES.get(phase.name, DEFAULT_TRACE_PRESSURE)
            
        p_correction = kB * T * np.log(p_eff)
        
        G += dg_thermal + p_correction
        
    # 3. 固相物理处理: Dulong-Petit 极限热容积分
    else:
        if T > 298.15:
            # 晶格高温热容极限: 3 * N * k_B
            c_p_limit = 3 * phase.n_atoms * kB
            
            # 严格推导的热力学自由能积分: dG = dH - T*dS
            delta_g_vib = c_p_limit * ((T - 298.15) - T * np.log(T / 298.15))
            
            G += delta_g_vib
        
    # 4. 实验误差 / 智能体干预对齐 (Bias)
    # Gemma 可以在这里给 BaCO3 等难以分解的物质施加人为偏置
    G += interventions.get('g_bias', {}).get(phase.name, 0.0)
    
    # 5. [调试专用] 核心组分日志输出
    # if phase.name in ["BaCO3", "TiO2", "BaTiO3", "CO2"]:
    #     print(f"  [Phi] {phase.name} at {int(T)}K: G={G:.4f} eV/FU ")
        
    return G