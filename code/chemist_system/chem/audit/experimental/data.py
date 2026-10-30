
#!/usr/bin/env python3
import itertools


def _mprester():
    """Lazy import: mp_api pulls a libstdc++ that can clash with scipy's
    extensions if loaded at module scope."""
    from mp_api.client import MPRester
    return MPRester

def get_all_sub_chemsys(chem_sys_str):
    """
    将 "Ba-Ti-O-C" 转换为所有可能的子系统组合列表。
    并且严格按照字母顺序排序 (MP 数据库的要求，例如 C-O 而不是 O-C)。
    """
    elements = chem_sys_str.split('-')
    all_combinations = []
    
    # 遍历 1元, 2元, 3元, 4元 组合
    for i in range(1, len(elements) + 1):
        for combo in itertools.combinations(elements, i):
            # MP API 要求 chemsys 字符串内的元素必须按字母表排序
            sorted_combo = sorted(combo)
            all_combinations.append("-".join(sorted_combo))
            
    return all_combinations

def fetch_phases_cleaned(chem_sys, api_key):
    '''
    Fetch ALL sub-phases using MP Summary API to get Standard Formation Energies.
    '''
    # 1. 获取所有子系统的列表
    # 对于 "Ba-Ti-O-C"，会生成 ['Ba', 'C', 'O', 'Ti', 'Ba-C', 'C-O', ..., 'Ba-C-O-Ti']
    search_systems = get_all_sub_chemsys(chem_sys)
    
    # print(f"  [Data] Fetching full network for {chem_sys}...")
    # print(f"  [Data] Querying {len(search_systems)} sub-systems: {search_systems}")
    
    MPRester = _mprester()
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        # 2. 将列表传给 chemsys 参数，API 会自动把它们视为 OR (或) 逻辑
        docs = mpr.materials.summary.search(
            chemsys=search_systems, 
            fields=[
                "formula_pretty", 
                "composition", 
                "formation_energy_per_atom", 
                "energy_per_atom",           
                "volume"                     
            ]
        )
        
        unique_phases = {}
        for doc in docs:
            if doc.formation_energy_per_atom is None:
                continue
                
            formula = doc.formula_pretty
            
            # 保留最稳定的同质异构体
            if formula not in unique_phases or doc.energy_per_atom < unique_phases[formula]['energy_per_atom']:
                unique_phases[formula] = {
                    'formula_pretty': formula,
                    'composition': doc.composition, 
                    'formation_energy_per_atom': doc.formation_energy_per_atom,
                    'energy_per_atom': doc.energy_per_atom,
                    'volume': doc.volume if doc.volume else 0.0
                }
        
        result = sorted(unique_phases.values(), key=lambda x: x['formula_pretty'])
        # print(f"  [Data] Successfully retrieved {len(result)} unique phases.")
        return result
