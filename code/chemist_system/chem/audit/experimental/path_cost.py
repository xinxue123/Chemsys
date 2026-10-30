#!/home/ucas/miniconda3/envs/python12/bin/python
r"""
Path Cost Model: Quantifying Synthesis Pathway Quality.

This module implements the central contribution of our framework:
converting the complex ion-correlation transport essence into a 
single, physically-meaningful scalar: the path cost.

Path Cost = w_thermo * Cost_thermo + w_kinetic * Cost_kinetic + w_ion * Cost_ion

Where:
  Cost_thermo = \Delta G_reaction / (k_B T) — normalized thermodynamic driving force
  Cost_kinetic = E_a / (k_B T)              — kinetic barrier height  
  Cost_ion     = ln(D_0/K_D)/ln(10)         — transport resistance via ion-correlation

This transforms the "alchemy" of solid-state synthesis into
a well-defined optimization landscape where AI agents can search
for minimum-cost synthesis pathways.
"""

import numpy as np
import warnings
from .ion_transport import IonCorrelationModel

kB = 8.617333262e-5  # Boltzmann in eV/K


class PathCostModel:
    """
    Quantify the 'cost' or 'difficulty' index for each synthesis pathway.
    
    Lower cost = more favorable pathway.
    
    The cost function integrates:
    1. Thermodynamic cost: energetic favorability of each reaction step
    2. Kinetic cost: activation energy barrier from BEP scaling
    3. Ion-correlation cost: transport resistance from KD ratios
    """
    
    def __init__(self, alpha_bep=0.5, Ea_base=1.25):
        """
        Parameters
        ----------
        alpha_bep : float
            BEP linear coefficient. Literature value 0.5 (Nørskov et al.)
        Ea_base : float
            Base activation energy in eV for solid-state reactions
        """
        self.alpha_bep = alpha_bep
        self.Ea_base = Ea_base
        self.ion_model = IonCorrelationModel(D0_ref=5e-3, Ea_base=1.6)
        
        # Weights for total cost — tuned to balance thermodynamic and kinetic terms
        # These can be adjusted per system based on empirical calibration
        self.w_thermo = 1.0
        self.w_kinetic = 1.5   # Slightly higher weight: barriers dominate kinetics
        self.w_ion = 2.0       # Strongest weight: transport selectivity is key insight
        
    def compute_reaction_cost(self, dG, T):
        """
        Compute thermodynamic + kinetic cost for a single reaction.
        
        Parameters
        ----------
        dG : float
            Reaction free energy change (eV per atom)
        T : float
            Temperature in Kelvin
            
        Returns
        -------
        tuple: (cost_thermo, cost_kinetic, cost_total)
        """
        cost_thermo = -dG / (kB * T) if T > 0 else 0.0
        
        # BEP-based activation energy: E_a = Ea_base + alpha * dG_n
        Ea = max(self.Ea_base + self.alpha_bep * dG, 0.05)
        cost_kinetic = Ea / (kB * T) if T > 0 else 0.0
        
        cost_total = self.w_thermo * cost_thermo + self.w_kinetic * cost_kinetic
        
        return cost_thermo, cost_kinetic, cost_total
    
    def compute_path_cost(self, pathway, T, h_f_298_dict,
                          product_KD_dict=None,
                          intermediates_penalty=True,
                          trap_penalty=True):
        """
        Compute the total path cost for an entire synthesis pathway.
        
        Parameters
        ----------
        pathway : list of dicts
            Each dict contains: {'reactants': [...], 'products': [...], 
                                  'coefficients': {...}}
        T : float
            Temperature in Kelvin
        h_f_298_dict : dict
            phase_name -> formation_energy_per_atom at 298K
        product_KD_dict : dict
            phase_name -> KD value (cm^2/s) from ion-correlation model
        intermediates_penalty : bool
            Add penalty for each intermediate step
        trap_penalty : bool
            Add penalty when stable competitors exist (interception trap)
            
        Returns
        -------
        dict with per-step costs, total cost, trap report, path quality score
        """
        step_costs = []
        total_cost = 0.0
        trap_report = []
        
        for i, step in enumerate(pathway):
            reactants = step.get('reactants', [])
            products = step.get('products', [])
            
            # Compute thermodynamic driving force
            G_reactants = sum(h_f_298_dict.get(r, 0.0) for r in reactants)
            G_products = sum(h_f_298_dict.get(p, 0.0) for p in products)
            dG = G_products - G_reactants  # negative = exothermic
            
            # Per-step costs
            c_th, c_kn, c_step = self.compute_reaction_cost(dG, T)
            
            # Ion-correlation transport penalty on products
            c_ion = 0.0
            if product_KD_dict:
                for prod in products:
                    KD = product_KD_dict.get(prod, np.nan)
                    if np.isfinite(KD) and KD > 0:
                        # Cost inversely proportional to KD
                        # Normalized to reference KD = 1e-5 cm^2/s
                        KD_ref = 1e-5
                        c_ion += self.w_ion * max(0, np.log10(KD_ref / KD))
                    else:
                        c_ion += self.w_ion * 2.0  # Unknown phase: moderate penalty
            
            # Trap penalty: products that are thermodynamically disfavored
            # relative to competing phases get intercepted
            if trap_penalty:
                competing = self._find_competitors(products, h_f_298_dict, T)
                if competing:
                    c_ion += self.w_ion * 0.5 * len(competitors := competing)
                    trap_report.append({
                        'step': i,
                        'product': products[0],
                        'competitor': competing,
                        'penalty': 0.5 * len(competitors)
                    })
            
            step_cost = c_th + c_kn + c_ion
            step_costs.append({
                'step': i,
                'reaction': f"{'+'.join(reactants)} -> {'+'.join(products)}",
                'dG': dG,
                'cost_thermo': c_th,
                'cost_kinetic': c_kn,
                'cost_ion': c_ion,
                'cost_total': step_cost
            })
            
            total_cost += step_cost
        
        # Intermediates penalty: fewer steps = more direct route
        intermediate_penalty = 0.0
        if intermediates_penalty and len(pathway) > 1:
            intermediate_penalty = 0.8 * (len(pathway) - 1)
            total_cost += intermediate_penalty
        
        # Path quality score (lower cost = better path, 0-100 scale)
        n_steps = len(pathway)
        if n_steps > 0:
            avg_cost = total_cost / n_steps
            quality_score = max(0, 100 - avg_cost * 15)
        else:
            quality_score = 0
            
        return {
            'step_costs': step_costs,
            'total_cost': total_cost,
            'intermediate_penalty': intermediate_penalty,
            'trap_report': trap_report,
            'quality_score': quality_score,
            'n_steps': n_steps,
            'temperature': T
        }
    
    def _find_competitors(self, product_phases, h_f_298_dict, T):
        """
        Find thermodynamically more stable competitors to product phases.
        
        This implements the "interception trap" concept: when a synthesis
        target is not the global free energy minimum, kinetic detours are needed.
        """
        competitors = []
        for target in product_phases:
            target_G = h_f_298_dict.get(target, np.inf)
            if target_G == np.inf:
                continue
            for phase, G in h_f_298_dict.items():
                if phase != target and G < target_G - 0.01:
                    # More stable by >10 meV: potential competitor
                    competitors.append(phase)
        return list(set(competitors))[:3]
    
    def rank_pathways(self, pathways, T, h_f_298_dict, product_KD_dict=None):
        """
        Rank multiple candidate pathways by their total cost.
        
        Parameters
        ----------
        pathways : list of list of dicts
            Multiple candidate synthesis routes
        T : float
            Temperature
        h_f_298_dict : dict
            Phase formation energies
        product_KD_dict : dict
            Phase KD values
            
        Returns
        -------
        list of (pathway_idx, cost_result) sorted by total_cost ascending
        """
        results = []
        for i, pathway in enumerate(pathways):
            cost_result = self.compute_path_cost(
                pathway, T, h_f_298_dict, product_KD_dict
            )
            results.append((i, cost_result))
        
        # Sort by total cost (ascending = better paths first)
        results.sort(key=lambda x: x[1]['total_cost'])
        
        for rank, (idx, res) in enumerate(results):
            res['rank'] = rank + 1
            
        return results


def illustrate_path_cost():
    """Show path cost computation for the Ba-Ti-O system."""
    
    # Formation energies (eV/atom) from MP DFT
    h_f = {
        'BaO':    -5.38,    # Reference
        'TiO2':   -9.40,    # Reference (rutile stable)
        'Ba2TiO4': -6.52,   # Most stable intermediate
        'BaTiO3': -6.42,    # Target (perovskite)
        'BaTi2O5':-6.28,    # Competing metastable
        'BaTi4O9':-6.35,    # Ti-rich product
        'BaCO3':  -4.88,    # Precursor (decomposes ~1050°C)
        'CO2':     0.0,     # Reference gas phase
    }
    
    model = PathCostModel(alpha_bep=0.5, Ea_base=1.25)
    ion_model = IonCorrelationModel(D0_ref=5e-3, Ea_base=1.6)
    
    # Compute KD values for each phase
    KD_dict = {
        'BaO':     ion_model.compute_KD('BaO', 1273),
        'TiO2':    ion_model.compute_KD('TiO2', 1273),
        'Ba2TiO4': ion_model.compute_KD('Ba2TiO4', 1273),
        'BaTiO3':  ion_model.compute_KD('BaTiO3', 1273),
        'BaTi2O5': ion_model.compute_KD('BaTi2O5', 1273),
        'BaTi4O9': ion_model.compute_KD('BaTi4O9', 1273),
    }
    
    T = 1273.0  # 1000°C
    
    # Define candidate pathways
    pathways = [
        # Path A: Ba2TiO4 intermediate (observed in experiments)
        [{'reactants': ['BaO', 'TiO2'], 'products': ['Ba2TiO4']},
         {'reactants': ['Ba2TiO4', 'TiO2'], 'products': ['BaTiO3']}],
        
        # Path B: Direct BaTiO3 formation
        [{'reactants': ['BaO', 'TiO2'], 'products': ['BaTiO3']}],
        
        # Path C: BaTi2O5 intermediate (often observed as byproduct)
        [{'reactants': ['BaO', 'TiO2'], 'products': ['BaTi2O5']},
         {'reactants': ['BaTi2O5', 'TiO2'], 'products': ['BaTiO3']}],
         
        # Path D: Ti-rich intermediate route
        [{'reactants': ['BaO', 'TiO2'], 'products': ['BaTi4O9']},
         {'reactants': ['BaTi4O9', 'BaO'], 'products': ['BaTiO3']}],
    ]
    
    path_names = ['A: Ba2TiO4 intermediate', 
                  'B: Direct path',
                  'C: BaTi2O5 intermediate',
                  'D: Ti-rich intermediate']
    
    print("=== Path Cost Analysis: BaO + TiO2 -> BaTiO3 @ 1273K ===\n")
    
    all_results = model.rank_pathways(pathways, T, h_f, KD_dict)
    
    for idx, result in all_results:
        print(f"  Rank {result['rank']}: Path {path_names[idx]}")
        print(f"    Total Cost: {result['total_cost']:.3f} | Quality: {result['quality_score']:.1f}/100")
        print(f"    Steps: {result['n_steps']}", end='')
        if result['trap_report']:
            print(f" | *** TRAP DETECTED: {result['trap_report'][0]['competitor']} ***")
        else:
            print(f" | No traps")
        print(f"    Steps:")
        for step in result['step_costs']:
            print(f"      Step {step['step']+1}: {step['reaction']}")
            print(f"              dG={step['dG']:.3f} eV | "
                  f"C_th={step['cost_thermo']:.2f} | "
                  f"C_kn={step['cost_kinetic']:.2f} | "
                  f"C_ion={step['cost_ion']:.2f}")
        print()
    
    print("=== KD Comparison (key for ion-correlation insight) ===")
    for phase, KD in sorted(KD_dict.items(), key=lambda x: x[1] if np.isfinite(x[1]) else 1e30, reverse=True):
        marker = " <-- BEST KD" if phase == 'Ba2TiO4' else (" <-- TARGET" if phase == 'BaTiO3' else "")
        print(f"  {phase:<12}: KD = {KD:.2e} cm^2/s{marker}")


if __name__ == '__main__':
    illustrate_path_cost()