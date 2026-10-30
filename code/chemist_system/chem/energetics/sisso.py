"""SISSO-descriptor finite-T Gibbs correction (Bartel et al., Nat. Commun.
2018), pure python. Estimates the vibrational free energy G^delta(T) [eV/atom]
from per-atom volume and mean atomic mass, so the auditor can finite-T-correct
the 0 K DFT hull without phonon calculations."""
import math
from .periodic import ATOMIC_MASS, parse_formula
ELEM_VATOM = {"Al": 16.48, "Ce": 34.79, "Ge": 22.66, "Ga": 19.6, "Si": 20.0}

def g_delta(v_atom, m_atom, T):
    """Bartel 2018 G^delta(T) eV/atom; v_atom Ang^3/atom, m_atom amu."""
    if T is None or T <= 0 or v_atom <= 0:
        return 0.0
    return ((-2.48e-4*math.log(v_atom) - 8.94e-5*m_atom/v_atom)*T
            + 0.181*math.log(T) - 0.882)

def mean_mass(formula):
    c = parse_formula(formula); n = sum(c.values()) or 1
    return sum(ATOMIC_MASS[e]*k for e, k in c.items()) / n

def dGf_correction(formula, v_atom, T):
    """Finite-T correction to the formation energy (eV/atom):
    G^delta(compound,T) - sum_i x_i G^delta(element_i,T)."""
    if T is None or T <= 0:
        return 0.0
    c = parse_formula(formula); n = sum(c.values()) or 1
    gc = g_delta(v_atom, mean_mass(formula), T)
    ge = 0.0
    for e, k in c.items():
        ve = ELEM_VATOM.get(e, v_atom)
        ge += (k / n) * g_delta(ve, ATOMIC_MASS[e], T)
    return gc - ge
