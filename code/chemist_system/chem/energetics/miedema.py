"""Miedema semi-empirical model for intermetallic formation enthalpy.

Computes the chemical-interaction enthalpy of intermetallics from tabulated
single-element parameters (phi*, n_ws^{1/3}, V^{2/3}); no DFT or Materials
Project key needed, so the auditor energetics are self-contained. Absolute
magnitudes are approximate (R term omitted); used to RANK phase competition,
for which the chemical term is robust. A DFT/MP source plugs in behind the API.
Params: de Boer et al., Cohesion in Metals (1988)."""
# element: (phi* [V], n_ws^{1/3} [d.u.], V^{2/3} [cm^2])
PARAM = {
    "Al": (4.20, 1.39, 4.64),
    "Ce": (3.18, 1.19, 7.69),
    "Ge": (4.55, 1.37, 4.60),
    "Ga": (4.10, 1.31, 5.20),
    "Si": (4.70, 1.50, 4.20),
}
P = 12.3
QP = 9.4

def binary_enthalpy(a, b, xa):
    """Chemical formation enthalpy (kJ/mol-atom) of A-B at atom fraction xa."""
    pa, na, va = PARAM[a]; pb, nb, vb = PARAM[b]
    xb = 1.0 - xa
    csa = xa * va / (xa * va + xb * vb); csb = 1.0 - csa
    dphi = pa - pb; dn = na - nb
    gamma = -P * dphi ** 2 + P * QP * dn ** 2
    ninv = 1.0 / na + 1.0 / nb
    order = 1.0 + 8.0 * (csa * csb) ** 2
    return order * csa * csb * (xa * va + xb * vb) * 2.0 * gamma / ninv

def formation_enthalpy(formula):
    """Compound formation enthalpy (kJ/mol-atom): stoichiometry-weighted sum of
    binary chemical terms over all element pairs."""
    from . import periodic as _P
    comp = _P.parse_formula(formula); els = list(comp); tot = sum(comp.values())
    x = {e: comp[e] / tot for e in els}
    H = 0.0
    for i in range(len(els)):
        for j in range(i + 1, len(els)):
            a, b = els[i], els[j]; pair = x[a] + x[b]
            if pair <= 0:
                continue
            H += pair * binary_enthalpy(a, b, x[a] / pair)
    return H
