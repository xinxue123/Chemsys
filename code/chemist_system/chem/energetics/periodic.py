"""Minimal universal periodic-table helpers (no external deps)."""
import re

ATOMIC_MASS = {
    "H":1.008,"Li":6.94,"B":10.81,"C":12.011,"N":14.007,"O":15.999,
    "F":18.998,"Na":22.99,"Mg":24.305,"Al":26.982,"Si":28.085,"P":30.974,
    "S":32.06,"K":39.098,"Ca":40.078,"Ti":47.867,"V":50.942,"Cr":51.996,
    "Mn":54.938,"Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.38,
    "Ga":69.723,"Ge":72.63,"Y":88.906,"Zr":91.224,"Nb":92.906,"Mo":95.95,
    "Ba":137.327,"La":138.905,"Ce":140.116,"Ta":180.948,"W":183.84,
    "Er":167.259,
}

def parse_formula(formula):
    """Return {element: count} for a formula like 'CeAl2Ge2'."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula):
        if not el:
            continue
        out[el] = out.get(el, 0.0) + (float(n) if n else 1.0)
    return out

def molar_mass(formula):
    return sum(ATOMIC_MASS[e] * n for e, n in parse_formula(formula).items())

def stoich_atom_fractions(formula):
    """Atom fractions from a formula's stoichiometry."""
    c = parse_formula(formula); tot = sum(c.values()) or 1.0
    return {e: n / tot for e, n in c.items()}
