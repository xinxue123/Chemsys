"""Universal powder-XRD pattern calculator from a CIF file (pure numpy)."""
import re, math
import numpy as np

# Cromer-Mann: a1,b1,a2,b2,a3,b3,a4,b4,c  (neutral atoms)
CROMER_MANN = {
 "Al":(6.4202,3.0387,1.9002,0.7426,1.5936,31.5472,1.9646,85.0886,1.1151),
 "Ge":(16.0816,2.8509,6.3747,0.2516,3.7068,11.4468,3.6830,54.7625,2.1313),
 "Ga":(15.2354,3.0669,6.7006,0.2412,4.3591,10.7805,2.9623,61.4135,1.7189),
 "Ce":(21.1671,2.81219,19.7695,0.226836,11.8513,17.6083,3.33049,127.113,1.86264),
 "O":(3.0485,13.2771,2.2868,5.7011,1.5463,0.3239,0.8670,32.9089,0.2508),
}

def _f0(elem, s):
    """Atomic scattering factor f(s), s=sin(theta)/lambda (1/A)."""
    cm = CROMER_MANN.get(elem)
    if cm is None:
        return 1.0  # neutral fallback
    a = cm[0:8:2]; b = cm[1:8:2]; c = cm[8]
    s2 = s * s
    return float(sum(ai*math.exp(-bi*s2) for ai,bi in zip(a,b)) + c)

def parse_cif(path):
    """Return dict(cell=(a,b,c,al,be,ga), atoms=[(elem,x,y,z,occ)], symops=[str])."""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    def g(tag):
        m = re.search(tag + r"\s+([-\d.]+)", txt)
        return float(m.group(1)) if m else None
    cell = (g("_cell_length_a"), g("_cell_length_b"), g("_cell_length_c"),
            g("_cell_angle_alpha"), g("_cell_angle_beta"), g("_cell_angle_gamma"))
    symops = re.findall(r"\d+\s+'([^']+)'", txt)
    return {"cell": cell, "symops": symops, "_txt": txt}

def parse_atoms(txt):
    """Parse the _atom_site_ loop generically using column tags."""
    lines = txt.splitlines(); i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_":
            tags = []; j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                tags.append(lines[j].strip()); j += 1
            if any("_atom_site_fract_x" in t for t in tags):
                col = {t: k for k, t in enumerate(tags)}
                return _read_atom_rows(lines, j, tags, col)
            i = j
        else:
            i += 1
    return []

def _read_atom_rows(lines, j, tags, col):
    ix = col["_atom_site_fract_x"]; iy = col["_atom_site_fract_y"]
    iz = col["_atom_site_fract_z"]
    it = col.get("_atom_site_type_symbol", col.get("_atom_site_label"))
    io = col.get("_atom_site_occupancy")
    atoms = []
    while j < len(lines):
        s = lines[j].strip()
        if not s or s.startswith(("_", "loop_", "#")):
            break
        p = s.split()
        if len(p) >= len(tags):
            el = re.sub(r"[^A-Za-z]", "", p[it])
            occ = float(p[io]) if io is not None else 1.0
            atoms.append((el, float(p[ix]), float(p[iy]), float(p[iz]), occ))
        j += 1
    return atoms

def _apply_symop(op, x, y, z):
    expr = op.lower().replace(" ", "").split(",")
    env = {"x": x, "y": y, "z": z}
    return tuple(eval(e, {"__builtins__": {}}, env) % 1.0 for e in expr)

def expand_atoms(atoms, symops):
    """Apply symmetry operators; dedupe atoms within tolerance."""
    full = []
    for el, x, y, z, occ in atoms:
        for op in symops:
            nx, ny, nz = _apply_symop(op, x, y, z)
            dup = any(a[0] == el and abs(a[1]-nx) < 1e-3 and abs(a[2]-ny) < 1e-3
                      and abs(a[3]-nz) < 1e-3 for a in full)
            if not dup:
                full.append((el, nx, ny, nz, occ))
    return full

def _recip_metric(cell):
    a, b, c, al, be, ga = cell
    al, be, ga = math.radians(al), math.radians(be), math.radians(ga)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    V = a*b*c*math.sqrt(max(1e-12, 1 - ca*ca - cb*cb - cg*cg + 2*ca*cb*cg))
    # reciprocal lattice parameters
    asr = b*c*math.sin(al)/V; bsr = a*c*math.sin(be)/V; csr = a*b*sg/V
    cas = (cb*cg - ca)/(math.sin(be)*sg)
    cbs = (ca*cg - cb)/(math.sin(al)*sg)
    cgs = (ca*cb - cg)/(math.sin(al)*math.sin(be))
    return asr, bsr, csr, cas, cbs, cgs, V

def d_spacing(h, k, l, rm):
    asr, bsr, csr, cas, cbs, cgs, V = rm
    inv_d2 = (h*h*asr*asr + k*k*bsr*bsr + l*l*csr*csr
              + 2*h*k*asr*bsr*cgs + 2*h*l*asr*csr*cbs + 2*k*l*bsr*csr*cas)
    return 1.0/math.sqrt(inv_d2) if inv_d2 > 0 else 0.0

def calc_pattern(cif_path, wavelength=1.540562, tt_min=10.0, tt_max=80.0,
                 hmax=6, i_cutoff=1.0):
    """Powder-XRD reference peaks from a CIF -> [(2theta, I_0_100, hkl)]."""
    cif = parse_cif(cif_path)
    atoms = expand_atoms(parse_atoms(cif["_txt"]), cif["symops"] or ["x,y,z"])
    rm = _recip_metric(cif["cell"]); refl = {}
    for h in range(-hmax, hmax+1):
      for k in range(-hmax, hmax+1):
        for l in range(-hmax, hmax+1):
          if h==0 and k==0 and l==0: continue
          tt, s = _refl_tt(h, k, l, rm, wavelength, tt_min, tt_max)
          if tt is None: continue
          key = round(tt, 2)
          refl[key] = refl.get(key, 0.0) + _refl_I(h, k, l, s, tt, atoms)
    if not refl: return []
    Imax = max(refl.values()) or 1.0
    return sorted((tt, 100.0*I/Imax, None) for tt, I in refl.items() if 100.0*I/Imax >= i_cutoff)

def _refl_tt(h, k, l, rm, wavelength, tt_min, tt_max):
    d = d_spacing(h, k, l, rm)
    if d <= 0: return None, None
    st = wavelength/(2*d)
    if st >= 1.0: return None, None
    tt = 2*math.degrees(math.asin(st))
    if tt < tt_min or tt > tt_max: return None, None
    return tt, st/wavelength

def _refl_I(h, k, l, s, tt, atoms):
    F = 0j
    for el, x, y, z, occ in atoms:
        ph = 2*math.pi*(h*x + k*y + l*z)
        F += occ*_f0(el, s)*complex(math.cos(ph), math.sin(ph))
    th = math.radians(tt/2)
    LP = (1 + math.cos(math.radians(tt))**2)/(math.sin(th)**2*math.cos(th))
    return (abs(F)**2)*LP

def pattern_from_struct(cell, atoms_asym, symops, wavelength=1.540562,
                        tt_min=10.0, tt_max=80.0, hmax=6, i_cutoff=1.0):
    """calc_pattern from in-memory cell/atoms/symops."""
    atoms = expand_atoms(atoms_asym, symops or ["x,y,z"])
    rm = _recip_metric(cell); refl = {}
    for h in range(-hmax, hmax+1):
      for k in range(-hmax, hmax+1):
        for l in range(-hmax, hmax+1):
          if h==0 and k==0 and l==0: continue
          tt, s = _refl_tt(h, k, l, rm, wavelength, tt_min, tt_max)
          if tt is None: continue
          key = round(tt, 2)
          refl[key] = refl.get(key, 0.0) + _refl_I(h, k, l, s, tt, atoms)
    if not refl: return []
    Imax = max(refl.values()) or 1.0
    return sorted((tt, 100.0*I/Imax, None) for tt, I in refl.items() if 100.0*I/Imax >= i_cutoff)

FCC = ["x,y,z","x,y+1/2,z+1/2","x+1/2,y,z+1/2","x+1/2,y+1/2,z"]
def element_pattern(symbol, a, struct="fcc", **kw):
    """Reference pattern for a pure cubic element (fcc or diamond)."""
    cell = (a, a, a, 90, 90, 90)
    if struct == "diamond":
        base = [(symbol,0,0,0,1.0),(symbol,0.25,0.25,0.25,1.0)]
    else:
        base = [(symbol,0,0,0,1.0)]
    return pattern_from_struct(cell, base, FCC, **kw)

ATOMIC_Z = {"Al":13,"Ge":32,"Ga":31,"Ce":58,"O":8}
def cell_volume(cell):
    a,b,c,al,be,ga = cell
    al,be,ga = map(math.radians,(al,be,ga))
    ca,cb,cg = math.cos(al),math.cos(be),math.cos(ga)
    return a*b*c*math.sqrt(max(1e-12,1-ca*ca-cb*cb-cg*cg+2*ca*cb*cg))

def scattering_power(cell, atoms_full):
    """Relative diffracting power ~ (sum Z)^2 / V^2  (Klug-Alexander proxy)."""
    zsum = sum(ATOMIC_Z.get(el,1)*occ for el,_,_,_,occ in atoms_full)
    V = cell_volume(cell)
    return (zsum*zsum)/(V*V) if V > 0 else 0.0

def cif_struct(cif_path):
    """Return (cell, full_atoms) for a CIF (symmetry-expanded)."""
    c = parse_cif(cif_path)
    atoms = expand_atoms(parse_atoms(c["_txt"]), c["symops"] or ["x,y,z"])
    return c["cell"], atoms
