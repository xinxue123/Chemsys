"""Universal data-driven QPA over a directory of XRD .xy patterns.
Reference patterns computed from CIFs / elemental structures via
xrd_pattern (no hardcoded peak lists). Composition parsed from
filename charge ratios. Outputs canonical JSON dataset."""
import os, re, glob, json
import numpy as np
from . import pattern as xp

WAVELENGTH = 1.540562

def build_basis(cif_dir, elements):
    """Phase basis: name->{peaks,rir,elements}. CIF phases + pure elements."""
    basis = {}
    for path in sorted(glob.glob(os.path.join(cif_dir, "*.cif"))):
        name = os.path.splitext(os.path.basename(path))[0]
        cell, atoms = xp.cif_struct(path)
        pk = xp.calc_pattern(path, wavelength=WAVELENGTH, i_cutoff=4)
        basis[name] = {"peaks": [(t, i) for t, i, _ in pk],
                       "rir": xp.scattering_power(cell, atoms),
                       "elements": sorted(set(a[0] for a in atoms))}
    return basis

def add_elements(basis, elements):
    """Add pure-element phases (fcc/diamond) to the basis."""
    for el, (a, st) in elements.items():
        pk = xp.element_pattern(el, a, st, wavelength=WAVELENGTH, i_cutoff=4)
        cell = (a, a, a, 90, 90, 90)
        nat = 4 if st == "fcc" else 8
        atoms = [(el, 0, 0, 0, 1.0)] * nat
        basis[el] = {"peaks": [(t, i) for t, i, _ in pk],
                     "rir": xp.scattering_power(cell, atoms),
                     "elements": [el]}
    return basis

def parse_composition(fname, order=("Ce", "Al", "Ge")):
    """Extract charge ratio A-B-C from filename -> atom fractions dict."""
    m = re.search(r"(\d+)-(\d+)-(\d+)", os.path.basename(fname))
    if not m:
        return None
    ratio = [float(x) for x in m.groups()]
    M = {"Ce": 140.12, "Al": 26.98, "Ge": 72.63, "Ga": 69.72, "O": 16.0}
    # filename ratios are nominal charge masses (g); convert to atom fractions
    moles = [r / M[el] for r, el in zip(ratio, order)]
    tot = sum(moles) or 1.0
    return {el: mo / tot for el, mo in zip(order, moles)}, dict(zip(order, ratio))

def load_xy(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]

def snip_background(y, n_iter=40):
    """SNIP background estimation (LLS space)."""
    yy = np.log(np.log(np.sqrt(np.clip(y, 0, None) + 1) + 1) + 1)
    z = yy.copy()
    for p in range(n_iter, 0, -1):
        a = z[p:-p]
        avg = 0.5 * (z[:-2*p] + z[2*p:])
        z[p:-p] = np.minimum(a, avg)
    return (np.exp(np.exp(z) - 1) - 1) ** 2 - 1

def integrate_peak(x, y, center, half=0.45):
    """Integrated area of y over [center-half, center+half]."""
    m = (x >= center - half) & (x <= center + half)
    if not m.any():
        return 0.0
    return float(np.trapz(np.clip(y[m], 0, None), x[m]))

def select_markers(basis, tol=0.6, min_rel=15):
    """For each phase pick isolated strong reference peaks (not overlapping
    another phase's comparable peak). Returns name -> [(2theta, Irel)]."""
    markers = {}
    for name, info in basis.items():
        others = [(t, i) for o, oi in basis.items() if o != name
                  for t, i in oi["peaks"] if i >= min_rel]
        iso = []
        for t, i in sorted(info["peaks"], key=lambda r: -r[1]):
            if i < min_rel:
                continue
            clash = any(abs(t - ot) < tol and oi >= 0.5 * i for ot, oi in others)
            if not clash:
                iso.append((t, i))
        markers[name] = iso[:3] if iso else sorted(info["peaks"], key=lambda r: -r[1])[:1]
    return markers

def qpa(x, y, basis, markers):
    """RIR quantitative phase analysis -> {phase: wt_fraction}."""
    yb = snip_background(y)
    c = {}
    for name, info in basis.items():
        mk = markers[name]
        vals = []
        for t, irel in mk:
            area = integrate_peak(x, yb, t)
            vals.append(area / max(irel / 100.0, 1e-6))
        meas = float(np.median(vals)) if vals else 0.0
        c[name] = meas / max(info["rir"], 1e-30)
    tot = sum(c.values()) or 1.0
    return {k: v / tot for k, v in c.items()}

def run(data_dir, cif_dir, target, elements, order=("Ce","Al","Ge"), out_path=None):
    basis = build_basis(cif_dir, elements)
    add_elements(basis, elements)
    owned = assign_ownership(basis)
    samples = []
    sys_elems = set(order)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.xy"))):
        token = re.split(r"[_0-9]", os.path.basename(path))[0]
        felems = set(re.findall(r"[A-Z][a-z]?", token))
        if felems and not felems <= sys_elems:
            continue
        comp = parse_composition(path, order)
        if comp is None:
            continue
        frac, charge = comp
        x, y = load_xy(path)
        w = qpa_local(x, y, basis, owned)
        _, ysig = qpa_uncertainty(x, y, basis, owned, target)
        samples.append({"file": os.path.basename(path), "charge_ratio": charge,
            "atom_fraction": {k: round(v, 4) for k, v in frac.items()},
            "phase_wt": {k: round(100*v, 2) for k, v in w.items()},
            "yield": round(100*w.get(target, 0.0), 2),
            "yield_sigma": round(ysig, 2)})
    samples.sort(key=lambda s: -s["yield"])
    return basis, owned, samples

def run_and_save(data_dir, cif_dir, target, elements, order=("Ce","Al","Ge"), out_path=None):
    basis, owned, samples = run(data_dir, cif_dir, target, elements, order)
    result = {"target": target, "n_samples": len(samples),
              "basis_rir": {k: round(v["rir"]/basis[target]["rir"], 4) for k, v in basis.items()},
              "diagnostic_peaks": {k: owned[k] for k in basis},
              "samples": samples}
    if out_path:
        json.dump(result, open(out_path, "w"), indent=2)
    return result

def render_pattern(grid, peaks, fwhm=0.22):
    """Render reference peaks as summed Gaussians on a 2theta grid."""
    sig = fwhm / 2.3548
    out = np.zeros_like(grid)
    for t, irel in peaks:
        out += irel * np.exp(-0.5 * ((grid - t) / sig) ** 2)
    return out

def nnls_mu(A, b, iters=500):
    """Non-negative least squares via multiplicative updates."""
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.full(A.shape[1], 1.0)
    for _ in range(iters):
        x *= Atb / (AtA @ x + 1e-12)
    return x

def qpa_fullpattern(x, y, basis, tt_lo=25.0, tt_hi=75.0):
    """Full-pattern RIR QPA via NNLS over all reference peaks.
    Fit restricted to [tt_lo, tt_hi] where crystalline peaks dominate."""
    yb = np.clip(snip_background(y, n_iter=120), 0, None)
    m = (x >= tt_lo) & (x <= tt_hi)
    xf, bf = x[m], yb[m]
    names = list(basis.keys())
    cols = []
    for n in names:
        pk = [(t, i) for t, i in basis[n]["peaks"] if tt_lo <= t <= tt_hi]
        col = render_pattern(xf, pk)
        area = np.trapz(col, xf) or 1.0
        cols.append(col / area)
    A = np.vstack(cols).T
    scale = nnls_mu(A, bf)
    c = {n: scale[i] / max(basis[n]["rir"], 1e-30) for i, n in enumerate(names)}
    tot = sum(c.values()) or 1.0
    return {k: v / tot for k, v in c.items()}

def net_height(x, y, t, hw=0.22, gap=(0.5, 1.0)):
    """Peak height above a local linear baseline (median of flanks)."""
    win = (x >= t - hw) & (x <= t + hw)
    base = (((x >= t - gap[1]) & (x <= t - gap[0])) |
            ((x >= t + gap[0]) & (x <= t + gap[1])))
    if not win.any() or not base.any():
        return 0.0
    return max(0.0, float(y[win].max() - np.median(y[base])))

def assign_ownership(basis, tol=0.45, margin=1.35):
    """Each phase owns reference peaks where it dominates (no other phase has a
    comparable peak within tol). Universal overlap-disambiguation rule."""
    owned = {}
    for name, info in basis.items():
        keep = []
        for t, i in info["peaks"]:
            contest = [oi for o, od in basis.items() if o != name
                       for ot, oi in od["peaks"] if abs(ot - t) < tol]
            if not contest or i >= margin * max(contest):
                keep.append((t, i))
        owned[name] = keep
    return owned

def qpa_local(x, y, basis, owned, min_rel=12, hw=0.22):
    """RIR QPA from local-baseline net heights over owned diagnostic peaks."""
    c = {}
    for name, info in basis.items():
        pk = [(t, i) for t, i in owned[name] if i >= min_rel]
        if not pk:
            pk = sorted(owned[name] or info["peaks"], key=lambda r: -r[1])[:1]
        num = sum(net_height(x, y, t, hw=hw) * (i / 100.0) for t, i in pk)
        den = sum(i / 100.0 for _, i in pk) or 1.0
        S = num / den
        c[name] = S / max(info["rir"], 1e-30)
    tot = sum(c.values()) or 1.0
    return {k: v / tot for k, v in c.items()}


# Default Ce-Al-Ge elemental references (fcc Al, diamond Ge); extend as needed.
DEFAULT_ELEMENTS = {"Al": (4.0495, "fcc"), "Ge": (5.6575, "diamond")}

def qpa_uncertainty(x, y, basis, owned, target, reps=40, seed=0):
    """Estimate target-phase wt% uncertainty by perturbing the peak window and
    adding Poisson counting noise; returns (mean_pct, std_pct)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        hw = float(rng.uniform(0.18, 0.28))
        yn = y + rng.normal(0.0, np.sqrt(np.clip(y, 1, None)))
        w = qpa_local(x, yn, basis, owned, hw=hw)
        vals.append(100.0 * w.get(target, 0.0))
    v = np.asarray(vals)
    return float(v.mean()), float(v.std())


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "."
    cifs = sys.argv[2] if len(sys.argv) > 2 else data
    target = sys.argv[3] if len(sys.argv) > 3 else "CeAl2Ge2"
    out = sys.argv[4] if len(sys.argv) > 4 else "arcmelt_dataset.json"
    r = run_and_save(data, cifs, target, DEFAULT_ELEMENTS, out_path=out)
    print(f"{r['n_samples']} samples -> {out}")
    for s in r["samples"]:
        print(f"  {s['file']:26s} yield({target})={s['yield']:6.2f}%")

