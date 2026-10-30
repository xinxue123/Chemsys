#!/usr/bin/env python3
"""
xrd_analyzer.py — XRD phase identification tool for materials synthesis loop.

Usage:
    python xrd_analyzer.py db                          # list all phases in DB
    python xrd_analyzer.py db-add <json>               # add phase from JSON
    python xrd_analyzer.py seed                        # seed common ceramic phases
    python xrd_analyzer.py scan <dat_file> [threshold] # identify phases in .dat file
    python xrd_analyzer.py query <formula>             # show reference peaks for a formula

Schema:
    phases(id, formula, common_name, source, pdf_id, symmetry,
           a, b, c, alpha, beta, gamma, created_at)
    peaks(phase_id, two_theta, hkl, d, I)  -- one row per peak
"""

import sqlite3, json, sys, argparse, re
from pathlib import Path
from datetime import date
from scipy.signal import find_peaks as scipy_find_peaks

import os
from ..paths import XRD_DB as _DEFAULT_XRD_DB

DB_PATH = Path(os.environ.get("CHEMIST_XRD_DB", _DEFAULT_XRD_DB))

# ── DB bootstrap ────────────────────────────────────────────────────────────

def dict_from_row(row, cols):
    return dict(zip(cols, row))

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS phases (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            formula     TEXT    NOT NULL,
            common_name TEXT,
            source      TEXT    NOT NULL DEFAULT 'COD',
            pdf_id      TEXT,
            symmetry    TEXT,
            a REAL, b REAL, c REAL,
            alpha REAL, beta REAL, gamma REAL,
            created_at  TEXT    NOT NULL DEFAULT (date('now')),
            UNIQUE(formula, source, pdf_id)
        );
        CREATE TABLE IF NOT EXISTS peaks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            phase_id   INTEGER NOT NULL REFERENCES phases(id) ON DELETE CASCADE,
            two_theta  REAL    NOT NULL,
            hkl        TEXT,
            d          REAL,
            I          INTEGER,
            UNIQUE(phase_id, two_theta)
        );
        CREATE INDEX IF NOT EXISTS idx_peaks_phase ON peaks(phase_id);
        CREATE INDEX IF NOT EXISTS idx_peaks_2theta ON peaks(two_theta);
    """)
    conn.commit()
    return conn

def get_conn():
    return init_db()

# ── Phase CRUD ──────────────────────────────────────────────────────────────

def add_phase(conn: sqlite3.Connection,
              formula: str, common_name: str, source: str = "COD",
              pdf_id: str = None, symmetry: str = None,
              peaks: list = None) -> int:
    """
    Insert a phase, return its integer ID.
    peaks = [(two_theta, hkl, d, I), ...]
    """
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO phases
          (formula, common_name, source, pdf_id, symmetry)
        VALUES (?, ?, ?, ?, ?)
    """, (formula, common_name, source, pdf_id, symmetry))
    conn.commit()
    cur.execute("SELECT id FROM phases WHERE formula=? AND source=?",
                (formula, source))
    phase_id = cur.fetchone()[0]
    if peaks:
        for two_theta, hkl, d, I in peaks:
            cur.execute("""
                INSERT OR IGNORE INTO peaks
                  (phase_id, two_theta, hkl, d, I)
                VALUES (?, ?, ?, ?, ?)
            """, (phase_id, two_theta, hkl, d, I))
        conn.commit()
    return phase_id

def seed_common_phases():
    """Seed DB with common ceramic phases — MP/COD reference peaks (CuKa, λ=1.540562 Å).

    Reference data sources:
      ErTa7O19 → MP mp-773103, P-6c2, a=6.2579, c=20.0671 Å  (pymatgen XRDCalculator)
      ErTaO4   → MP mp-3444,   P2/c,  a=5.079,  c=5.4145 Å
      Er2O3    → COD COD-1000096, Ia-3
      Ta2O5    → MP mp-11787,  Cmcm,  a=6.8519, c=9.9419 Å
      ErTa3O9  → MP mp-772023, R3
      Er3TaO7  → MP mp-775306, R32
      BaTiO3   → COD COD-1000097, P4mm
    """
    conn = init_db()
    phases = [
        ("ErTa7O19","Erbium tantalate (7:1)","MP","mp-773103","P-6c2",
         [(16.36,'(1,0,-1)',5.4195,33),(17.68,'(0,0,0)',5.0168,24),
          (18.61,'(1,0,-1)',4.7684,11),(24.17,'(1,0,-1)',3.6815,4),
          (26.65,'(0,0,0)',3.3445,66),(28.88,'(2,-1,-1)',3.0916,100),
          (29.91,'(2,-1,-1)',2.9871,44),(31.43,'(1,0,-1)',2.8462,8),
          (31.57,'(2,-1,-1)',2.8342,6),(33.76,'(2,-1,-1)',2.6549,53),
          (34.28,'(2,0,-2)',2.6160,5),(35.80,'(0,0,0)',2.5084,3),
          (36.41,'(2,-1,-1)',2.4676,51),(37.73,'(2,0,-2)',2.3842,6),
          (39.44,'(2,-1,-1)',2.2849,1),(39.59,'(1,0,-1)',2.2764,2),
          (42.78,'(2,-1,-1)',2.1137,13),(50.52,'(3,0,-3)',1.8065,41),
          (54.31,'(2,-1,-1)',1.6892,24),(58.03,'(3,0,-3)',1.5895,40)]),
        ("ErTaO4","Erbium tantalate (1:1)","MP","mp-3444","P2/c",
         [(24.12,'(1,1,0)',3.689,18),(27.15,'(0,0,2)',3.283,100),
          (29.72,'(-1,1,1)',3.003,44),(34.51,'(-1,1,2)',2.596,29),
          (35.93,'(0,2,0)',2.496,55),(40.57,'(-2,0,2)',2.222,33),
          (43.51,'(1,0,3)',2.078,36),(47.22,'(0,2,2)',1.924,43),
          (49.46,'(-2,1,3)',1.842,41),(51.12,'(2,1,2)',1.785,33)]),
        ("Er2O3","Erbium(III) oxide","COD","COD-1000096","Ia-3",
         [(20.64,'(2,1,1)',4.299,40),(29.32,'(2,2,2)',3.046,100),
          (33.74,'(4,0,0)',2.655,60),(36.11,'(4,1,1)',2.486,45),
          (48.54,'(4,3,1)',1.875,35),(58.63,'(4,4,0)',1.574,30),
          (63.01,'(6,1,1)',1.474,25)]),
        ("Ta2O5","Tantalum(V) oxide","COD","COD-1000889","Pmmn",
         [(22.92,'(0,0,1)',3.880,55),(28.32,'(1,1,0)',3.151,100),
          (36.73,'(1,1,1)',2.447,65),(45.06,'(2,0,0)',2.013,40),
          (50.52,'(2,0,1)',1.806,35),(55.62,'(0,0,2)',1.652,25),
          (62.30,'(2,0,2)',1.489,20)]),
        ("ErTa3O9","Erbium tantalate (1:3)","MP","mp-772023","R3",
         [(12.10,'(1,0,1)',7.31,25),(17.45,'(1,1,0)',5.08,45),
          (24.61,'(2,0,2)',3.62,100),(29.33,'(1,1,3)',3.04,60),
          (31.78,'(3,0,1)',2.81,40),(35.24,'(2,2,2)',2.54,35),
          (38.82,'(4,0,0)',2.32,50),(43.15,'(3,1,2)',2.10,45)]),
        ("Er3TaO7","Erbium tantalate (3:1)","MP","mp-775306","R32",
         [(14.85,'(1,1,0)',5.96,30),(24.41,'(1,0,2)',3.65,100),
          (29.72,'(1,1,3)',3.00,70),(33.85,'(2,2,0)',2.65,55),
          (36.52,'(3,0,1)',2.46,40),(41.23,'(2,2,2)',2.19,35),
          (44.87,'(3,1,2)',2.02,45),(50.12,'(4,0,0)',1.82,50)]),
        ("BaTiO3","Barium titanate","COD","COD-1000097","P4mm",
         [(22.12,'100',4.019,45),(31.46,'101',2.842,100),(38.93,'110',2.313,75),
          (44.96,'111',2.013,30),(50.82,'200',1.796,55),(56.12,'201',1.638,25),
          (65.80,'211',1.418,40)]),
        ("BaCO3","Barium carbonate","COD","COD-1000043","Pmcn",
         [(23.52,'111',3.783,60),(24.41,'021',3.647,65),(27.06,'200',3.289,100),
          (32.82,'210',2.729,55),(42.10,'002',2.144,45),(47.52,'112',1.914,35)]),
        ("TiO2","Titanium dioxide (rutile)","COD","COD-1000063","Pbcn",
         [(27.44,'110',3.252,100),(36.08,'101',2.486,75),(41.23,'111',2.189,40),
          (54.31,'211',1.688,65),(56.64,'220',1.625,30),(64.08,'002',1.452,20)]),
        ("ZrNCl","Zirconium nitride chloride","COD","COD-1008834","R-3m",
         [(12.82,'001',6.910,100),(20.34,'002',4.371,70),(25.51,'100',3.490,65),
          (28.90,'101',3.087,55),(35.78,'102',2.506,40),(44.20,'110',2.047,30)]),
        ("ZrCl4","Zirconium tetrachloride","COD","COD-1001143","monoclinic",
         [(15.24,'111',5.824,100),(22.16,'200',4.019,75),(26.86,'211',3.324,65),
          (30.52,'220',2.929,50),(35.18,'310',2.555,35)]),
        ("ZrN","Zirconium nitride","COD","COD-1000062","rocksalt",
         [(33.64,'111',2.665,100),(38.78,'200',2.319,80),(56.52,'220',1.627,55),
          (68.05,'311',1.378,35)]),
        ("Ba2TiO4","Dibarium titanate","COD","COD-1000391","P21/c",
         [(19.14,'110',4.643,60),(28.64,'111',3.120,100),(30.06,'002',2.976,85),
          (41.82,'202',2.159,45),(47.08,'211',1.932,35)]),
        ("BaO","Barium oxide","COD","COD-1008729","Fm-3m",
         [(24.40,'111',3.645,100),(28.33,'200',3.152,80),(40.12,'220',2.246,55),
          (49.50,'311',1.839,30),(57.80,'222',1.594,25)]),
        ("CeAl2Ge2","Cerium dialuminium digermanide","LIT","PCD-1614986","P-3m1",
         [(21.42,'001',4.153,8),(27.65,'101',3.226,100),(32.19,'110',2.786,55),
          (33.33,'111',2.688,70),(34.10,'102',2.630,45),(35.44,'200',2.531,38),
          (38.54,'201',2.338,62),(39.26,'112',2.296,48),(42.16,'210',2.143,35),
          (43.88,'202',2.063,28),(48.03,'103',1.894,15)]),
        ("Ce2Al3Ge4","Dicerium trialuminium tetragermanide","LIT","est","Cmce",
         [(26.82,'020',3.323,100),(28.15,'200',3.171,70),(30.44,'002',2.937,55),
          (34.92,'022',2.570,45),(39.78,'220',2.266,35),(44.21,'202',2.048,25)]),
        ("CeAlGe","Cerium aluminium germanide","LIT","est","I41md",
         [(25.41,'101',3.505,100),(28.63,'110',3.120,65),(33.18,'002',2.701,50),
          (38.45,'202',2.342,40),(42.67,'211',2.118,30),(48.92,'301',1.862,25)]),
        ("Al","Aluminium metal","COD","COD-1521459","Fm-3m",
         [(38.47,'111',2.338,100),(44.74,'200',2.025,50),(65.13,'220',1.432,35)]),
        ("Ge","Germanium","COD","COD-1011194","Fd-3m",
         [(27.28,'111',3.269,100),(45.31,'220',2.000,55),(53.68,'311',1.706,40)]),
        ("Ce","Cerium metal","COD","COD-1521012","Fm-3m",
         [(28.62,'111',3.121,100),(33.42,'200',2.682,55),(47.65,'220',1.908,35)]),
    ]
    for args in phases:
        add_phase(conn, *args)
    print(f"Seeded {len(phases)} phases")
    conn.close()

def list_phases(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("""
        SELECT p.formula, p.common_name, p.source, p.pdf_id, p.symmetry,
               COUNT(pk.id) as n_peaks
        FROM phases p
        LEFT JOIN peaks pk ON p.id = pk.phase_id
        GROUP BY p.id
        ORDER BY p.formula
    """)
    print(f"{'Formula':<12} {'Name':<32} {'Source':<6} {'PDF ID':<16} {'Spg':<12} {'Peaks'}")
    print("-" * 88)
    for row in cur.fetchall():
        formula, name, source, pdf_id, symmetry, n = row
        print(f"{formula:<12} {name:<32} {source:<6} {str(pdf_id):<16} {str(symmetry):<12} {n}")
    conn.close()

def query_phase(conn: sqlite3.Connection, formula: str):
    cur = conn.cursor()
    cur.execute("SELECT id FROM phases WHERE formula=?", (formula,))
    row = cur.fetchone()
    if not row:
        print(f"Phase '{formula}' not found in DB.")
        conn.close()
        return
    phase_id = row[0]
    cur.execute("SELECT common_name, source, pdf_id, symmetry FROM phases WHERE id=?", (phase_id,))
    meta = cur.fetchone()
    print(f"\nFormula   : {formula}")
    print(f"Name      : {meta[0]}")
    print(f"Source    : {meta[1]}  PDF: {meta[2]}  Symmetry: {meta[3]}")
    print(f"\n{'':10} {'2θ (°)':<10} {'hkl':<8} {'d (Å)':<10} {'I (%)':<8}")
    print("-" * 46)
    cur.execute("SELECT two_theta, hkl, d, I FROM peaks WHERE phase_id=? ORDER BY two_theta", (phase_id,))
    for two_theta, hkl, d, I in cur.fetchall():
        print(f"{'':10} {two_theta:<10.2f} {hkl:<8} {d:<10.4f} {I:<8}")
    conn.close()

# ── XRD scan ────────────────────────────────────────────────────────────────

def parse_dat(path: str) -> tuple[list, list]:
    """
    Parse a .dat file, return (two_theta_list, intensity_list).
    Skips comment lines (#) and blank lines.
    Handles:
      - 2-column: 2theta  I
      - 3-column: 2theta  I  ? (some formats have a third col, skip it)
    """
    two_thetas, intensities = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = re.split(r'[\t ,]+', line)
        if len(parts) >= 2:
            try:
                tt = float(parts[0])
                i  = float(parts[1])
                if 5 <= tt <= 90:
                    two_thetas.append(tt)
                    intensities.append(i)
            except ValueError:
                continue
    return two_thetas, intensities

def _simple_peak_detect(two_thetas: list, intensities: list,
               height_frac: float = 0.05, min_dist: float = 0.5) -> list:
    """
    Simple absolute-difference peak detection.
    Returns list of (two_theta, intensity) for peaks above height_frac * max(I).
    """
    if not intensities:
        return []
    max_i = max(intensities)
    threshold = height_frac * max_i
    peaks = []
    intensities = list(intensities)
    for i in range(1, len(intensities) - 1):
        if (intensities[i] > intensities[i-1] and
            intensities[i] > intensities[i+1] and
            intensities[i] >= threshold):
            peaks.append((two_thetas[i], intensities[i]))
    # merge nearby peaks
    if not peaks:
        return peaks
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p[0] - merged[-1][0] < min_dist:
            if p[1] > merged[-1][1]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged

# Alias kept for callers that import find_peaks from this module
find_peaks = _simple_peak_detect

def match_peak(obs_tw: float, ref_peaks: list, tolerance: float = 0.35) -> tuple:
    """Find closest reference peak within tolerance. Returns (two_theta, hkl, delta)."""
    best = None
    for p in ref_peaks:
        delta = abs(p[0] - obs_tw)
        if delta < tolerance and (best is None or delta < best[3]):
            best = (p[0], p[1], p[2], delta)
    if best:
        return best[0], best[1], best[3]  # two_theta, hkl, delta
    return None, None, None

def identify_phase(conn: sqlite3.Connection,
                   obs_peaks: list,
                   required_match_frac: float = 0.60,
                   tolerance: float = 0.35) -> dict:
    """
    Match observed peaks against all phases in DB.
    Returns dict with per-phase: {phase, n_peaks, n_matched, frac, confidence, matched_peaks}
    """
    cur = conn.cursor()
    cur.execute("SELECT id, formula, common_name, source FROM phases")
    phases = cur.fetchall()

    # Sort observed peaks by intensity descending → strongest peaks matched first
    obs_peaks = sorted(obs_peaks, key=lambda x: x[1], reverse=True)

    results = []
    for phase_id, formula, common_name, source in phases:
        cur.execute("SELECT two_theta, hkl, d, I FROM peaks WHERE phase_id=?", (phase_id,))
        ref_peaks = [(row[0], row[1], row[2], row[3]) for row in cur.fetchall()]
        if not ref_peaks:
            continue

        n_obs = len(obs_peaks)
        n_ref = len(ref_peaks)
        # Many-to-many matching with SYSTEMATIC SHIFT CORRECTION.
        # Step 1: find all (obs, ref) pairs within tolerance.
        # Step 2: compute average shift from those pairs.
        # Step 3: re-match with obs_shifted = obs_tw - avg_shift.
        # This handles zero-point errors and sample displacement in XRD data.
        all_candidates = []  # (obs_tw, obs_I, ref_idx, ref_tw, ref_hkl, delta, weight)
        for obs_tw, obs_I in obs_peaks:
            for idx, rp in enumerate(ref_peaks):
                delta = abs(rp[0] - obs_tw)
                if delta < tolerance:
                    weight = min(obs_I / 1000.0, 1.0)
                    all_candidates.append((obs_tw, obs_I, idx, rp[0], rp[1], delta, weight))

        if not all_candidates:
            continue

        # Compute systematic shift: weighted average of (ref_tw - obs_tw)
        # weight = obs_I/1000 so strong peaks dominate
        total_w = sum(c[6] for c in all_candidates)
        avg_shift = sum((c[3] - c[0]) * c[6] for c in all_candidates) / total_w if total_w > 0 else 0.0

        # For each ref peak, find best obs match after applying shift correction.
        # Use obs_shifted = obs_tw + avg_shift, compare to ref_tw.
        best_for_ref = {}
        for obs_tw, obs_I in obs_peaks:
            obs_shifted = obs_tw + avg_shift
            for idx, rp in enumerate(ref_peaks):
                delta = abs(rp[0] - obs_shifted)
                if delta < tolerance:
                    weight = min(obs_I / 1000.0, 1.0)
                    if idx not in best_for_ref or delta < best_for_ref[idx][6]:
                        best_for_ref[idx] = (obs_tw, obs_shifted, obs_I, idx, rp[0], rp[1], delta, weight)

        matches = list(best_for_ref.values())
        n_m, n_r = len(matches), len(ref_peaks)
        total_weight = sum(m[7] for m in matches)
        weighted_confidence = total_weight / n_r if n_r > 0 else 0.0
        confidence = weighted_confidence

        results.append({
            "phase":        formula,
            "common_name":  common_name,
            "source":       source,
            "n_obs_peaks":  n_obs,
            "n_ref_peaks":  n_ref,
            "n_matched":    len(matches),
            "match_frac":   len(matches) / n_obs if n_obs > 0 else 0,
            "confidence":   confidence,
            "matched_peaks": [{"obs": m[0], "ref_tw": m[4], "hkl": m[5], "delta": m[6], "weight": m[7]} for m in matches],
        })

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results

def scan_dat(dat_path: str, height_frac: float = 0.05, tolerance: float = 0.35,
             required_match_frac: float = 0.60):
    conn = init_db()
    two_thetas, intensities = parse_dat(dat_path)
    # Use prominence-based peak detection with reasonable defaults
    max_i = max(intensities)
    peaks_out, props = scipy_find_peaks(intensities, height=max_i * 0.05, distance=3, prominence=80)
    obs_peaks = [(two_thetas[i], intensities[i]) for i in peaks_out]
    print(f"[scan] File        : {dat_path}")
    print(f"[scan] Data points: {len(two_thetas)}")
    print(f"[scan] Peaks found: {len(obs_peaks)}")
    if not obs_peaks:
        print("[scan] No peaks detected. Try lowering --height-frac.")
        return
    print(f"[scan] Top peaks   : {[(f'{tw:.2f}', int(i)) for tw, i in obs_peaks[:8]]}")
    print()
    results = identify_phase(conn, obs_peaks, tolerance=tolerance)
    if not results:
        print("No matching phases found.")
        return
    print(f"{'Formula':<12} {'Name':<28} {'n_m/n_o':<10} {'n_m/n_r':<10} {'Confidence':<10}")
    print("-" * 75)
    for r in results[:8]:
        print(f"{r['phase']:<12} {r['common_name']:<28} "
              f"{r['n_matched']}/{r['n_obs_peaks']:<8} {r['n_matched']}/{r['n_ref_peaks']:<8} "
              f"{r['confidence']:.2f}")
    conn.close()
    return results

# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="xrd_analyzer.py",
                                  description="XRD phase identification tool")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("db", help="List all phases in database")

    db_add = sub.add_parser("db-add", help="Add phase from JSON file")
    db_add.add_argument("json", help="JSON file with phase data")

    sub.add_parser("seed", help="Seed common ceramic phases")

    scan = sub.add_parser("scan", help="Scan a .dat XRD file")
    scan.add_argument("dat_file", help="Path to .dat file")
    scan.add_argument("threshold", nargs="?", type=float, default=0.60,
                      help="Min match fraction (default: 0.60)")

    query = sub.add_parser("query", help="Show reference peaks for a formula")
    query.add_argument("formula", help="Chemical formula, e.g. BaTiO3")

    args = ap.parse_args()

    if args.cmd == "db":
        list_phases(get_conn())

    elif args.cmd == "db-add":
        with open(args.json) as f:
            data = json.load(f)
        conn = get_conn()
        for item in data if isinstance(data, list) else [data]:
            pid = add_phase(conn,
                            item["formula"],
                            item.get("common_name", item["formula"]),
                            item.get("source", "COD"),
                            item.get("pdf_id"),
                            item.get("symmetry"),
                            item.get("peaks"))
            print(f"  Added: {item['formula']} (ID={pid})")
        conn.close()

    elif args.cmd == "seed":
        seed_common_phases()

    elif args.cmd == "scan":
        scan_dat(args.dat_file, tolerance=0.35, required_match_frac=args.threshold)

    elif args.cmd == "query":
        query_phase(get_conn(), args.formula)

    else:
        ap.print_help()

if __name__ == "__main__":
    main()
