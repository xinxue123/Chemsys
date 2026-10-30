from pathlib import Path

from .... import paths

BASE_DIR = paths.RIETVELD_ENGINE
CIF_CACHE_DIR = paths.RIETVELD_CIFS
RUNS_DIR = paths.RIETVELD_RUNS
DEFAULT_LIMITS = (10.0, 89.0)
DEFAULT_RADIATION = 'CuKa'
MAX_PHASE_SET_ROUNDS = 3
MAX_REFINEMENT_STEPS = 7
RELIABILITY_THRESHOLDS = {
    'min_nonzero_phases': 1,
    'max_rwp_for_minimum_reliable': 35.0,
    'max_target_wt_jump': 0.15,
    'max_zero_shift_abs_deg': 1.0,
    'max_cell_rel_change': 0.05,
    'max_gof_for_minimum_reliable': 5.0,
    'max_gof_for_publication_ready': 3.0,
    'max_r_pattern_for_minimum_reliable': 25.0,
    'max_r_bkg_for_minimum_reliable': 40.0,
    'max_relative_target_sigma': 0.35,
}
DEFAULT_CEAL2GE2_PHASES = {
    'active': ['CeAl2Ge2', 'Ce2Al3Ge4', 'CeAlGe'],
    'candidate': ['Al', 'CeO2', 'Al2O3', 'Ge', 'Ce', 'Ga'],
}
PHASE_FORMULAS = {
    'CeAl2Ge2': 'CeAl2Ge2',
    'Ce2Al3Ge4': 'Ce2Al3Ge4',
    'CeAlGe': 'CeAlGe',
    'Al': 'Al',
    'CeO2': 'CeO2',
    'Al2O3': 'Al2O3',
    'Ge': 'Ge',
    'Ce': 'Ce',
    'Ga': 'Ga',
}
LOCAL_CIF_HINTS = {
    'CeAl2Ge2': str(paths.ARC_CIFS / 'CeAl2Ge2.cif'),
    'Ce2Al3Ge4': str(paths.ARC_CIFS / 'Ce2Al3Ge4.cif'),
    'CeAlGe': str(paths.ARC_CIFS / 'CeAlGe.cif'),
    'Al2O3': str(paths.ARC_CIFS / 'Al2O3.cif'),
}
# Optional known database IDs. Empty means do not guess.
COD_ID_HINTS = {
}
paths.ensure(BASE_DIR, CIF_CACHE_DIR, RUNS_DIR)
