"""Exploratory held-out follow-up: scopes from pure BO's random first points."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'), str(ROOT/'code/experiment_tools')]
from hypothesis_scope_v2.random_start import main
if __name__ == '__main__':
    main()
