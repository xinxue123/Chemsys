"""Adaptive counterexample R2 research; first experiment and baselines frozen."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.r2_tuning_study_v2 import main
    main()
