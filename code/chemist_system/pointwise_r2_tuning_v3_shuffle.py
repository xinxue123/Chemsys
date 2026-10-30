"""Focused controlled-validation R2 research, same total experiment budget."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.r2_tuning_study_v3_shuffle import main
    main()
