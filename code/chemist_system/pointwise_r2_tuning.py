"""Versioned R2 tuning entrypoint; all previous controls are read-only."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.r2_tuning_study import main
    main()
