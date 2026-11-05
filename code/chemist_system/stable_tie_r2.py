"""Focused controlled-validation R2 research, same total experiment budget."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.stable_tie_study import main
    main()
