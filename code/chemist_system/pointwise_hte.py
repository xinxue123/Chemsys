"""Supported corrected sequential HTE experiment entrypoint."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.pointwise_hte_study import main
    main()
