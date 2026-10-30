"""Run the isolated R2 same-region trust fix and compare to frozen controls."""
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)

if __name__=='__main__':
    from chem.benchmarks.r2_region_trust_study import main
    main()
