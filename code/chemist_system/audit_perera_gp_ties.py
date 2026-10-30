from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent)
if __name__=='__main__':
    from chem.benchmarks.perera_gp_tie_audit import main
    main()
