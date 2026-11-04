"""Isolated program-KF experiment entry point."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from program_kf_weight.study import main
if __name__=='__main__':main()
