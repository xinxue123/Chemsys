"""Offline checks for the compact GitHub edition; no model calls or GP fits."""
from pathlib import Path
import argparse,csv,json,sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('results','verify','integrity','catalogue'))
    args=parser.parse_args()
    from verify_github_results import verify_results,integrity
    result={}
    if args.command in ('results','verify'):result['results']=verify_results()
    if args.command in ('integrity','verify'):result['integrity']=integrity()
    if args.command=='catalogue':
        with (ROOT/'data_catalogue/experiment_register.csv').open(encoding='utf-8',newline='') as f:groups=list(csv.DictReader(f))
        result=dict(experiment_and_diagnostic_groups=len(groups),primary_comparison_records=487,
            archive_availability='Local-only; see LOCAL_DATA_REQUIREMENTS.md')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
