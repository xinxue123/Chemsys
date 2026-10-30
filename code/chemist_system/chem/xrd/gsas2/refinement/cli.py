#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from .workflow import RietveldWorkflow
from .... import paths


def parse_user_cifs(items):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise SystemExit(f'--cif expects Phase=/path/file.cif, got: {item}')
        name, path = item.split('=', 1)
        out[name.strip()] = path.strip()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='GSAS-II automated Rietveld workflow phase-1')
    parser.add_argument('--raw', required=True, help='Raw XRD pattern path')
    parser.add_argument('--target', required=True, help='Target phase name, e.g. CeAl2Ge2')
    parser.add_argument('--system', default='', help='Chemistry system, e.g. Ce-Al-Ge')
    parser.add_argument('--phase', action='append', default=[], help='Candidate/active phase hint; repeatable')
    parser.add_argument('--cif', action='append', default=[], help='Explicit CIF mapping Phase=/path/to/file.cif; repeatable')
    parser.add_argument('--output-dir', default=None, help='Run output directory')
    parser.add_argument('--max-rounds', type=int, default=3)
    parser.add_argument('--gsas2-path', default=str(paths.GSAS2_HOME))
    parser.add_argument('--smart', action='store_true', help='Use robust multi-start model-selection pipeline')
    parser.add_argument('--instprm', default=None, help='Calibrated GSAS-II instrument parameter file (required with --smart)')
    parser.add_argument('--low', type=float, default=10.0)
    parser.add_argument('--high', type=float, default=80.0)
    parser.add_argument('--systematic-floor', type=float, default=0.09,
                        help='Absolute phase-fraction systematic floor (default 0.09 from A0 calibration)')
    parser.add_argument('--min-bic-improvement', type=float, default=10.0)
    parser.add_argument('--soft-element-balance', default=None,
                        help='Optional bulk cation prior, e.g. Mg=1,Ni=1,Ti=2; original refinement is unchanged')
    parser.add_argument('--balance-sigma', type=float, default=0.03,
                        help='One-sigma tolerance on normalized cation fractions (default 0.03)')
    args = parser.parse_args(argv)

    user_cifs = parse_user_cifs(args.cif)
    if args.smart:
        from .smart_pipeline import PhaseSpec, SmartGSAS2Pipeline
        if not args.instprm:
            parser.error('--instprm is required with --smart')
        if not args.output_dir:
            parser.error('--output-dir is required with --smart')
        if args.target not in user_cifs:
            parser.error(f'--smart requires --cif {args.target}=/path/to/parent.cif')
        missing = [name for name in args.phase if name not in user_cifs]
        if missing:
            parser.error('--smart candidate phases need explicit --cif mappings: ' + ','.join(missing))
        pipeline = SmartGSAS2Pipeline(
            xrd=args.raw,
            parent=PhaseSpec(args.target, user_cifs[args.target]),
            candidates=[PhaseSpec(name, user_cifs[name]) for name in args.phase],
            instprm=args.instprm,
            output_dir=args.output_dir,
            gsas2_path=args.gsas2_path,
            limits=(args.low, args.high),
            systematic_floor=args.systematic_floor,
            min_bic_improvement=args.min_bic_improvement,
        )
        report = pipeline.run()
        if args.soft_element_balance:
            from .soft_balance import apply_soft_element_balance
            balance_path = Path(args.output_dir) / 'soft_element_balance_report.json'
            report['soft_element_balance'] = apply_soft_element_balance(
                report,
                args.soft_element_balance,
                balance_sigma=args.balance_sigma,
                systematic_floor=args.systematic_floor,
                cif_base_dir=Path.cwd(),
                output_path=balance_path,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    chemistry = [x for x in args.system.replace(',', '-').split('-') if x] if args.system else None
    workflow = RietveldWorkflow(gsas2_path=args.gsas2_path)
    report = workflow.refine_pattern(
        args.raw,
        target_phase=args.target,
        chemistry_system=chemistry,
        phase_hints=args.phase or None,
        user_cifs=user_cifs,
        max_rounds=args.max_rounds,
        output_dir=args.output_dir,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
