import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from .config import RUNS_DIR, MAX_PHASE_SET_ROUNDS
from .... import paths
from .cif_manager import CifManager
from .phase_manager import PhaseManager
from .gsas2_runner import GSAS2Runner
from .refinement_strategy import RefinementStrategy
from .diagnostics import RefinementDiagnostics
from .report import RefinementReport
from .pdf_manager import pdf_to_le_bail_cif, PDF_PHASES


class RietveldWorkflow:
    def __init__(self, gsas2_path: str = str(paths.GSAS2_HOME), conda_env: str = 'gsas2'):
        self.gsas2_path = gsas2_path
        self.conda_env = conda_env
        self.cif_manager = CifManager()
        self.strategy = RefinementStrategy()
        self.diagnostics = RefinementDiagnostics()

    def _step_dicts(self, history):
        return [s.__dict__ for s in (history.steps if history else [])]

    def _create_run_dir(self, raw_xrd_path, output_dir=None):
        run_dir = Path(output_dir) if output_dir else RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(raw_xrd_path).stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _report_from_dict(self, data):
        return RefinementReport(
            input=data.get('input', {}),
            active_phases=data.get('active_phases', []),
            candidate_phases=data.get('candidate_phases', []),
            rejected_phases=data.get('rejected_phases', []),
            final=data.get('final', {}),
            history=data.get('history', []),
            reliability=data.get('reliability', {}),
            phase_log=data.get('phase_log', {}),
            warnings=data.get('warnings', []),
            artifacts=data.get('artifacts', {}),
        )

    def _should_retry_via_conda(self, exc):
        msg = str(exc)
        markers = [
            'Could not read file:',
            'CIF',
            'pycifrw',
            'CifFile',
            'pyspg not found',
            'Unable to find GSAS-II binaries',
            'GSASII',
        ]
        return any(m.lower() in msg.lower() for m in markers)

    def _probe_local_gsas2_environment(self):
        probe_code = r'''
import importlib.util
import json
import os
import sys

issues = []
gsas2_path = sys.argv[1]

if not gsas2_path or not os.path.isdir(gsas2_path):
    issues.append(f'missing_gsas2_path:{gsas2_path}')

if importlib.util.find_spec('CifFile') is None:
    issues.append('missing_python_module:CifFile')

ready = False
if not issues:
    try:
        if gsas2_path not in sys.path:
            sys.path.insert(0, gsas2_path)
        from GSASII import GSASIIpath
        GSASIIpath.SetBinaryPath()
        if not getattr(GSASIIpath, 'binaryPath', None) and GSASIIpath.binaryPath is not None:
            raise RuntimeError('GSASII binaryPath unresolved')
        from GSASII import GSASIIspc  # triggers pyspg import
        from GSASII import GSASIIscriptable as _G2  # noqa: F401
        ready = True
    except Exception as exc:
        issues.append(f'import_failure:{type(exc).__name__}:{exc}')

print(json.dumps({'ready': ready, 'issues': issues}, ensure_ascii=False))
'''
        try:
            proc = subprocess.run(
                [sys.executable, '-c', probe_code, str(self.gsas2_path)],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                check=False,
            )
        except Exception as exc:
            return {'ready': False, 'issues': [f'probe_launch_failed:{type(exc).__name__}:{exc}']}

        stdout = (proc.stdout or '').strip()
        stderr = (proc.stderr or '').strip()
        if proc.returncode != 0:
            issue = f'probe_exit_nonzero:{proc.returncode}'
            if stderr:
                issue += f':{stderr.splitlines()[-1][:300]}'
            return {'ready': False, 'issues': [issue]}

        try:
            data = json.loads(stdout.splitlines()[-1]) if stdout else {'ready': False, 'issues': ['probe_empty_output']}
        except Exception as exc:
            issue = f'probe_parse_failed:{type(exc).__name__}:{exc}'
            if stdout:
                issue += f':stdout_tail={stdout.splitlines()[-1][:300]}'
            return {'ready': False, 'issues': [issue]}

        if stderr:
            data.setdefault('issues', []).append(f'probe_stderr_suppressed:{stderr.splitlines()[-1][:300]}')
        return data

    def _refine_pattern_local(self, raw_xrd_path, target_phase, chemistry_system=None, phase_hints=None, user_cifs=None, allow_download=True, max_rounds=MAX_PHASE_SET_ROUNDS, output_dir=None):
        run_dir = self._create_run_dir(raw_xrd_path, output_dir=output_dir)
        shutil.copy2(raw_xrd_path, run_dir / Path(raw_xrd_path).name)

        pm = PhaseManager(self.cif_manager)
        pm.initialize_phase_set(target_phase=target_phase, chemistry_system=chemistry_system, phase_hints=phase_hints, user_cifs=user_cifs)

        # ── Detect PDF-only phases that need Le Bail ────────────────────
        le_bail_phases = set()
        pdf_cif_map = {}  # phase_name -> temp CIF path
        for pname in list(pm.states.keys()):
            state = pm.states[pname]
            if state.cif_record and state.cif_record.valid:
                continue  # already has a valid CIF
            if pname in PDF_PHASES:
                pdf_info = PDF_PHASES[pname]
                cif_path = pdf_to_le_bail_cif(
                    phase_name=pdf_info["phase_name"],
                    formula=pdf_info["formula"],
                    cell_a=pdf_info["cell_a"],
                    cell_b=pdf_info["cell_b"],
                    cell_c=pdf_info["cell_c"],
                    cell_alpha=pdf_info["cell_alpha"],
                    cell_beta=pdf_info["cell_beta"],
                    cell_gamma=pdf_info["cell_gamma"],
                    space_group=pdf_info["space_group"],
                    it_number=pdf_info["it_number"],
                    equiv_positions=pdf_info["equiv_positions"],
                    output_dir=str(run_dir / "pdf_cifs"),
                )
                pdf_cif_map[pname] = cif_path
                le_bail_phases.add(pname)
                # Register the generated CIF as if user-provided
                user_cifs = dict(user_cifs or {})
                user_cifs[pname] = cif_path

        # Re-initialize with PDF CIFs included
        if pdf_cif_map:
            pm.initialize_phase_set(target_phase=target_phase,
                                    chemistry_system=chemistry_system,
                                    phase_hints=phase_hints,
                                    user_cifs=dict(user_cifs))

        history = None
        artifacts = {}
        final = {}
        warnings = []
        reliability = {'reliable': False, 'level': 'failed', 'reasons': ['workflow_not_started']}

        for round_idx in range(max_rounds):
            round_dir = run_dir / f'round_{round_idx+1}'
            round_dir.mkdir(exist_ok=True)
            runner = GSAS2Runner(output_dir=str(round_dir), gsas2_path=self.gsas2_path)
            active_records = pm.active_records()
            if not active_records:
                warnings.append('No active CIF records available')
                break

            proj, hist, phases, gpx_path = runner.build_project(
                raw_xrd_path, active_records, le_bail_phases=le_bail_phases)
            recipe = self.strategy.build_recipe(
                pm.active_names(), le_bail_phases=le_bail_phases)
            history = runner.run_recipe(proj, hist, phases, recipe)

            do_micro, reason = self.diagnostics.should_refine_microstructure(history, target_phase=target_phase)
            if do_micro:
                extra = runner.run_recipe(
                    proj,
                    hist,
                    phases,
                    self.strategy.build_microstructure_recipe(pm.active_names(), target_phase_name=target_phase),
                )
                history.steps.extend(extra.steps)
            else:
                warnings.append(f'microstructure_refinement_skipped:{reason}')

            diag = self.diagnostics.assess_history(history, target_phase=target_phase)
            final_step = history.steps[-1] if history.steps else None
            if final_step:
                for warn in getattr(final_step, 'warnings', []) or []:
                    if warn.startswith('phase_add_failed:'):
                        phase_name = warn.split(':', 2)[1]
                        if pm.disable_phase(phase_name, reason='GSAS-II could not read/add CIF during project build'):
                            warnings.append(f'phase_disabled:{phase_name}:gsas_add_failed')
                if final_step.phase_fractions:
                    pm.update_from_result(final_step.phase_fractions)
                final = {
                    'rwp': final_step.rwp,
                    'chi2': final_step.chi2,
                    'gof': final_step.gof,
                    'r_pattern': final_step.r_pattern,
                    'r_bkg': final_step.r_bkg,
                    'wr_bkg': final_step.wr_bkg,
                    'wr_min': final_step.wr_min,
                    'durbin_watson': final_step.durbin_watson,
                    'phase_fractions': final_step.phase_fractions,
                    'phase_fraction_sigmas': getattr(final_step, 'phase_fraction_sigmas', {}),
                    'cell_params': final_step.cell_params,
                    'size_strain': getattr(final_step, 'size_strain', {}),
                    'zero_shift': final_step.zero_shift,
                    'publication_ready': diag.get('publication_ready', False),
                }
                artifacts = {
                    'gpx_path': final_step.gpx_path,
                    'lst_path': final_step.lst_path,
                    'run_dir': str(run_dir),
                    'round_dir': str(round_dir),
                    'report_path': str(run_dir / 'report.json'),
                }
            reliability = diag
            if diag['reliable']:
                break
            if not final_step or not final_step.phase_fractions:
                break

            action_flags = diag.get('actions', {})
            changed_phase_set = False
            for phase_name in action_flags.get('disable_phases', []):
                if pm.disable_phase(phase_name, reason='Disabled after low refined fraction and unstable fit diagnostics'):
                    warnings.append(f'phase_disabled:{phase_name}')
                    changed_phase_set = True
            if action_flags.get('add_phase'):
                added = pm.activate_next_candidate(reason=action_flags.get('add_phase_reason', 'Activated due to poor fit diagnostics'))
                if added:
                    warnings.append(f'phase_added:{added}')
                    changed_phase_set = True
            if not changed_phase_set or not pm.active_records():
                break
        else:
            reliability = dict(reliability or {})
            reliability['reliable'] = False
            reliability['publication_ready'] = False
            reliability['level'] = reliability.get('level', 'interpretable')
            reasons = list(reliability.get('reasons', []))
            reasons.append('max_rounds exhausted')
            reliability['reasons'] = reasons

        report = RefinementReport(
            input={'raw_xrd': raw_xrd_path, 'target_phase': target_phase, 'chemistry_system': chemistry_system or []},
            active_phases=pm.active_names(),
            candidate_phases=pm.candidate_names(),
            rejected_phases=pm.rejected_names(),
            final=final,
            history=self._step_dicts(history),
            reliability=reliability,
            phase_log=pm.phase_log(),
            warnings=warnings,
            artifacts=artifacts,
        )
        (run_dir / 'report.json').write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return report

    def _refine_pattern_via_conda(self, raw_xrd_path, target_phase, chemistry_system=None, phase_hints=None, user_cifs=None, max_rounds=MAX_PHASE_SET_ROUNDS, output_dir=None, fallback_reason=None):
        run_dir = self._create_run_dir(raw_xrd_path, output_dir=output_dir)
        report_path = run_dir / 'report.json'

        cmd = [
            'conda', 'run', '--no-capture-output', '-n', self.conda_env,
            'python', '-m', 'chem.xrd.gsas2.refinement.cli',
            '--raw', str(raw_xrd_path),
            '--target', str(target_phase),
            '--gsas2-path', str(self.gsas2_path),
            '--max-rounds', str(max_rounds),
            '--output-dir', str(run_dir),
        ]
        if chemistry_system:
            cmd.extend(['--system', '-'.join(chemistry_system)])
        for ph in phase_hints or []:
            cmd.extend(['--phase', str(ph)])
        for name, path in (user_cifs or {}).items():
            cmd.extend(['--cif', f'{name}={path}'])

        env = os.environ.copy()
        plugin_dir = str(paths.HOME)
        env['PYTHONPATH'] = plugin_dir + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['GSAS2_REFINEMENT_SUBPROCESS'] = '1'

        subprocess.run(cmd, check=True, env=env)
        if not report_path.exists():
            raise RuntimeError(f'GSAS-II conda fallback finished without report.json: {report_path}')
        data = json.loads(report_path.read_text())
        warnings = data.get('warnings', [])
        warnings.append(f'executed_via_conda_env:{self.conda_env}')
        if fallback_reason:
            warnings.append(fallback_reason)
        data['warnings'] = warnings
        report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return self._report_from_dict(data)

    def refine_pattern(self, raw_xrd_path, target_phase, chemistry_system=None, phase_hints=None, user_cifs=None, allow_download=True, max_rounds=MAX_PHASE_SET_ROUNDS, output_dir=None):
        if os.environ.get('GSAS2_REFINEMENT_SUBPROCESS') == '1':
            return self._refine_pattern_local(
                raw_xrd_path,
                target_phase,
                chemistry_system=chemistry_system,
                phase_hints=phase_hints,
                user_cifs=user_cifs,
                allow_download=allow_download,
                max_rounds=max_rounds,
                output_dir=output_dir,
            )

        probe = self._probe_local_gsas2_environment()
        if not probe.get('ready'):
            reason = 'preflight_local_gsas2_unavailable:' + '; '.join(probe.get('issues') or ['unknown'])
            return self._refine_pattern_via_conda(
                raw_xrd_path,
                target_phase,
                chemistry_system=chemistry_system,
                phase_hints=phase_hints,
                user_cifs=user_cifs,
                max_rounds=max_rounds,
                output_dir=output_dir,
                fallback_reason=reason,
            )

        try:
            return self._refine_pattern_local(
                raw_xrd_path,
                target_phase,
                chemistry_system=chemistry_system,
                phase_hints=phase_hints,
                user_cifs=user_cifs,
                allow_download=allow_download,
                max_rounds=max_rounds,
                output_dir=output_dir,
            )
        except Exception as exc:
            if not self._should_retry_via_conda(exc):
                raise
            return self._refine_pattern_via_conda(
                raw_xrd_path,
                target_phase,
                chemistry_system=chemistry_system,
                phase_hints=phase_hints,
                user_cifs=user_cifs,
                max_rounds=max_rounds,
                output_dir=output_dir,
                fallback_reason=f'local_runtime_retry_to_conda:{type(exc).__name__}:{exc}',
            )


class GSAS2WorkflowEstimator:
    def __init__(self, system: str, calibration: dict = None, gsas2_path: str = '/home/ucas/GSASIIv2', conda_env: str = 'gsas2', **kwargs):
        self.system = system
        self.calibration = calibration or {}
        self.workflow = RietveldWorkflow(gsas2_path=gsas2_path, conda_env=conda_env)

    def estimate(self, xrd_data, **kwargs):
        from chem.xrd import YieldResult

        target_phase = 'CeAl2Ge2' if self.system == 'ceal2ge2' else self.system
        chemistry_system = ['Ce', 'Al', 'Ge'] if self.system == 'ceal2ge2' else None
        report = self.workflow.refine_pattern(
            xrd_data,
            target_phase=target_phase,
            chemistry_system=chemistry_system,
            phase_hints=kwargs.get('phase_hints'),
            user_cifs=kwargs.get('user_cifs'),
        )
        fracs = report.final.get('phase_fractions', {})
        frac_sigmas = report.final.get('phase_fraction_sigmas', {})
        target_wt = float(fracs.get(target_phase, 0.0))
        raw_yield = target_wt * 100.0
        slope = self.calibration.get('slope', 1.0)
        intercept = self.calibration.get('intercept', 0.0)
        sigma = self.calibration.get('cal_sigma', 5.0)
        yield_pct = max(0.0, min(100.0, slope * raw_yield + intercept))
        conf = 'high' if report.reliability.get('reliable') else 'medium'
        note = '; '.join(report.reliability.get('reasons', [])) or report.reliability.get('level', '')
        meta = {
            'backend': 'gsas2_workflow',
            'report_path': report.artifacts.get('report_path'),
            'run_dir': report.artifacts.get('run_dir'),
            'gpx_path': report.artifacts.get('gpx_path'),
            'lst_path': report.artifacts.get('lst_path'),
            'reliability_level': report.reliability.get('level'),
            'publication_ready': report.reliability.get('publication_ready', False),
            'metrics': report.reliability.get('metrics', {}),
            'phase_log': report.phase_log,
            'target_phase': target_phase,
            'target_cell': (report.final.get('cell_params') or {}).get(target_phase),
            'size_strain': (report.final.get('size_strain') or {}).get(target_phase),
            'zero_shift': report.final.get('zero_shift'),
            'warnings': report.warnings,
        }
        return YieldResult(
            yield_pct=yield_pct,
            sigma_pct=sigma,
            method='GSAS2Rietveld',
            R=target_wt,
            raw_yield=raw_yield,
            phase_fractions=fracs,
            phase_fractions_sigma=frac_sigmas,
            phase_peaks_used=meta,
            n_peaks_matched=len(fracs),
            confidence=conf,
            confidence_note=note,
        )
