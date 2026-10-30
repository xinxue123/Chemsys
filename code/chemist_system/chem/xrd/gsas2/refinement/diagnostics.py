from .config import RELIABILITY_THRESHOLDS


class RefinementDiagnostics:
    def _target_fraction(self, final, target_phase):
        return float((final.phase_fractions or {}).get(target_phase, 0.0))

    def _target_fraction_sigma(self, final, target_phase):
        return float((getattr(final, 'phase_fraction_sigmas', {}) or {}).get(target_phase, 0.0))

    def assess_history(self, history, target_phase: str):
        if not history.steps:
            return {
                'reliable': False,
                'level': 'failed',
                'reasons': ['No refinement steps executed'],
                'metrics': {},
                'actions': {'add_phase': False, 'disable_phases': [], 'reason': 'no_history'},
            }

        final = history.steps[-1]
        reasons = []
        target_fraction = self._target_fraction(final, target_phase)
        target_sigma = self._target_fraction_sigma(final, target_phase)
        rel_target_sigma = (target_sigma / target_fraction) if target_fraction > 1e-12 else None
        reliable = final.success and final.rwp is not None and bool(final.phase_fractions)
        publication_ready = False

        if not final.success:
            reasons.append('Final GSAS-II step failed')
        if final.rwp is None:
            reliable = False
            reasons.append('Missing Rwp from GSAS-II output')
        elif final.rwp > RELIABILITY_THRESHOLDS['max_rwp_for_minimum_reliable']:
            reliable = False
            reasons.append(f'Rwp too high: {final.rwp}')
        if final.gof is not None and final.gof > RELIABILITY_THRESHOLDS['max_gof_for_minimum_reliable']:
            reliable = False
            reasons.append(f'GOF too high: {final.gof}')
        if final.r_pattern is not None and final.r_pattern > RELIABILITY_THRESHOLDS['max_r_pattern_for_minimum_reliable']:
            reliable = False
            reasons.append(f'Pattern R too high: {final.r_pattern}')
        if final.r_bkg is not None and final.r_bkg > RELIABILITY_THRESHOLDS['max_r_bkg_for_minimum_reliable']:
            reliable = False
            reasons.append(f'Background residual too high: {final.r_bkg}')
        if target_phase not in (final.phase_fractions or {}):
            reliable = False
            reasons.append(f'Target phase {target_phase} missing from refined fractions')
        if final.zero_shift is not None and abs(final.zero_shift) > RELIABILITY_THRESHOLDS['max_zero_shift_abs_deg']:
            reliable = False
            reasons.append(f'Zero shift too large: {final.zero_shift}')
        if target_fraction <= 0.0:
            reliable = False
            reasons.append(f'Target phase {target_phase} refined to zero fraction')
        if rel_target_sigma is not None and rel_target_sigma > RELIABILITY_THRESHOLDS['max_relative_target_sigma']:
            reliable = False
            reasons.append(f'Target phase uncertainty too large: sigma/frac={rel_target_sigma:.3f}')

        add_phase = False
        add_phase_reason = ''
        if not reliable:
            # ★ ★ 增强 1a (★ 2026-06-11): Rwp=None 时也触发 add_phase
            # ★ 原因: 原来只有 wr_bkg 和 r_bkg 接近才触发, Rwp=None 时整个 if-else 不工作
            # ★ 实际 MnCrFe 测试中 Rwp=None 时 add_phase 完全失效, 导致 candidate 永远不激活
            if final.rwp is None:
                add_phase = True
                add_phase_reason = 'Rwp unavailable; activating candidate phases to test alternative compositions (★ chem-enhanced)'
            # ★ 增强 1b: target_fraction < 0.5% 触发
            # ★ 原因: 目标相精修到 ~0% 是 ★ 必有竞争相 ★ 的强诊断信号
            elif target_fraction < 0.005:
                add_phase = True
                add_phase_reason = f'Target phase {target_phase} refined to {target_fraction:.1e} (<0.5%); likely competing phase present (★ chem-enhanced)'
            # ★ 增强 1c: Rwp > 50% 触发
            # ★ 原因: Rwp 极高 (>50%) 必有主要相未被包含
            elif final.rwp > 50.0:
                add_phase = True
                add_phase_reason = f'Rwp={final.rwp:.1f}% extremely high; major missing phase (★ chem-enhanced)'
            # ★ 原有逻辑保留
            elif final.wr_bkg is not None and final.rwp is not None and abs(final.wr_bkg - final.rwp) < 0.25:
                add_phase = True
                add_phase_reason = 'Background-weighted residual tracks total residual; likely missing phase rather than profile-only issue'
            elif final.r_bkg is not None and final.r_pattern is not None and final.r_bkg > final.r_pattern + 8.0:
                add_phase = True
                add_phase_reason = 'Background residual dominates pattern residual; candidate impurity/background phase should be tested'

        disable_phases = []
        if not reliable:
            for name, frac in (final.phase_fractions or {}).items():
                if name == target_phase:
                    continue
                if frac < 0.01:
                    disable_phases.append(name)

        if reliable:
            if (
                final.rwp <= 15.0
                and (final.gof is None or final.gof <= RELIABILITY_THRESHOLDS['max_gof_for_publication_ready'])
                and (rel_target_sigma is None or rel_target_sigma <= 0.20)
            ):
                publication_ready = True
                level = 'publication_ready'
            else:
                level = 'minimum_reliable'
        else:
            level = 'interpretable'

        return {
            'reliable': reliable,
            'publication_ready': publication_ready,
            'level': level,
            'reasons': reasons,
            'metrics': {
                'rwp': final.rwp,
                'chi2': final.chi2,
                'gof': final.gof,
                'r_pattern': final.r_pattern,
                'r_bkg': final.r_bkg,
                'wr_bkg': final.wr_bkg,
                'wr_min': final.wr_min,
                'durbin_watson': final.durbin_watson,
                'zero_shift': final.zero_shift,
                'target_phase': target_phase,
                'target_fraction': target_fraction,
                'target_fraction_sigma': target_sigma,
                'target_fraction_rel_sigma': rel_target_sigma,
                'n_phases': len(final.phase_fractions or {}),
            },
            'actions': {
                'add_phase': add_phase,
                'add_phase_reason': add_phase_reason,
                'disable_phases': disable_phases,
            },
        }

    def should_refine_microstructure(self, history, target_phase: str):
        if not history.steps:
            return False, 'no_history'
        final = history.steps[-1]
        target_fraction = self._target_fraction(final, target_phase)
        if not final.success or final.rwp is None:
            return False, 'final_step_failed'
        if final.rwp <= 15.0:
            return False, 'already_publication_ready'
        if final.rwp > RELIABILITY_THRESHOLDS['max_rwp_for_minimum_reliable']:
            return False, 'rwp_too_high_for_stable_microstructure_refinement'
        if final.gof is not None and final.gof > RELIABILITY_THRESHOLDS['max_gof_for_minimum_reliable']:
            return False, 'gof_too_high_for_microstructure_refinement'
        if target_fraction < 0.5:
            return False, 'target_not_dominant'
        return True, 'refine_target_microstructure'



def peak_match_score(exp_xrd_path, active_phases, two_theta_range=(15, 90), top_n_peaks=10, tolerance_deg=0.4):
    """
    New function (Jun 2026): Match experimental XRD peaks against active phases.
    
    Args:
        exp_xrd_path: experimental XRD file path
        active_phases: list of active phases [{name, cif_path}, ...]
        two_theta_range: range of 2theta
        top_n_peaks: number of top experimental peaks to consider
        tolerance_deg: peak matching tolerance in degrees
    
    Returns:
        dict with unmatched_ratio, matched_peaks, unmatched_peaks, recommendation
    """
    import os
    import numpy as np
    from scipy.signal import find_peaks
    from pymatgen.io.cif import CifParser
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    
    try:
        data = np.loadtxt(exp_xrd_path, skiprows=1)
        two_theta = data[:, 0]
        intensity = data[:, 1]
        if two_theta[0] > two_theta[-1]:
            two_theta = two_theta[::-1]
            intensity = intensity[::-1]
    except Exception as e:
        return {'unmatched_ratio': 1.0, 'matched_peaks': [], 'unmatched_peaks': [], 'recommendation': f'Cannot load XRD: {e}'}
    
    mask = (two_theta >= two_theta_range[0]) & (two_theta <= two_theta_range[1])
    two_theta = two_theta[mask]
    intensity = intensity[mask]
    
    peaks, props = find_peaks(intensity, height=intensity.max()*0.10, distance=15)
    if len(peaks) == 0:
        return {'unmatched_ratio': 1.0, 'matched_peaks': [], 'unmatched_peaks': [], 'recommendation': 'No peaks found'}
    
    peak_data = sorted(
        [(two_theta[p], intensity[p]) for p in peaks],
        key=lambda x: -x[1]
    )[:top_n_peaks]
    
    exp_peaks_2theta = [t for t, _ in peak_data]
    exp_peaks_intensity = [i for _, i in peak_data]
    
    xrd_calc = XRDCalculator(wavelength='CuKa')
    theo_peaks_2theta = set()
    theo_peaks_by_phase = {}
    for phase in active_phases:
        cif_path = phase.get('cif_path') if isinstance(phase, dict) else None
        phase_name = phase.get('name') if isinstance(phase, dict) else str(phase)
        if not cif_path or not os.path.exists(cif_path):
            continue
        try:
            parser = CifParser(cif_path)
            struct = parser.parse_structures(primitive=False)[0]
            pattern = xrd_calc.get_pattern(struct, two_theta_range=two_theta_range)
            for t in pattern.x:
                theo_peaks_2theta.add(t)
            theo_peaks_by_phase[phase_name] = [(t, I) for t, I in zip(pattern.x, pattern.y) if I > 20]
        except Exception:
            continue
    
    if not theo_peaks_2theta:
        return {
            'unmatched_ratio': 1.0,
            'matched_peaks': [],
            'unmatched_peaks': list(zip(exp_peaks_2theta, exp_peaks_intensity)),
            'recommendation': 'No theoretical peaks computed; check active phase CIFs'
        }
    
    matched_peaks = []
    unmatched_peaks = []
    for t_exp, i_exp in zip(exp_peaks_2theta, exp_peaks_intensity):
        closest = min(theo_peaks_2theta, key=lambda x: abs(x - t_exp))
        if abs(closest - t_exp) < tolerance_deg:
            for phase_name, peaks_list in theo_peaks_by_phase.items():
                if any(abs(p[0] - t_exp) < tolerance_deg for p in peaks_list):
                    matched_peaks.append((t_exp, phase_name, closest))
                    break
        else:
            unmatched_peaks.append((t_exp, i_exp))
    
    unmatched_ratio = len(unmatched_peaks) / len(peak_data) if peak_data else 1.0
    
    if unmatched_ratio > 0.30:
        unmatch_str = ', '.join([f"{t:.2f}" for t, _ in unmatched_peaks[:5]])
        recommendation = (
            f'{unmatched_ratio*100:.0f}% peaks unmatched at 2theta = {unmatch_str}; '
            f'activate candidate phases (especially spinel/perovskite/pyrochlore).'
        )
    elif unmatched_ratio > 0.15:
        unmatch_str = ', '.join([f"{t:.2f}" for t, _ in unmatched_peaks[:3]])
        recommendation = (
            f'{unmatched_ratio*100:.0f}% peaks unmatched at 2theta = {unmatch_str}; '
            f'consider activating candidates.'
        )
    else:
        recommendation = f'{unmatched_ratio*100:.0f}% peaks unmatched; current phase set acceptable.'
    
    return {
        'unmatched_ratio': unmatched_ratio,
        'matched_peaks': matched_peaks,
        'unmatched_peaks': unmatched_peaks,
        'recommendation': recommendation,
    }


def chemistry_insight(target_phase, target_fraction, rwp, unmatched_peaks_two_theta=None, 
                     chemistry_system=None, max_suggestions=5):
    """
    New function (Jun 2026): Chemist diagnostic signals to user-friendly advice.
    
    Args:
        target_phase: target phase name
        target_fraction: refined fraction (or None)
        rwp: Rwp (or None)
        unmatched_peaks_two_theta: list of unmatched two_theta positions
        chemistry_system: list of elements
        max_suggestions: max recommendations to return
    
    Returns:
        dict with severity, summary, recommendations
    """
    insights = {
        'severity': 'info',
        'summary': 'Refinement completed.',
        'recommendations': [],
    }
    
    # Severity assessment
    if target_fraction is None or target_fraction < 0.005:
        insights['severity'] = 'critical'
        insights['summary'] = (
            f"Target phase {target_phase} refined to {target_fraction or 0:.1e} (<0.5%). "
            f"The expected phase is essentially absent."
        )
        insights['recommendations'].extend([
            "1. Target phase nearly absent (~0%); competing phase likely present",
            "2. Download same-system candidate phases from MP (spinel/perovskite/pyrochlore)",
            f"3. Chemistry system = {chemistry_system or 'N/A'}; recommend searching 3-element spinel AB2O4",
        ])
    elif target_fraction < 0.05:
        insights['severity'] = 'warning'
        insights['summary'] = (
            f"Target phase {target_phase} refined to {target_fraction:.1%} (<5%). "
            f"Major competing phase likely present."
        )
    elif target_fraction < 0.30:
        insights['severity'] = 'warning'
        insights['summary'] = (
            f"Target phase {target_phase} at {target_fraction:.1%} (low). "
            f"Process optimization may be needed."
        )
    
    if rwp is None:
        insights['severity'] = 'critical'
        insights['recommendations'].append(
            "4. Rwp unavailable; check GSAS-II binary module (pyspg/pypowder/pydiffax)"
        )
    elif rwp > 50:
        if insights['severity'] != 'critical':
            insights['severity'] = 'critical'
        insights['recommendations'].append(
            f"4. Rwp={rwp:.1f}% extremely high; major missing phase"
        )
    elif rwp > 20:
        insights['recommendations'].append(
            f"4. Rwp={rwp:.1f}% moderately high; consider profile parameters"
        )
    
    # Unmatched peaks
    if unmatched_peaks_two_theta and len(unmatched_peaks_two_theta) > 0:
        peaks_str = ', '.join([f"{t:.2f} deg" for t in unmatched_peaks_two_theta[:5]])
        insights['recommendations'].append(
            f"5. Unmatched peaks at two_theta = {peaks_str}; activate candidate phases"
        )
        if chemistry_system and len(chemistry_system) == 4:
            insights['recommendations'].append(
                f"6. 4-element system {chemistry_system}; recommend searching spinel AB2O4 structure"
            )
    
    insights['recommendations'] = insights['recommendations'][:max_suggestions]
    
    return insights
