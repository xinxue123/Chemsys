from typing import List, Optional
from .config import DEFAULT_CEAL2GE2_PHASES, PHASE_FORMULAS
from .report import PhaseState

class PhaseManager:
    def __init__(self, cif_manager):
        self.cif_manager = cif_manager
        self.states = {}
        self.target_phase = None

    def initialize_phase_set(self, target_phase: str, chemistry_system: Optional[list] = None, phase_hints: Optional[list] = None, user_cifs: Optional[dict] = None):
        self.target_phase = target_phase
        user_cifs = user_cifs or {}
        if target_phase == 'CeAl2Ge2':
            cfg = DEFAULT_CEAL2GE2_PHASES
        else:
            cfg = {'active': [target_phase], 'candidate': phase_hints or []}
        for name in cfg['active'] + cfg['candidate']:
            formula = PHASE_FORMULAS.get(name)
            rec = self.cif_manager.resolve_cif(name, formula=formula, preferred_path=user_cifs.get(name))
            status = 'active' if name in cfg['active'] and rec.valid else 'candidate'
            reason = '' if rec.valid else rec.notes
            self.states[name] = PhaseState(phase_name=name, formula=formula, cif_record=rec, status=status, reason=reason, source='default')
        return self.active_records()

    def active_records(self):
        return [s.cif_record for s in self.states.values() if s.status == 'active' and s.cif_record and s.cif_record.valid]

    def candidate_names(self):
        return [s.phase_name for s in self.states.values() if s.status == 'candidate']

    def rejected_names(self):
        return [s.phase_name for s in self.states.values() if s.status in {'rejected', 'disabled'}]

    def active_names(self):
        return [s.phase_name for s in self.states.values() if s.status == 'active']

    def phase_log(self):
        return {
            name: {
                'status': state.status,
                'reason': state.reason,
                'trial_count': state.trial_count,
                'last_fraction': state.last_fraction,
                'source': state.source,
                'cif_path': getattr(state.cif_record, 'file_path', None),
            }
            for name, state in self.states.items()
        }

    def activate_next_candidate(self, reason: str = ''):
        for state in self.states.values():
            if state.status == 'candidate' and state.cif_record and state.cif_record.valid:
                state.status = 'active'
                state.reason = reason or 'Activated due to unresolved residuals'
                return state.phase_name
        return None

    def disable_phase(self, phase_name: str, reason: str):
        state = self.states.get(phase_name)
        if not state or phase_name == self.target_phase:
            return False
        state.status = 'disabled'
        state.reason = reason
        return True

    def update_from_result(self, phase_fractions: dict, low_fraction_threshold: float = 0.005):
        for name, state in self.states.items():
            frac = float(phase_fractions.get(name, 0.0))
            state.last_fraction = frac
            if state.status == 'active' and name != self.target_phase and frac < low_fraction_threshold and state.trial_count >= 1:
                state.status = 'disabled'
                state.reason = 'Low fraction in repeated refinements'
            elif state.status == 'active':
                state.trial_count += 1
