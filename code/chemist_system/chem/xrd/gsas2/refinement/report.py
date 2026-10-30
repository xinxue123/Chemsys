from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

@dataclass
class CifRecord:
    phase_name: str
    formula: Optional[str]
    file_path: Optional[str]
    source: str = 'unknown'
    source_id: Optional[str] = None
    space_group: Optional[str] = None
    cell: Optional[dict] = None
    valid: bool = False
    trust_level: str = 'unverified'
    notes: str = ''

@dataclass
class PhaseState:
    phase_name: str
    formula: Optional[str]
    cif_record: Optional[CifRecord]
    status: str = 'candidate'
    reason: str = ''
    trial_count: int = 0
    last_fraction: float = 0.0
    source: str = 'default'

@dataclass
class StepResult:
    step_name: str
    success: bool
    rwp: Optional[float] = None
    chi2: Optional[float] = None
    gof: Optional[float] = None
    r_exp: Optional[float] = None
    r_pattern: Optional[float] = None
    r_bkg: Optional[float] = None
    wr_bkg: Optional[float] = None
    wr_min: Optional[float] = None
    durbin_watson: Optional[float] = None
    phase_fractions: Dict[str, float] = field(default_factory=dict)
    phase_fraction_sigmas: Dict[str, float] = field(default_factory=dict)
    cell_params: Dict[str, dict] = field(default_factory=dict)
    size_strain: Dict[str, dict] = field(default_factory=dict)
    zero_shift: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    lst_path: Optional[str] = None
    gpx_path: Optional[str] = None

@dataclass
class RefinementHistory:
    steps: List[StepResult] = field(default_factory=list)

@dataclass
class RefinementReport:
    input: Dict[str, Any]
    active_phases: List[str]
    candidate_phases: List[str]
    rejected_phases: List[str]
    final: Dict[str, Any]
    history: List[Dict[str, Any]]
    reliability: Dict[str, Any]
    phase_log: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
