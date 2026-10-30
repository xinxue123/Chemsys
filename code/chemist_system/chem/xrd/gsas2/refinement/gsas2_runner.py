import os
import re
from pathlib import Path

from .... import paths
try:
    from .config import DEFAULT_LIMITS
except (ImportError, ValueError):
    # ★ ★ Fallback: 允许直接执行
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from config import DEFAULT_LIMITS  # type: ignore
try:
    from .report import StepResult, RefinementHistory
except (ImportError, ValueError):
    import sys, os
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from report import StepResult, RefinementHistory  # type: ignore


class GSAS2Runner:
    def __init__(self, output_dir: str, gsas2_path: str = str(paths.GSAS2_HOME)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gsas2_path = gsas2_path
        self.inst_file = self.output_dir / "default.instprm"
        self.project_warnings = []
        self._ensure_inst_file()
        self._gsas2 = None

    def _ensure_inst_file(self):
        if self.inst_file.exists():
            return
        lines = [
            "#GSAS-II instrument parameter file; default CuKa lab powder data for Bruker D8 ADVANCE; do not edit manually",
            "Type:PXC;Lam:1.5406;Zero:0.0;Polariz.:0.5;U:0.08074;V:-0.00113;W:0.00004;X:0.0;Y:0.0;Z:0.0;SH/L:0.002;Azimuth:0.0",
            "Diff-type:Bragg-Brentano;Gonio.radius:217.5;InstrName:Bruker D8 ADVANCE CuKa default",
        ]
        self.inst_file.write_text("\n".join(lines) + "\n")

    def _import_gsas2(self):
        if self._gsas2 is not None:
            return self._gsas2
        import sys
        if self.gsas2_path not in sys.path:
            sys.path.insert(0, self.gsas2_path)
        from GSASII import GSASIIscriptable as G2
        self._gsas2 = G2
        return G2

    def build_project(self, raw_xrd_path, active_cif_records, limits=DEFAULT_LIMITS,
                      le_bail_phases=None):
        """Build GSAS-II project with optional Le Bail phase support.
        
        le_bail_phases: set of phase names to run in Le Bail (Pawley) mode.
        """
        G2 = self._import_gsas2()
        gpx_path = str(self.output_dir / "project_initial.gpx")
        proj = G2.G2Project(newgpx=gpx_path)
        hist = proj.add_powder_histogram(raw_xrd_path, str(self.inst_file))
        hist.set_refinements({"Limits": [float(limits[0]), float(limits[1])]})
        try:
            hist.data["Sample Parameters"]["Scale"][1] = False
        except Exception:
            pass
        phases = {}
        le_bail_phases = le_bail_phases or set()
        for rec in active_cif_records:
            try:
                ph = proj.add_phase(rec.file_path, phasename=rec.phase_name,
                                    histograms=[hist])
            except Exception as exc:
                self.project_warnings.append(
                    f"phase_add_failed:{rec.phase_name}:{type(exc).__name__}:{exc}"
                )
                continue
            phases[rec.phase_name] = ph
            # Enable Le Bail (Pawley) mode for PDF-only phases
            if rec.phase_name in le_bail_phases:
                try:
                    ph.data["Histograms"][hist.name]["LeBail"] = True
                except Exception:
                    self.project_warnings.append(
                        f"le_bail_enable_failed:{rec.phase_name}"
                    )
            try:
                proj.link_histogram_phase(histogram=hist, phase=ph)
            except Exception as exc:
                self.project_warnings.append(
                    f"phase_link_failed:{rec.phase_name}:{type(exc).__name__}:{exc}"
                )
        if not phases:
            raise RuntimeError("No active phases could be added to GSAS-II project")
        proj.save()
        return proj, hist, phases, gpx_path

    def _apply_step(self, hist, phases, payload):
        hist_refs = payload.get("hist") or {}
        if hist_refs:
            hist.set_refinements(hist_refs)
        phase_hap_all = payload.get("phase_hap_all") or {}
        if phase_hap_all:
            for ph in phases.values():
                ph.set_HAP_refinements(phase_hap_all)
        phase_by_name_hap = payload.get("phase_by_name_hap") or {}
        for name, refs in phase_by_name_hap.items():
            if name in phases and refs:
                phases[name].set_HAP_refinements(refs)
        phase_all = payload.get("phase_all") or {}
        if phase_all:
            for ph in phases.values():
                ph.set_refinements(phase_all)
        phase_by_name = payload.get("phase_by_name") or {}
        for name, refs in phase_by_name.items():
            if name in phases and refs:
                phases[name].set_refinements(refs)

    def run_recipe(self, project, hist, phases, recipe):
        history = RefinementHistory()
        for step_name, payload in recipe:
            warnings = []
            ok = True
            try:
                self._apply_step(hist, phases, payload)
                ctrls = project["Controls"]["data"]
                ctrls["max cyc"] = 50
                project.do_refinements([{}])
                project.save()
            except Exception as e:
                ok = False
                warnings.append(str(e))
            gpx_path = str(project.filename)
            lst_path = gpx_path.replace(".gpx", ".lst")
            result = self.extract_results(lst_path)
            result.step_name = step_name
            result.success = ok
            result.warnings.extend(self.project_warnings)
            result.warnings.extend(warnings)
            result.lst_path = lst_path
            result.gpx_path = gpx_path
            history.steps.append(result)
            if not ok:
                break
        return history

    def extract_results(self, lst_path: str):
        if not os.path.exists(lst_path):
            return StepResult(step_name="unknown", success=False,
                              warnings=["LST file missing"])
        txt = Path(lst_path).read_text(errors="ignore")
        rwp = None
        chi2 = None
        gof = None
        r_pattern = None
        r_bkg = None
        wr_bkg = None
        wr_min = None
        durbin_watson = None
        m = re.search(r"Final refinement wR\\s*=\\s*([\\d.]+)%", txt)
        if not m:
            m = re.search(r"wR\\s*=\\s*([\\d.]+)%", txt)
        if m:
            rwp = float(m.group(1))
        m = re.search(r"chi\\*\\*2\\s*=\\s*([\\d.]+)", txt)
        if m:
            chi2 = float(m.group(1))
        m = re.search(r"GOF\\s*=\\s*([\\d.]+)", txt)
        if m:
            gof = float(m.group(1))
        m = re.search(
            r"Other residuals:\\s*R\\s*=\\s*([\\d.]+)%,\\s*R-bkg\\s*=\\s*([\\d.]+)%,\\s*wR-bkg\\s*=\\s*([\\d.]+)%\\s*wRmin\\s*=\\s*([\\d.]+)%",
            txt)
        if m:
            r_pattern = float(m.group(1))
            r_bkg = float(m.group(2))
            wr_bkg = float(m.group(3))
            wr_min = float(m.group(4))
        m = re.search(r"Durbin-Watson statistic\\s*=\\s*([\\d.]+)", txt)
        if m:
            durbin_watson = float(m.group(1))

        zero_shift = None
        m = re.search(
            r"Instrument Parameters:\\s*\\n\\s*names\\s*:\\s*Zero\\s*\\n\\s*value\\s*:\\s*([-+\\d.Ee]+)",
            txt)
        if m:
            try:
                zero_shift = float(m.group(1))
            except Exception:
                pass
        if zero_shift is None:
            m = re.search(
                r"name\\s*:\\s*Z\\s+Zero\\s*\\n\\s*value\\s*:\\s*[-+\\d.Ee]+\\s+([-+\\d.Ee]+)",
                txt)
            if m:
                try:
                    zero_shift = float(m.group(1))
                except Exception:
                    pass

        raw_weight = {}
        raw_weight_sigma = {}
        raw_phase = {}
        cell_params = {}
        size_strain = {}
        current = None
        for line in txt.splitlines():
            s = line.strip()
            m = re.match(r"(?:Result for phase|Phase name):\\s*([A-Za-z0-9_+\\-]+)", s)
            if m:
                current = m.group(1)
                continue
            m = re.match(r"Phase:\\s*([A-Za-z0-9_+\\-]+)\\s+in histogram:", s)
            if m:
                current = m.group(1)
                continue
            m = re.match(
                r"Phase fraction\\s*:\\s*([-+\\d.Ee]+)(?:,\\s*sig\\s*([-+\\d.Ee]+))?\\s*Weight fraction\\s*:\\s*([-+\\d.Ee]+)(?:,\\s*sig\\s*([-+\\d.Ee]+))?",
                s)
            if m and current:
                raw_phase[current] = float(m.group(1))
                raw_weight[current] = float(m.group(3))
                if m.group(4) is not None:
                    raw_weight_sigma[current] = float(m.group(4))
                continue
            m = re.match(r"Phase fraction\\s*:\\s*([-+\\d.Ee]+)", s)
            if m and current:
                raw_phase[current] = float(m.group(1))
                continue
            m = re.match(
                r"values:\\s*([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)\\s+([-+\\d.Ee]+)",
                s)
            if m and current and current not in cell_params:
                vals = [float(m.group(i)) for i in range(1, 8)]
                if vals[0] > 1 and vals[1] > 1 and vals[2] > 1:
                    cell_params[current] = {
                        "a": vals[0], "b": vals[1], "c": vals[2],
                        "alpha": vals[3], "beta": vals[4], "gamma": vals[5],
                        "volume": vals[6],
                    }
            m = re.match(
                r"Size model\\s*:\\s*([^:]+):\\s*([-+\\d.Ee]+)\\s*Refine\\?\\s*(\\w+)",
                s)
            if m and current:
                size_strain.setdefault(current, {})["size"] = {
                    "model": m.group(1).strip(), "value": float(m.group(2)),
                    "refine": m.group(3) == "True"
                }
            m = re.match(
                r"Mustrain model:\\s*([^:]+):\\s*([-+\\d.Ee]+)\\s*Refine\\?\\s*(\\w+)",
                s)
            if m and current:
                size_strain.setdefault(current, {})["mustrain"] = {
                    "model": m.group(1).strip(), "value": float(m.group(2)),
                    "refine": m.group(3) == "True"
                }

        raw = raw_weight if raw_weight else raw_phase
        total = sum(raw.values()) if raw else 0.0
        fracs = {k: (v / total if total else 0.0) for k, v in raw.items()}
        frac_sigmas = {k: (raw_weight_sigma.get(k, 0.0) / total if total else 0.0)
                       for k in raw}
        warnings = []
        if "Warning:" in txt:
            warnings = [line.strip() for line in txt.splitlines()
                        if line.strip().startswith("Warning:")]
        result = StepResult(
            step_name="unknown",
            success=True,
            rwp=rwp,
            chi2=chi2,
            gof=gof,
            r_pattern=r_pattern,
            r_bkg=r_bkg,
            wr_bkg=wr_bkg,
            wr_min=wr_min,
            durbin_watson=durbin_watson,
            phase_fractions=fracs,
            phase_fraction_sigmas=frac_sigmas,
        )
        result.cell_params = cell_params
        result.zero_shift = zero_shift
        result.size_strain = size_strain
        result.warnings.extend(warnings)
        return result
