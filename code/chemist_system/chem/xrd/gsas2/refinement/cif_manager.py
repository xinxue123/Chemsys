import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from .config import CIF_CACHE_DIR, LOCAL_CIF_HINTS, COD_ID_HINTS, PHASE_FORMULAS
from .... import config as _cfg
from .report import CifRecord


class CifManager:
    def __init__(self):
        self.cache_dir = CIF_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_cif(
        self,
        phase_name: str,
        formula: Optional[str] = None,
        preferred_path: Optional[str] = None,
        allow_download: bool = True,
    ) -> CifRecord:
        formula = formula or PHASE_FORMULAS.get(phase_name)
        for p, src in ((preferred_path, 'user'),):
            if p and Path(p).exists():
                rec = self.validate_cif(phase_name, formula, Path(p), source=src)
                if rec.valid:
                    return rec
        cached = self.cache_dir / f'{phase_name}.cif'
        if cached.exists():
            rec = self.validate_cif(phase_name, formula, cached, source='cache')
            if rec.valid:
                return rec
        for p, src in ((LOCAL_CIF_HINTS.get(phase_name), 'local_hint'),):
            if p and Path(p).exists():
                rec = self.validate_cif(phase_name, formula, Path(p), source=src)
                if rec.valid:
                    return rec
        if allow_download:
            rec = self.download_cif(phase_name, formula)
            if rec.valid:
                return rec
        return CifRecord(
            phase_name=phase_name,
            formula=formula,
            file_path=None,
            valid=False,
            source='missing',
            trust_level='missing',
            notes='No local/cache CIF found; automatic download failed. Provide MP_API_KEY/MPRESTER_API_KEY or an explicit local CIF.',
        )

    def validate_cif(self, phase_name: str, formula: Optional[str], path: Path, source: str = 'local') -> CifRecord:
        text = path.read_text(errors='ignore')
        valid = ('_cell_length_' in text and 'loop_' in text and '_atom_site' in text)
        space_group = None
        cell = {}
        for line in text.splitlines():
            s = line.strip()
            if s.startswith('_space_group_name_H-M_alt') or s.startswith('_symmetry_space_group_name_H-M'):
                parts = s.split(None, 1)
                space_group = parts[1].strip().strip("'") if len(parts) > 1 else None
            for key in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'):
                tag = f'_cell_length_{key}' if key in ('a', 'b', 'c') else f'_cell_angle_{key}'
                if s.startswith(tag):
                    parts = s.split()
                    if len(parts) >= 2:
                        try:
                            cell[key] = float(re.sub(r'\(.*\)', '', parts[1]))
                        except Exception:
                            pass
        rec = CifRecord(
            phase_name=phase_name,
            formula=formula,
            file_path=str(path),
            source=source,
            valid=valid,
            trust_level='local' if source not in {'missing', 'download_failed', 'download_skipped'} else 'missing',
            space_group=space_group,
            cell=cell or None,
        )
        if valid and source != 'cache':
            self.cache_record(rec, text)
        return rec

    def cache_record(self, rec: CifRecord, text: str):
        cache_path = self.cache_dir / f'{rec.phase_name}.cif'
        if not cache_path.exists():
            cache_path.write_text(text)
        meta = self.cache_dir / f'{rec.phase_name}.json'
        meta.write_text(json.dumps({
            'phase_name': rec.phase_name,
            'formula': rec.formula,
            'source': rec.source,
            'source_id': rec.source_id,
            'space_group': rec.space_group,
            'cell': rec.cell,
        }, ensure_ascii=False, indent=2))

    def _get_mp_api_key(self) -> Optional[str]:
        """Return MP API key from environment or local api_config; never print or store it."""
        key = os.environ.get('MP_API_KEY') or os.environ.get('MPRESTER_API_KEY')
        if key:
            return key
        config_path = None
        if config_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('solid_state_audit_api_config', config_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                return getattr(module, 'MP_API_KEY', None) or getattr(module, '_MP_KEY', None)
            except Exception:
                return None
        return None

    def _download_mp_cif(self, phase_name: str, formula: Optional[str]) -> CifRecord:
        """Download a CIF from Materials Project by exact formula.

        This is deliberately conservative: it queries MP by formula and selects the
        lowest energy_above_hull entry returned by MP. If MP/API/pymatgen is not
        available, it reports the limitation instead of fabricating a CIF.
        """
        api_key = self._get_mp_api_key()
        if not api_key:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_skipped',
                trust_level='missing',
                notes='MP_API_KEY/MPRESTER_API_KEY not available in environment.',
            )
        if not formula:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_skipped',
                trust_level='missing',
                notes='Formula is required for Materials Project CIF lookup.',
            )
        try:
            from pymatgen.core import Structure
            from pymatgen.io.cif import CifWriter
        except Exception as exc:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_failed',
                trust_level='missing',
                notes=f'pymatgen unavailable for MP CIF writing: {type(exc).__name__}: {exc}',
            )

        # Preferred path: official mp-api client, when installed in the current
        # interpreter. This avoids guessing MP REST details and records the MP id.
        def _mp_sort_key(doc):
            e = getattr(doc, 'energy_above_hull', None) if not isinstance(doc, dict) else doc.get('energy_above_hull')
            stable = getattr(doc, 'is_stable', False) if not isinstance(doc, dict) else doc.get('is_stable', False)
            theoretical = getattr(doc, 'theoretical', True) if not isinstance(doc, dict) else doc.get('theoretical', True)
            return (not bool(stable), e if e is not None else 1e9, bool(theoretical))

        try:
            from mp_api.client import MPRester
            with MPRester(api_key) as mpr:
                docs = mpr.materials.summary.search(
                    formula=formula,
                    fields=['material_id', 'formula_pretty', 'energy_above_hull', 'is_stable', 'theoretical', 'symmetry', 'structure'],
                )
            if docs:
                docs = sorted(docs, key=_mp_sort_key)
                doc = docs[0]
                mp_id = str(getattr(doc, 'material_id', 'unknown'))
                structure = getattr(doc, 'structure')
                cif_text = str(CifWriter(structure, symprec=0.01))
                path = self.cache_dir / f'{phase_name}.cif'
                path.write_text(cif_text)
                rec = self.validate_cif(phase_name, formula, path, source='MP')
                rec.source_id = mp_id
                rec.trust_level = 'database'
                if rec.valid:
                    self.cache_record(rec, cif_text)
                    meta = self.cache_dir / f'{phase_name}.json'
                    meta.write_text(json.dumps({
                        'phase_name': rec.phase_name,
                        'formula': rec.formula,
                        'source': rec.source,
                        'source_id': rec.source_id,
                        'space_group': rec.space_group,
                        'cell': rec.cell,
                        'energy_above_hull': getattr(doc, 'energy_above_hull', None),
                        'is_stable': getattr(doc, 'is_stable', None),
                        'theoretical': getattr(doc, 'theoretical', None),
                        'formula_pretty': getattr(doc, 'formula_pretty', None),
                        'symmetry': getattr(doc, 'symmetry', None),
                    }, ensure_ascii=False, indent=2, default=str))
                    return rec
                rec.notes = f'Downloaded Materials Project {mp_id}, but CIF validation failed.'
                return rec
        except ModuleNotFoundError:
            pass
        except Exception:
            # Fall through to direct REST attempt; do not expose API key or noisy internals.
            pass

        query = urllib.parse.urlencode({
            'formula': formula,
            'fields': 'material_id,formula_pretty,energy_above_hull,is_stable,theoretical,symmetry,structure',
        })
        url = f'https://api.materialsproject.org/materials/summary/?{query}'
        req = urllib.request.Request(url, headers={'X-API-KEY': api_key})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
        except Exception as exc:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_failed',
                trust_level='missing',
                notes=f'Materials Project query failed: {type(exc).__name__}: {exc}',
            )

        docs = payload.get('data') or []
        if not docs:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_missing',
                trust_level='missing',
                notes=f'No Materials Project structure found for formula={formula}.',
            )
        docs = sorted(docs, key=_mp_sort_key)
        doc = docs[0]
        mp_id = str(doc.get('material_id') or 'unknown')
        try:
            structure = Structure.from_dict(doc['structure'])
            cif_text = str(CifWriter(structure, symprec=0.01))
        except Exception as exc:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='MP_failed',
                source_id=mp_id,
                trust_level='missing',
                notes=f'Materials Project structure-to-CIF conversion failed: {type(exc).__name__}: {exc}',
            )

        path = self.cache_dir / f'{phase_name}.cif'
        path.write_text(cif_text)
        rec = self.validate_cif(phase_name, formula, path, source='MP')
        rec.source_id = mp_id
        rec.trust_level = 'database'
        if rec.valid:
            self.cache_record(rec, cif_text)
            meta = self.cache_dir / f'{phase_name}.json'
            meta.write_text(json.dumps({
                'phase_name': rec.phase_name,
                'formula': rec.formula,
                'source': rec.source,
                'source_id': rec.source_id,
                'space_group': rec.space_group,
                'cell': rec.cell,
                'energy_above_hull': doc.get('energy_above_hull'),
                'formula_pretty': doc.get('formula_pretty'),
                'symmetry': doc.get('symmetry'),
            }, ensure_ascii=False, indent=2, default=str))
        else:
            rec.notes = f'Downloaded Materials Project {mp_id}, but CIF validation failed.'
        return rec

    def download_cif(self, phase_name: str, formula: Optional[str] = None) -> CifRecord:
        """Best-effort CIF download: Materials Project first, then explicit COD ID.

        Formula-only lookup is allowed for MP because MP returns structured entries
        and the selected MP material_id is recorded. COD lookup remains restricted
        to explicit configured IDs; it never guesses by formula.
        """
        formula = formula or PHASE_FORMULAS.get(phase_name)

        mp_rec = self._download_mp_cif(phase_name, formula)
        if mp_rec.valid:
            return mp_rec

        cod_id = COD_ID_HINTS.get(phase_name)
        if not cod_id:
            mp_note = mp_rec.notes or 'MP lookup unavailable or failed.'
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='download_skipped',
                notes=f'{mp_note} No explicit COD ID hint configured; refusing to guess COD CIF by formula.',
            )
        url = f'https://www.crystallography.net/cod/{cod_id}.cif'
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode('utf-8', errors='ignore')
        except Exception as exc:
            return CifRecord(
                phase_name=phase_name,
                formula=formula,
                file_path=None,
                valid=False,
                source='download_failed',
                source_id=str(cod_id),
                notes=str(exc),
            )
        path = self.cache_dir / f'{phase_name}.cif'
        path.write_text(text)
        rec = self.validate_cif(phase_name, formula, path, source='COD')
        rec.source_id = str(cod_id)
        if not rec.valid:
            rec.notes = f'Downloaded COD {cod_id}, but CIF validation failed.'
        return rec
