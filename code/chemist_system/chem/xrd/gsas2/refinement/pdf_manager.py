#!/usr/bin/env python3
"""PDF-to-LeBail bridge: create minimal GSAS-II compatible CIF from PDF card data."""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict


def pdf_to_le_bail_cif(
    phase_name, formula, cell_a, cell_b, cell_c,
    cell_alpha, cell_beta, cell_gamma, space_group, it_number,
    equiv_positions, output_dir=None, **kwargs
):
    """Generate a minimal CIF suitable for Le Bail (no real atoms needed).

    Puts a single placeholder atom with near-zero occupancy so GSAS-II
    can load the phase. The LeBail flag will be set at runtime to bypass
    structure-factor computation.
    """
    lines = [
        "data_" + phase_name,
        "_chemical_name_common '" + phase_name + " - PDF-only Le Bail'",
        "_chemical_formula_sum '" + formula + "'",
        "_cell_length_a %.4f" % cell_a,
        "_cell_length_b %.4f" % cell_b,
        "_cell_length_c %.4f" % cell_c,
        "_cell_angle_alpha %.2f" % cell_alpha,
        "_cell_angle_beta %.2f" % cell_beta,
        "_cell_angle_gamma %.2f" % cell_gamma,
        "_symmetry_space_group_name_H-M '" + space_group + "'",
        "_symmetry_Int_Tables_number %d" % it_number,
        "loop_",
        " _symmetry_equiv_pos_site_id",
        " _symmetry_equiv_pos_as_xyz",
    ]
    for idx, pos in enumerate(equiv_positions, 1):
        lines.append(" %d '%s'" % (idx, pos))
    lines.extend([
        "loop_",
        " _atom_type_symbol",
        " _atom_type_oxidation_number",
        " X 0",
        "loop_",
        " _atom_site_label",
        " _atom_site_type_symbol",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
        " _atom_site_occupancy",
        " X1 X 0.1 0.1 0.1 0.001",
    ])
    content = "\n".join(lines) + "\n"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, phase_name + "_LB.cif")
    else:
        fd, path = tempfile.mkstemp(suffix=".cif", prefix=phase_name + "_LB_")
        os.close(fd)
    Path(path).write_text(content)
    return path


# Pre-canned PDF phases (add new PDF entries here)
# Also loads from pdf_phases_cache.json for persistence
_PDF_CACHE_FILE = Path(__file__).parent / "pdf_phases_cache.json"
PDF_PHASES: Dict[str, dict] = {
    "Gd4P6O21": {
        "phase_name": "Gd4P6O21",
        "formula": "Gd4P6O21",
        "cell_a": 25.88, "cell_b": 25.88, "cell_c": 25.88,
        "cell_alpha": 90.0, "cell_beta": 90.0, "cell_gamma": 90.0,
        "space_group": "P 21 3", "it_number": 198,
        "equiv_positions": [
            "x, y, z",
            "-x+1/2, -y, z+1/2",
            "-x, y+1/2, -z+1/2",
            "x+1/2, -y+1/2, -z",
            "z, x, y",
            "z+1/2, -x+1/2, -y",
            "-z+1/2, -x, y+1/2",
            "-z, x+1/2, -y+1/2",
            "y, z, x",
            "-y, z+1/2, -x+1/2",
            "y+1/2, -z+1/2, -x",
            "-y+1/2, -z, x+1/2",
        ],
        "source": "PDF#00-030-0555",
        "notes": "Cubic, F(30)=3.7. Footnote: matches GdPO4 reflections.",
    },
}

# Merge cached PDF phases (persistent across sessions)
if _PDF_CACHE_FILE.exists():
    try:
        import json
        cached = json.loads(_PDF_CACHE_FILE.read_text())
        for k, v in cached.items():
            if k not in PDF_PHASES:
                PDF_PHASES[k] = v
    except Exception:
        pass
