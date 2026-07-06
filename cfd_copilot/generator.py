"""Render a :class:`CaseSpec` into a complete, on-disk OpenFOAM case.

Templates are deterministic and derived from validated OpenFOAM v2412 tutorials,
so the generated dictionaries are syntactically correct by construction. The LLM
only ever produces the spec; this module produces the files.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from cfd_copilot.schema import CaseSpec, CaseType, TurbulenceModel

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _files_for(spec: CaseSpec) -> List[str]:
    """Relative case paths to render for a given case type."""
    if spec.case_type == CaseType.CAVITY:
        return [
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "constant/transportProperties",
            "0/U",
            "0/p",
        ]
    if spec.case_type == CaseType.CHANNEL:
        files = [
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "0/U",
            "0/p",
            "0/nut",
            "0/k",
        ]
        if spec.turbulence == TurbulenceModel.K_EPSILON:
            files.append("0/epsilon")
        else:  # kOmegaSST
            files.append("0/omega")
        return files
    if spec.case_type == CaseType.FORWARD_STEP:
        return [
            "system/blockMeshDict",
            "system/controlDict",
            "system/fvSchemes",
            "system/fvSolution",
            "constant/thermophysicalProperties",
            "constant/turbulenceProperties",
            "0/U",
            "0/p",
            "0/T",
        ]
    raise ValueError(f"Unsupported case type: {spec.case_type}")


def _context(spec: CaseSpec) -> Dict:
    ctx: Dict = {
        "geometry": spec.geometry,
        "mesh": spec.mesh,
        "control": spec.control,
        "fluid": spec.fluid,
        "velocity": spec.velocity,
        "turbulence": spec.turbulence.value,
        "turb": spec.turbulence_initials(),
    }

    if spec.case_type == CaseType.FORWARD_STEP:
        length = spec.geometry.length
        height = spec.geometry.height
        depth = spec.geometry.depth
        step_x = round(0.2 * length, 6)
        step_h = round(0.2 * height, 6)
        # Cell divisions; shared block edges reuse the same variable so the mesh
        # is always conformal.
        density = max(spec.mesh.nx / max(step_x, 1e-9), 4.0)
        nx_inlet = max(int(round(step_x * density)), 4)
        nx_outlet = max(int(round((length - step_x) * density)), 4)
        ny_lower = max(int(round(step_h * density)), 2)
        ny_upper = max(int(round((height - step_h) * density)), 4)
        # Cp from gamma and specific gas constant for thermodynamic consistency.
        Cp = spec.fluid.gamma * spec.R_specific() / (spec.fluid.gamma - 1.0)
        ctx.update(
            {
                "length": length,
                "height": height,
                "step_x": step_x,
                "step_h": step_h,
                "zf": round(-depth / 2.0, 6),
                "zb": round(depth / 2.0, 6),
                "nx_inlet": nx_inlet,
                "nx_outlet": nx_outlet,
                "ny_lower": ny_lower,
                "ny_upper": ny_upper,
                "Cp": round(Cp, 4),
            }
        )

    return ctx


def generate_case(spec: CaseSpec, out_dir: Path, overwrite: bool = True) -> Path:
    """Render ``spec`` into ``out_dir`` and return the case directory path."""
    case_dir = Path(out_dir) / spec.name
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Case directory already exists: {case_dir}")
        shutil.rmtree(case_dir)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
    )

    ctx = _context(spec)
    for rel in _files_for(spec):
        template = env.get_template(f"{spec.case_type.value}/{rel}.jinja")
        rendered = template.render(**ctx)
        # Strip the leading import-statement blank line artifact.
        rendered = rendered.lstrip("\n")
        target = case_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")

    # ParaView marker file for convenient post-processing.
    (case_dir / f"{spec.name}.foam").write_text("", encoding="utf-8")
    # Persist the spec next to the case for provenance / reproducibility.
    (case_dir / "case_spec.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )
    return case_dir
