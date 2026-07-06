"""Validate a generated case by meshing it and checking mesh quality.

This is the agent's feedback signal: instead of trusting the LLM, we ask OpenFOAM
itself whether the case is well-formed, and turn any failure into a concise,
actionable error message the agent can react to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cfd_copilot.openfoam import CommandResult, run_foam


@dataclass
class ValidationReport:
    ok: bool
    stage: str  # "blockMesh" | "checkMesh"
    n_cells: Optional[int] = None
    max_non_ortho: Optional[float] = None
    max_skewness: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    logs: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            extra = []
            if self.n_cells is not None:
                extra.append(f"{self.n_cells} cells")
            if self.max_non_ortho is not None:
                extra.append(f"maxNonOrtho={self.max_non_ortho:g}")
            if self.max_skewness is not None:
                extra.append(f"maxSkewness={self.max_skewness:g}")
            return "Mesh valid (" + ", ".join(extra) + ")" if extra else "Mesh valid"
        head = self.errors[0] if self.errors else "unknown error"
        return f"Validation failed at {self.stage}: {head}"


def extract_foam_errors(text: str) -> List[str]:
    """Pull concise messages out of OpenFOAM FATAL ERROR / IO ERROR blocks."""
    errors: List[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "FOAM FATAL" in line:
            # Capture up to the next few non-empty, human-readable lines.
            chunk = []
            for nxt in lines[i + 1 : i + 8]:
                s = nxt.strip()
                if not s:
                    if chunk:
                        break
                    continue
                if s.startswith("From ") or s.startswith("in file") or s.startswith("#"):
                    continue
                chunk.append(s)
            if chunk:
                errors.append(" ".join(chunk)[:500])
    # checkMesh "*** " failure markers.
    for line in lines:
        if line.strip().startswith("***"):
            errors.append(line.strip()[:500])
    return errors


def _check_completed(res: CommandResult) -> bool:
    return res.ok and res.stdout.rstrip().endswith("End")


def validate_case(case_dir: Path, timeout: int = 600) -> ValidationReport:
    case_dir = Path(case_dir)

    block = run_foam("blockMesh", case_dir, timeout=timeout)
    if not _check_completed(block):
        errs = extract_foam_errors(block.stdout + "\n" + block.stderr) or [
            block.stderr.strip() or block.tail(8)
        ]
        return ValidationReport(
            ok=False,
            stage="blockMesh",
            errors=errs,
            logs=[p for p in [block.log_path] if p],
        )

    check = run_foam("checkMesh", case_dir, timeout=timeout)
    report = ValidationReport(ok=True, stage="checkMesh", logs=[])
    for p in (block.log_path, check.log_path):
        if p:
            report.logs.append(p)

    out = check.stdout
    report.n_cells = _grab_int(out, r"cells:\s+(\d+)")
    report.max_non_ortho = _grab_float(out, r"non-orthogonality Max:\s*([\d.eE+-]+)")
    report.max_skewness = _grab_float(out, r"Max skewness\s*=\s*([\d.eE+-]+)")

    if "Mesh OK." not in out:
        report.ok = False
        report.errors = extract_foam_errors(out) or ["checkMesh did not report 'Mesh OK.'"]
    else:
        # Soft quality warnings (do not fail the case, but surface them).
        if report.max_non_ortho is not None and report.max_non_ortho > 70:
            report.warnings.append(
                f"High mesh non-orthogonality ({report.max_non_ortho:g} deg)"
            )
        if report.max_skewness is not None and report.max_skewness > 4:
            report.warnings.append(f"High mesh skewness ({report.max_skewness:g})")
    return report


def _grab_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _grab_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None
