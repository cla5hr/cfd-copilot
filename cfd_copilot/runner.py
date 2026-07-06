"""Run the solver for a generated case and summarise convergence.

Like the validator, this turns raw solver output into a compact, structured
report the agent (and the user) can reason about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from cfd_copilot.openfoam import run_foam
from cfd_copilot.schema import CaseSpec
from cfd_copilot.validator import extract_foam_errors


@dataclass
class RunReport:
    ok: bool
    solver: str
    completed: bool = False
    converged: Optional[bool] = None  # steady solvers only
    iterations: Optional[int] = None
    last_time: Optional[str] = None
    final_residuals: Dict[str, float] = field(default_factory=dict)
    residual_history: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    log_path: Optional[Path] = None

    def summary(self) -> str:
        if not self.ok:
            head = self.errors[0] if self.errors else "solver failed"
            return f"Solver {self.solver} failed: {head}"
        bits = [f"{self.solver} finished"]
        if self.converged is True:
            bits.append(f"converged in {self.iterations} iterations")
        elif self.converged is False:
            bits.append(f"did NOT reach residual targets (ran {self.iterations} iters)")
        if self.last_time is not None:
            bits.append(f"last time/iter = {self.last_time}")
        if self.final_residuals:
            res = ", ".join(f"{k}={v:.2e}" for k, v in self.final_residuals.items())
            bits.append(f"final residuals: {res}")
        return "; ".join(bits)


_RESIDUAL_RE = re.compile(
    r"Solving for (\w+), Initial residual = ([\d.eE+-]+)"
)
_TIME_RE = re.compile(r"^Time = (\S+)", re.MULTILINE)
_CONVERGED_RE = re.compile(r"solution converged in (\d+) iterations")


def parse_residual_history(text: str) -> Tuple[Dict[str, float], Dict[str, List[Tuple[float, float]]]]:
    """Parse solver log into final residuals and per-field history."""
    finals: Dict[str, float] = {}
    history: Dict[str, List[Tuple[float, float]]] = {}
    step = 0.0
    for line in text.splitlines():
        tm = _TIME_RE.match(line.strip())
        if tm:
            try:
                step = float(tm.group(1))
            except ValueError:
                pass
            continue
        rm = _RESIDUAL_RE.search(line)
        if rm:
            field_name, value = rm.group(1), rm.group(2)
            try:
                val = float(value)
            except ValueError:
                continue
            finals[field_name] = val
            history.setdefault(field_name, []).append((step, val))
    return finals, history


def run_solver(spec: CaseSpec, case_dir: Path, timeout: int = 3600) -> RunReport:
    solver = spec.solver.value
    res = run_foam(solver, Path(case_dir), timeout=timeout)
    report = RunReport(ok=res.ok, solver=solver, log_path=res.log_path)

    out = res.stdout or ""
    report.completed = out.rstrip().endswith("End")

    report.final_residuals, report.residual_history = parse_residual_history(out)

    times = _TIME_RE.findall(out)
    if times:
        report.last_time = times[-1]

    conv = _CONVERGED_RE.search(out)
    if spec.steady:
        if conv:
            report.converged = True
            report.iterations = int(conv.group(1))
        else:
            report.converged = False
            if times:
                try:
                    report.iterations = int(float(times[-1]))
                except ValueError:
                    pass

    if not res.ok or not report.completed:
        report.ok = False
        report.errors = extract_foam_errors(out + "\n" + (res.stderr or "")) or [
            res.stderr.strip() or res.tail(10)
        ]
    return report
