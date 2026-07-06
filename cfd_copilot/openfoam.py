"""Thin wrapper around OpenFOAM command-line utilities.

Handles the one thing that trips people up: making sure the OpenFOAM environment
(``etc/bashrc``) is sourced so the solvers and their shared libraries are found.
If OpenFOAM is already on ``PATH`` (e.g. you sourced it in your shell), commands
run directly; otherwise we auto-detect a ``bashrc`` and source it per command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional

_BASHRC_GLOBS = [
    "/usr/lib/openfoam/openfoam*/etc/bashrc",
    "/opt/openfoam*/etc/bashrc",
    "/usr/share/openfoam/etc/bashrc",
    str(Path.home() / "OpenFOAM/OpenFOAM-*/etc/bashrc"),
]


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    log_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def tail(self, n: int = 15) -> str:
        lines = (self.stdout or "").splitlines()
        return "\n".join(lines[-n:])


def find_bashrc() -> Optional[str]:
    """Locate an OpenFOAM ``etc/bashrc`` to source, if any."""
    env = os.environ.get("FOAM_BASHRC")
    if env and Path(env).exists():
        return env
    # If WM_PROJECT_DIR is set the environment is likely already active.
    wm = os.environ.get("WM_PROJECT_DIR")
    if wm and (Path(wm) / "etc/bashrc").exists():
        return str(Path(wm) / "etc/bashrc")
    for pattern in _BASHRC_GLOBS:
        matches = sorted(glob(pattern))
        if matches:
            return matches[-1]  # newest version
    return None


def openfoam_available() -> bool:
    return shutil.which("blockMesh") is not None or find_bashrc() is not None


def run_foam(
    utility: str,
    case_dir: Path,
    args: Optional[List[str]] = None,
    timeout: Optional[int] = 1800,
    write_log: bool = True,
) -> CommandResult:
    """Run an OpenFOAM utility/solver inside ``case_dir``.

    Returns a :class:`CommandResult`; never raises on solver failure (the agent
    inspects the result and decides what to do).
    """
    case_dir = Path(case_dir)
    args = args or []
    inner = " ".join([utility, *args])

    bashrc = find_bashrc()
    if shutil.which(utility) is not None:
        # Already on PATH; run directly.
        shell_cmd = f"cd {_q(case_dir)} && {inner}"
    elif bashrc is not None:
        shell_cmd = f"source {_q(bashrc)} >/dev/null 2>&1 && cd {_q(case_dir)} && {inner}"
    else:
        return CommandResult(
            command=inner,
            returncode=127,
            stdout="",
            stderr=(
                "OpenFOAM not found. Source your OpenFOAM etc/bashrc or set "
                "FOAM_BASHRC to its path."
            ),
        )

    try:
        proc = subprocess.run(
            ["bash", "-lc", shell_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"Timed out after {timeout}s"
        code = 124

    log_path = None
    if write_log:
        log_path = case_dir / f"log.{utility}"
        try:
            log_path.write_text((stdout or "") + "\n" + (stderr or ""), encoding="utf-8")
        except OSError:
            log_path = None

    return CommandResult(inner, code, stdout, stderr, log_path)


def _q(path) -> str:
    """Quote a path for safe shell embedding."""
    s = str(path)
    return "'" + s.replace("'", "'\\''") + "'"
