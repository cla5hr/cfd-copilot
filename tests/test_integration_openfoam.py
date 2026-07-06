"""End-to-end tests that actually mesh and run cases.

Skipped automatically when OpenFOAM is not available (e.g. plain CI), so the rest
of the suite still passes everywhere.
"""

from pathlib import Path

import pytest

from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.openfoam import openfoam_available

pytestmark = pytest.mark.skipif(
    not openfoam_available(), reason="OpenFOAM not available in this environment"
)


def test_cavity_runs(tmp_path):
    st = AgentState(
        prompt="lid-driven cavity at 1 m/s, viscosity 0.01",
        use_llm=False,
        out_dir=Path(tmp_path),
    )
    run_pipeline(st)
    assert st.status == "success"
    assert st.validation.ok and st.validation.n_cells > 0
    assert st.run.ok and st.run.completed


def test_forward_step_meshes(tmp_path):
    # Validate-only keeps it fast; meshing the 3-block compressible case is the
    # part most likely to break, so this is a high-value smoke test.
    st = AgentState(
        prompt="supersonic flow over a forward step at Mach 3",
        use_llm=False,
        validate_only=True,
        out_dir=Path(tmp_path),
    )
    run_pipeline(st)
    assert st.status == "success"
    assert st.validation.ok
