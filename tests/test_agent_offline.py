"""Agent loop tests that mock OpenFOAM, so they run anywhere (CI included)."""

from pathlib import Path

import cfd_copilot.agent as agent
from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.runner import RunReport
from cfd_copilot.validator import ValidationReport


def _patch_generate(monkeypatch):
    monkeypatch.setattr(agent, "generate_case", lambda spec, out: Path(out) / spec.name)


def test_repairs_then_succeeds(monkeypatch):
    _patch_generate(monkeypatch)
    calls = {"n": 0}

    def fake_validate(case_dir, timeout=600):
        calls["n"] += 1
        if calls["n"] < 3:  # fail first two times
            return ValidationReport(ok=False, stage="blockMesh", errors=["boom"])
        return ValidationReport(ok=True, stage="checkMesh", n_cells=100)

    monkeypatch.setattr(agent, "validate_case", fake_validate)

    st = AgentState(prompt="lid-driven cavity", use_llm=False, validate_only=True)
    run_pipeline(st)

    assert st.status == "success"
    assert st.attempts == 2  # two repairs before success
    assert any("repair" in line for line in st.log)


def test_gives_up_after_max_attempts(monkeypatch):
    _patch_generate(monkeypatch)
    monkeypatch.setattr(
        agent,
        "validate_case",
        lambda case_dir, timeout=600: ValidationReport(ok=False, stage="blockMesh", errors=["x"]),
    )

    st = AgentState(prompt="lid-driven cavity", use_llm=False, validate_only=True)
    run_pipeline(st)

    assert st.status == "failed"
    assert st.attempts == st.settings.max_repair_attempts


def test_solver_failure_triggers_repair(monkeypatch):
    _patch_generate(monkeypatch)
    monkeypatch.setattr(
        agent,
        "validate_case",
        lambda case_dir, timeout=600: ValidationReport(ok=True, stage="checkMesh", n_cells=100),
    )
    runs = {"n": 0}

    def fake_run(spec, case_dir, timeout=3600):
        runs["n"] += 1
        ok = runs["n"] >= 2  # fail once, then succeed
        return RunReport(ok=ok, solver=spec.solver.value, completed=ok, errors=[] if ok else ["nan"])

    monkeypatch.setattr(agent, "run_solver", fake_run)

    st = AgentState(prompt="channel flow at 10 m/s", use_llm=False)
    run_pipeline(st)

    assert st.status == "success"
    assert runs["n"] == 2
    assert st.attempts == 1
