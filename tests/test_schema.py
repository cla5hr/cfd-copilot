from cfd_copilot.schema import CaseSpec, CaseType, Solver, TurbulenceModel


def test_cavity_defaults():
    s = CaseSpec(case_type=CaseType.CAVITY, name="My Cavity!")
    assert s.solver == Solver.ICO_FOAM
    assert s.turbulence == TurbulenceModel.LAMINAR
    assert s.name == "my_cavity"  # sanitised
    assert not s.steady


def test_channel_defaults_turbulent():
    s = CaseSpec(case_type=CaseType.CHANNEL)
    assert s.solver == Solver.SIMPLE_FOAM
    assert s.turbulence == TurbulenceModel.K_OMEGA_SST
    assert s.steady


def test_forward_step_is_compressible_and_uses_mach():
    s = CaseSpec(case_type=CaseType.FORWARD_STEP, mach=3.0)
    assert s.solver == Solver.RHO_CENTRAL_FOAM
    # Mach 3 in air at 300 K ~ 1041 m/s
    assert 1000 < s.velocity < 1100
    # forward-step domain auto-expands to length 3
    assert s.geometry.length == 3.0
    # timing recomputed for the compressible solver
    assert s.control.end_time < 0.1


def test_reynolds_incompressible():
    s = CaseSpec(case_type=CaseType.CHANNEL, velocity=10.0)
    s.fluid.nu = 1e-3
    s.geometry.height = 1.0
    assert abs(s.reynolds() - 1e4) < 1e-6


def test_speed_of_sound_air():
    s = CaseSpec(case_type=CaseType.FORWARD_STEP)
    a = s.speed_of_sound()
    assert 330 < a < 360  # ~347 m/s at 300 K
