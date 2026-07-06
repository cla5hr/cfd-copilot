from cfd_copilot.llm import extract_spec
from cfd_copilot.schema import CaseType, TurbulenceModel


def test_cavity_prompt():
    s = extract_spec("Lid-driven cavity at 2 m/s with viscosity 0.01", use_llm=False)
    assert s.case_type == CaseType.CAVITY
    assert s.velocity == 2.0
    assert s.fluid.nu == 0.01


def test_channel_prompt_with_reynolds_sets_nu():
    s = extract_spec("turbulent channel flow at 10 m/s, k-omega SST, Re 50000", use_llm=False)
    assert s.case_type == CaseType.CHANNEL
    assert s.turbulence == TurbulenceModel.K_OMEGA_SST
    # nu derived from Re: nu = U*H/Re = 10*1/50000 = 2e-4
    assert abs(s.fluid.nu - 2e-4) < 1e-9


def test_supersonic_prompt_selects_forward_step():
    s = extract_spec("supersonic flow over a step at Mach 3, 300 K, 101325 Pa", use_llm=False)
    assert s.case_type == CaseType.FORWARD_STEP
    assert s.mach == 3.0
    assert s.fluid.T == 300.0
    assert s.fluid.p == 101325.0


def test_rocket_keyword_routes_compressible():
    s = extract_spec("rocket exhaust high speed flow mach 2.5", use_llm=False)
    assert s.case_type == CaseType.FORWARD_STEP
    assert s.mach == 2.5
