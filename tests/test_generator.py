import re

from cfd_copilot.generator import generate_case
from cfd_copilot.schema import CaseSpec, CaseType, TurbulenceModel


def test_cavity_files_written(tmp_path):
    s = CaseSpec(case_type=CaseType.CAVITY, name="cav", velocity=2.0)
    d = generate_case(s, tmp_path)
    for rel in ["system/blockMeshDict", "system/controlDict", "0/U", "0/p",
                "constant/transportProperties", "case_spec.json"]:
        assert (d / rel).exists(), rel
    assert "icoFoam" in (d / "system/controlDict").read_text()
    assert "(2 0 0)" in (d / "0/U").read_text()  # lid velocity


def test_channel_field_selection(tmp_path):
    s = CaseSpec(case_type=CaseType.CHANNEL, turbulence=TurbulenceModel.K_EPSILON)
    d = generate_case(s, tmp_path)
    assert (d / "0/epsilon").exists()
    assert not (d / "0/omega").exists()
    assert "kEpsilon" in (d / "constant/turbulenceProperties").read_text()


def test_forward_step_mesh_is_conformal(tmp_path):
    s = CaseSpec(case_type=CaseType.FORWARD_STEP, mach=3.0)
    d = generate_case(s, tmp_path)
    block = (d / "system/blockMeshDict").read_text()
    assert "rhoCentralFoam" not in block  # solver lives in controlDict
    # The two lower/upper blocks over the inlet region must share nx_inlet.
    hexes = re.findall(r"hex \([^)]*\) \((\d+) (\d+) 1\)", block)
    assert len(hexes) == 3
    nx_inlet_block0 = hexes[0][0]
    nx_inlet_block1 = hexes[1][0]
    assert nx_inlet_block0 == nx_inlet_block1  # conformal shared edge
    ny_upper_block1 = hexes[1][1]
    ny_upper_block2 = hexes[2][1]
    assert ny_upper_block1 == ny_upper_block2  # conformal shared edge


def test_overwrite(tmp_path):
    s = CaseSpec(case_type=CaseType.CAVITY, name="dup")
    generate_case(s, tmp_path)
    # Should not raise when overwrite=True (default).
    generate_case(s, tmp_path)
