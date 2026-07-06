from cfd_copilot.validator import extract_foam_errors

FATAL = """
Create time

--> FOAM FATAL IO ERROR:
keyword nu not found in dictionary "transportProperties"

file: /case/constant/transportProperties

    From function ...
FOAM exiting
"""

CHECKMESH = """
Checking geometry...
 ***Number of regions: 2
 ***High aspect ratio cells found.
"""


def test_extract_io_error():
    errs = extract_foam_errors(FATAL)
    assert any("nu not found" in e for e in errs)


def test_extract_checkmesh_markers():
    errs = extract_foam_errors(CHECKMESH)
    assert any("Number of regions" in e for e in errs)
