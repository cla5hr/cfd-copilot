"""CFD Case Copilot.

An agentic assistant that turns natural-language descriptions of fluid-flow
problems into validated, runnable OpenFOAM cases.

Design principle (the reason this works with a small local LLM):
    The LLM never writes raw OpenFOAM dictionary files. It only extracts a
    structured ``CaseSpec`` (JSON). Validated Jinja2 templates render the actual
    case, and OpenFOAM itself (blockMesh / checkMesh / the solver) is the ground
    truth that the agent uses to verify and self-correct.
"""

from cfd_copilot.schema import CaseSpec

__all__ = ["CaseSpec"]
__version__ = "0.1.0"
