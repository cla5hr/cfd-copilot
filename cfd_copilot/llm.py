"""Natural language -> structured :class:`CaseSpec`.

Key reliability decision: a small local model (qwen2.5-coder:7b) is *not* asked to
write OpenFOAM. It is only asked to fill a small, flat JSON schema. Even that is
backed by a deterministic rule-based parser, so the pipeline still works with no
LLM at all (useful for CI and for the cloud demo).
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from cfd_copilot.config import Settings, settings as default_settings
from cfd_copilot.schema import (
    CaseSpec,
    CaseType,
    FluidProperties,
    GeometrySpec,
    MeshSpec,
    TurbulenceModel,
)


class ExtractionResult(BaseModel):
    """Flat, LLM-friendly view of the parameters we try to extract."""

    case_type: CaseType = Field(
        CaseType.CAVITY,
        description=(
            "cavity = lid-driven cavity (laminar, incompressible); "
            "channel = 2D channel/duct flow (turbulent RANS, incompressible); "
            "forward_step = supersonic flow over a forward-facing step "
            "(compressible, e.g. rocket/nozzle/shock problems)."
        ),
    )
    name: Optional[str] = Field(None, description="Short case name, snake_case")
    velocity: Optional[float] = Field(None, description="Velocity in m/s")
    mach: Optional[float] = Field(None, description="Mach number (compressible only)")
    reynolds: Optional[float] = Field(None, description="Reynolds number if stated")
    turbulence: Optional[TurbulenceModel] = None
    nu: Optional[float] = Field(None, description="Kinematic viscosity m^2/s")
    length: Optional[float] = Field(None, description="Domain length in m")
    height: Optional[float] = Field(None, description="Domain height in m")
    temperature: Optional[float] = Field(None, description="Temperature in K")
    pressure: Optional[float] = Field(None, description="Pressure in Pa")
    nx: Optional[int] = Field(None, description="Cells in x")
    ny: Optional[int] = Field(None, description="Cells in y")


SYSTEM_PROMPT = """You are a CFD setup assistant. Convert the user's request into \
structured parameters for an OpenFOAM case. Only choose from these case types:

- cavity: a lid-driven cavity. Laminar, incompressible. Driven by a moving lid.
- channel: 2D channel / duct / pipe-like internal flow. Turbulent (RANS), \
incompressible. Has an inlet and outlet.
- forward_step: supersonic / compressible external flow over a forward-facing step. \
Use this for high-speed, supersonic, shock, nozzle or rocket-exhaust style problems.

Rules:
- Extract numeric values with their correct SI units (m/s, m, K, Pa, m^2/s).
- If a Mach number is given, set 'mach' and choose forward_step.
- If the flow is described as turbulent, pick a turbulence model (kOmegaSST by default).
- Leave a field null if the user did not specify it. Do not invent values.
- Respond with the structured object only."""


def extract_spec(
    prompt: str,
    settings: Settings = default_settings,
    use_llm: bool = True,
) -> CaseSpec:
    """Return a :class:`CaseSpec` for a natural-language ``prompt``."""
    extraction: Optional[ExtractionResult] = None
    if use_llm:
        try:
            extraction = _extract_with_llm(prompt, settings)
        except Exception:
            extraction = None  # graceful fallback to rules
    if extraction is None:
        extraction = _extract_with_rules(prompt)

    return _to_case_spec(prompt, extraction)


def _extract_with_llm(prompt: str, settings: Settings) -> ExtractionResult:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )
    structured = llm.with_structured_output(ExtractionResult)
    result = structured.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    if isinstance(result, ExtractionResult):
        return result
    return ExtractionResult.model_validate(result)


def _extract_with_rules(prompt: str) -> ExtractionResult:
    """A deterministic, dependency-free parser. Good enough to drive a demo."""
    text = prompt.lower()
    res = ExtractionResult()

    # --- case type ---
    if any(w in text for w in ["cavity", "lid-driven", "lid driven"]):
        res.case_type = CaseType.CAVITY
    elif any(
        w in text
        for w in ["supersonic", "compressible", "shock", "nozzle", "rocket", "mach", "step"]
    ):
        res.case_type = CaseType.FORWARD_STEP
    elif any(w in text for w in ["channel", "duct", "pipe", "inlet", "turbulent"]):
        res.case_type = CaseType.CHANNEL

    # --- turbulence ---
    if "k-epsilon" in text or "kepsilon" in text or "k epsilon" in text:
        res.turbulence = TurbulenceModel.K_EPSILON
    elif "k-omega" in text or "komega" in text or "sst" in text:
        res.turbulence = TurbulenceModel.K_OMEGA_SST
    elif "laminar" in text:
        res.turbulence = TurbulenceModel.LAMINAR

    # --- numbers ---
    res.mach = _num(text, r"mach\s*(?:number)?\s*(?:of|=|:)?\s*([\d.]+)")
    if res.mach is None:
        res.mach = _num(text, r"\bm\s*=\s*([\d.]+)")
    if res.mach is not None and res.case_type != CaseType.FORWARD_STEP:
        res.case_type = CaseType.FORWARD_STEP
    res.velocity = _num(text, r"([\d.]+)\s*m\s*/?\s*s")
    res.reynolds = _num(text, r"\bre\s*(?:=|:|of|number)?\s*([\d.eE+]+)")
    res.nu = _num(text, r"(?:nu|viscosity)\s*(?:=|:|of)?\s*([\d.eE+-]+)")
    res.temperature = _num(text, r"([\d.]+)\s*k\b")
    res.pressure = _num(text, r"([\d.]+)\s*(?:pa|pascal)")
    return res


def _to_case_spec(prompt: str, ex: ExtractionResult) -> CaseSpec:
    fluid = FluidProperties()
    if ex.nu is not None:
        fluid.nu = ex.nu
    if ex.temperature is not None:
        fluid.T = ex.temperature
    if ex.pressure is not None:
        fluid.p = ex.pressure

    geometry = GeometrySpec()
    if ex.length is not None:
        geometry.length = ex.length
    if ex.height is not None:
        geometry.height = ex.height

    mesh = MeshSpec()
    if ex.nx is not None:
        mesh.nx = ex.nx
    if ex.ny is not None:
        mesh.ny = ex.ny

    # Derive nu from Reynolds number if viscosity not given (incompressible cases).
    if ex.reynolds and ex.velocity and ex.nu is None and ex.case_type != CaseType.FORWARD_STEP:
        fluid.nu = ex.velocity * geometry.height / ex.reynolds

    kwargs = dict(
        case_type=ex.case_type,
        name=ex.name or _auto_name(ex),
        description=prompt,
        fluid=fluid,
        geometry=geometry,
        mesh=mesh,
    )
    if ex.velocity is not None:
        kwargs["velocity"] = ex.velocity
    if ex.mach is not None:
        kwargs["mach"] = ex.mach
    if ex.turbulence is not None:
        kwargs["turbulence"] = ex.turbulence

    return CaseSpec(**kwargs)


def _auto_name(ex: ExtractionResult) -> str:
    base = ex.case_type.value
    if ex.mach is not None:
        return f"{base}_mach{str(ex.mach).replace('.', '_')}"
    if ex.velocity is not None:
        return f"{base}_u{str(ex.velocity).replace('.', '_')}"
    return base


def _num(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
