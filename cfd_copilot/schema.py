"""Structured representation of a CFD case.

This is the *contract* between the LLM and the deterministic template engine.
The LLM's only job is to fill this schema from natural language; it never writes
OpenFOAM syntax directly. Every field has a safe default, so a partially-specified
prompt still yields a runnable case.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CaseType(str, Enum):
    """Supported, template-backed canonical cases.

    Each maps to a validated OpenFOAM tutorial-derived template. Keeping the set
    small and well-tested is what makes generation reliable.
    """

    CAVITY = "cavity"            # lid-driven cavity, laminar, incompressible
    CHANNEL = "channel"          # 2D channel/duct, RANS, incompressible
    FORWARD_STEP = "forward_step"  # supersonic flow over a step, compressible


class FlowRegime(str, Enum):
    INCOMPRESSIBLE = "incompressible"
    COMPRESSIBLE = "compressible"


class TurbulenceModel(str, Enum):
    LAMINAR = "laminar"
    K_EPSILON = "kEpsilon"
    K_OMEGA_SST = "kOmegaSST"


# Solver chosen automatically from regime + steadiness; exposed for transparency.
class Solver(str, Enum):
    ICO_FOAM = "icoFoam"               # transient, laminar, incompressible
    SIMPLE_FOAM = "simpleFoam"         # steady, RANS, incompressible
    RHO_CENTRAL_FOAM = "rhoCentralFoam"  # transient, compressible (density-based)


class FluidProperties(BaseModel):
    # Incompressible
    nu: float = Field(1.0e-5, description="Kinematic viscosity [m^2/s]")
    rho: float = Field(1.0, description="Reference density [kg/m^3] (incompressible)")
    # Compressible (ideal gas)
    molWeight: float = Field(28.96, description="Molar weight [g/mol]")
    gamma: float = Field(1.4, description="Cp/Cv")
    Pr: float = Field(0.72, description="Prandtl number")
    mu: float = Field(1.8e-5, description="Dynamic viscosity [Pa.s] (compressible)")
    T: float = Field(300.0, description="Free-stream / initial temperature [K]")
    p: float = Field(1.0e5, description="Free-stream / initial pressure [Pa]")


class MeshSpec(BaseModel):
    nx: int = Field(40, ge=2, le=2000)
    ny: int = Field(40, ge=2, le=2000)
    nz: int = Field(1, ge=1, le=200)
    grading: float = Field(1.0, gt=0, description="Simple grading ratio")


class GeometrySpec(BaseModel):
    length: float = Field(1.0, gt=0, description="Streamwise length [m]")
    height: float = Field(1.0, gt=0, description="Cross-stream height [m]")
    depth: float = Field(0.1, gt=0, description="Out-of-plane thickness [m] (quasi-2D)")


class ControlSpec(BaseModel):
    end_time: float = Field(0.5, gt=0)
    delta_t: float = Field(0.005, gt=0)
    write_interval: float = Field(0.1, gt=0)
    # For steady (simpleFoam) these become iteration counts.
    max_iterations: int = Field(1000, ge=1)


class CaseSpec(BaseModel):
    """A complete, renderable description of a CFD case."""

    case_type: CaseType = CaseType.CAVITY
    name: str = Field("case", description="Folder/case name (sanitised)")
    description: str = Field("", description="Original natural-language request")

    flow_regime: FlowRegime = FlowRegime.INCOMPRESSIBLE
    turbulence: TurbulenceModel = TurbulenceModel.LAMINAR

    # Primary driving condition. Interpretation depends on case_type:
    #   cavity        -> lid velocity [m/s]
    #   channel       -> inlet velocity [m/s]
    #   forward_step  -> inlet velocity [m/s] (Mach used to derive it if given)
    velocity: float = Field(1.0, description="Characteristic velocity [m/s]")
    mach: Optional[float] = Field(None, description="Inlet Mach number (compressible)")

    fluid: FluidProperties = Field(default_factory=FluidProperties)
    geometry: GeometrySpec = Field(default_factory=GeometrySpec)
    mesh: MeshSpec = Field(default_factory=MeshSpec)
    control: ControlSpec = Field(default_factory=ControlSpec)

    @model_validator(mode="after")
    def _apply_case_defaults(self) -> "CaseSpec":
        self.name = _sanitize_name(self.name)

        if self.case_type == CaseType.CAVITY:
            self.flow_regime = FlowRegime.INCOMPRESSIBLE
            self.turbulence = TurbulenceModel.LAMINAR

        elif self.case_type == CaseType.CHANNEL:
            self.flow_regime = FlowRegime.INCOMPRESSIBLE
            if self.turbulence == TurbulenceModel.LAMINAR:
                self.turbulence = TurbulenceModel.K_OMEGA_SST

        elif self.case_type == CaseType.FORWARD_STEP:
            self.flow_regime = FlowRegime.COMPRESSIBLE
            self.turbulence = TurbulenceModel.LAMINAR
            # If a Mach number is given, derive velocity from speed of sound.
            if self.mach is not None:
                a = self.speed_of_sound()
                self.velocity = round(self.mach * a, 4)
            # The classic forward-step domain is a 3x1 channel; expand the
            # generic 1x1 default if the user did not specify a length.
            if self.geometry.length == 1.0:
                self.geometry.length = 3.0
            # Timing for a density-based compressible solver is dictated by the
            # CFL condition, not user preference: pick ~4 flow-through times and a
            # small seed time step (adjustTimeStep refines it from maxCo).
            transit = self.geometry.length / max(self.velocity, 1e-6)
            self.control.end_time = round(4.0 * transit, 6)
            self.control.write_interval = round(self.control.end_time / 5.0, 6)
            self.control.delta_t = 1.0e-6

        return self

    # ----------------------------------------------------------------- helpers
    @property
    def solver(self) -> Solver:
        if self.flow_regime == FlowRegime.COMPRESSIBLE:
            return Solver.RHO_CENTRAL_FOAM
        if self.case_type == CaseType.CAVITY:
            return Solver.ICO_FOAM
        return Solver.SIMPLE_FOAM

    @property
    def steady(self) -> bool:
        return self.solver == Solver.SIMPLE_FOAM

    def R_specific(self) -> float:
        """Specific gas constant R = R_universal / M."""
        return 8314.462618 / self.molWeight_grams()

    def molWeight_grams(self) -> float:
        return self.fluid.molWeight

    def speed_of_sound(self) -> float:
        return (self.fluid.gamma * self.R_specific() * self.fluid.T) ** 0.5

    def reynolds(self) -> float:
        """Reynolds number based on geometry height as the length scale."""
        if self.flow_regime == FlowRegime.INCOMPRESSIBLE:
            return self.velocity * self.geometry.height / max(self.fluid.nu, 1e-12)
        rho = self.fluid.p / (self.R_specific() * self.fluid.T)
        return rho * self.velocity * self.geometry.height / max(self.fluid.mu, 1e-12)

    def turbulence_initials(self) -> dict:
        """Estimate inlet k and omega from turbulence-intensity assumptions."""
        I = 0.05  # 5% turbulence intensity
        L = 0.07 * self.geometry.height  # mixing length ~ 7% of hydraulic scale
        k = 1.5 * (self.velocity * I) ** 2
        k = max(k, 1e-6)
        omega = (k ** 0.5) / max((0.09 ** 0.25) * L, 1e-9)
        epsilon = (0.09 ** 0.75) * (k ** 1.5) / max(L, 1e-9)
        return {"k": k, "omega": omega, "epsilon": epsilon}


def _sanitize_name(name: str) -> str:
    name = (name or "case").strip().lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    name = name.strip("_-") or "case"
    return name[:64]
