"""FastAPI backend exposing the copilot over HTTP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.config import settings as default_settings
from cfd_copilot.llm import extract_spec
from cfd_copilot.openfoam import find_bashrc, openfoam_available

app = FastAPI(
    title="CFD Case Copilot",
    description="Natural language -> validated, runnable OpenFOAM cases.",
    version="0.1.0",
)

_STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    page = _STATIC / "index.html"
    if page.is_file():
        return FileResponse(page)
    return {"message": "CFD Case Copilot API", "docs": "/docs"}


class SpecRequest(BaseModel):
    prompt: str
    use_llm: bool = True


class RunRequest(BaseModel):
    prompt: str
    use_llm: bool = True
    validate_only: bool = False
    run_solver: bool = True
    out_dir: str = "runs"


class AskRequest(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "openfoam_available": openfoam_available(),
        "foam_bashrc": find_bashrc(),
    }


@app.post("/spec")
def spec(req: SpecRequest) -> Dict[str, Any]:
    case = extract_spec(req.prompt, default_settings, use_llm=req.use_llm)
    return {"solver": case.solver.value, "spec": case.model_dump(mode="json")}


@app.post("/run")
def run(req: RunRequest) -> Dict[str, Any]:
    state = AgentState(
        prompt=req.prompt,
        settings=default_settings,
        use_llm=req.use_llm,
        validate_only=req.validate_only,
        run_solver=req.run_solver,
        out_dir=Path(req.out_dir),
    )
    run_pipeline(state)
    return _serialize_state(state)


@app.post("/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    from cfd_copilot.rag import answer

    return {"answer": answer(req.question, k=req.k, settings=default_settings)}


def _serialize_state(state: AgentState) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "status": state.status,
        "log": state.log,
        "case_dir": str(state.case_dir) if state.case_dir else None,
        "spec": state.spec.model_dump(mode="json") if state.spec else None,
    }
    if state.validation is not None:
        out["validation"] = {
            "ok": state.validation.ok,
            "summary": state.validation.summary(),
            "n_cells": state.validation.n_cells,
            "max_non_ortho": state.validation.max_non_ortho,
            "max_skewness": state.validation.max_skewness,
            "errors": state.validation.errors,
            "warnings": state.validation.warnings,
        }
    if state.run is not None:
        history: Dict[str, List[List[float]]] = {}
        for field, points in state.run.residual_history.items():
            history[field] = [[step, val] for step, val in points]
        out["run"] = {
            "ok": state.run.ok,
            "summary": state.run.summary(),
            "converged": state.run.converged,
            "iterations": state.run.iterations,
            "last_time": state.run.last_time,
            "final_residuals": state.run.final_residuals,
            "residual_history": history,
            "errors": state.run.errors,
        }
    return out
