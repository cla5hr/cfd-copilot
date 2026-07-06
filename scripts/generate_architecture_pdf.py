#!/usr/bin/env python3
"""Generate CFD Case Copilot architecture guide PDF."""

from pathlib import Path

from fpdf import FPDF


class GuidePDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "CFD Case Copilot - Architecture Guide", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.multi_cell(self._width(), 8, title)
        self.ln(2)

    def sub_title(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self._width(), 7, title)
        self.ln(1)

    def _width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self._width(), 5.5, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(self._width(), 5.5, f"  - {text}")

    def code_block(self, text: str):
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(20, 20, 20)
        w = self._width()
        for line in text.splitlines():
            self.multi_cell(w, 4.5, "  " + line, fill=True)
        self.ln(2)


def build_pdf(output: Path) -> None:
    pdf = GuidePDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title page
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 120)
    pdf.ln(30)
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(w, 12, "CFD Case Copilot\nArchitecture Guide")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(
        w,
        7,
        "A concise reference for explaining how the project works:\n"
        "file layout, pipeline, key functions, and interview talking points.",
    )
    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "cfd-case-copilot / OpenFOAM v2412 / Ollama")

    pdf.add_page()

    pdf.section_title("One-liner")
    pdf.body(
        "CFD Case Copilot turns plain English (e.g. 'supersonic flow over a forward "
        "step at Mach 3') into a validated, runnable OpenFOAM case - meshed, solved, "
        "and self-corrected if something fails."
    )

    pdf.section_title("Core Design Idea (say this first)")
    pdf.body("The LLM never writes OpenFOAM syntax. That is the main reliability trick.")
    pdf.ln(1)
    pdf.sub_title("Responsibility split")
    pdf.bullet("LLM (Ollama qwen2.5-coder:7b): parse English into a small JSON schema (CaseSpec)")
    pdf.bullet("Jinja2 templates: render valid OpenFOAM dictionaries from that schema")
    pdf.bullet("OpenFOAM (blockMesh, checkMesh, solver): ground truth - did it actually work?")
    pdf.bullet("Agent loop: if OpenFOAM fails, apply deterministic fixes and retry")
    pdf.ln(2)
    pdf.body("Summary: LLM for understanding, templates for correctness, OpenFOAM for verification.")

    pdf.section_title("Project Layout")
    pdf.code_block(
        """cfd-case-copilot/
  cfd_copilot/           <- core Python package
    schema.py            <- data contract (CaseSpec)
    llm.py               <- natural language -> CaseSpec
    generator.py         <- CaseSpec -> OpenFOAM files
    openfoam.py          <- run blockMesh, solvers, etc.
    validator.py         <- mesh + quality check
    runner.py            <- run solver, parse residuals
    agent.py             <- orchestration loop
    config.py            <- env vars / defaults
    cli.py               <- cfd-copilot command
    api.py               <- FastAPI REST server
    rag.py               <- optional tutorial Q&A
    templates/           <- Jinja2 OpenFOAM dicts (3 case types)
      cavity/
      channel/
      forward_step/
  app/streamlit_app.py   <- web UI
  tests/                 <- unit + OpenFOAM integration tests
  runs/                  <- generated cases (output)
  pyproject.toml         <- package config, installs cfd-copilot CLI"""
    )

    pdf.section_title("The Pipeline (End to End)")
    pdf.code_block(
        """User prompt
    |
    v
extract_spec()          (llm.py)
    |
    v
CaseSpec                (schema.py)
    |
    v
generate_case()         (generator.py)
    |
    v
OpenFOAM files on disk  (system/, constant/, 0/)
    |
    v
validate_case()         (validator.py: blockMesh + checkMesh)
    |
    +-- mesh OK -----> run_solver() (runner.py) -----> Success
    |
    +-- mesh fail ---> node_repair() (agent.py) -----> regenerate & retry"""
    )
    pdf.body("CLI entry: cfd-copilot run \"...\" -> cli.py -> run_pipeline() in agent.py")

    pdf.section_title("Module-by-Module")

    pdf.sub_title("1. schema.py - the contract")
    pdf.body("Defines CaseSpec: the structured object everything else uses.")
    pdf.bullet("CaseType: cavity | channel | forward_step")
    pdf.bullet("FlowRegime: incompressible / compressible")
    pdf.bullet("TurbulenceModel: laminar | kEpsilon | kOmegaSST")
    pdf.bullet("Solver (auto-picked): icoFoam | simpleFoam | rhoCentralFoam")
    pdf.bullet("Key fields: velocity, mach, fluid, geometry, mesh, control")
    pdf.bullet("CaseSpec._apply_case_defaults(): e.g. Mach 3 -> velocity = Mach x speed of sound")
    pdf.bullet("CaseSpec.solver: picks solver from case type + regime")
    pdf.bullet("CaseSpec.reynolds(): computes Reynolds number for display/logging")

    pdf.sub_title("2. llm.py - natural language -> CaseSpec")
    pdf.bullet("extract_spec(prompt, settings, use_llm=True): main entry; returns CaseSpec")
    pdf.bullet("_extract_with_llm(): calls Ollama with structured output")
    pdf.bullet("_extract_with_rules(): regex/rule fallback (--no-llm, no Ollama needed)")
    pdf.bullet("_to_case_spec(): merges extraction into full CaseSpec with defaults")
    pdf.body("Flow: try LLM -> on failure fall back to rules -> always produce a valid CaseSpec.")

    pdf.sub_title("3. generator.py - CaseSpec -> files on disk")
    pdf.bullet("_files_for(spec): which dictionaries to render per case type")
    pdf.bullet("_context(spec): template variables (velocity, mesh, step geometry, etc.)")
    pdf.bullet("generate_case(spec, out_dir): renders Jinja2 -> writes system/, constant/, 0/")
    pdf.body("Also writes {case_name}.foam (ParaView) and case_spec.json (provenance).")
    pdf.body("Templates are derived from real OpenFOAM v2412 tutorials.")

    pdf.sub_title("4. openfoam.py - shell wrapper")
    pdf.bullet("find_bashrc(): auto-find OpenFOAM etc/bashrc")
    pdf.bullet("openfoam_available(): is OpenFOAM installed?")
    pdf.bullet("run_foam(utility, case_dir): run blockMesh, checkMesh, icoFoam, etc.")

    pdf.sub_title("5. validator.py - mesh feedback")
    pdf.bullet("validate_case(case_dir): runs blockMesh then checkMesh")
    pdf.bullet("extract_foam_errors(text): pulls readable errors from OpenFOAM logs")
    pdf.bullet("ValidationReport: ok, cell count, max skewness, errors")

    pdf.sub_title("6. runner.py - solver feedback")
    pdf.bullet("run_solver(spec, case_dir): runs the solver from spec.solver")
    pdf.bullet("RunReport: ok, final residuals, last time step, convergence")

    pdf.sub_title("7. agent.py - orchestration")
    pdf.body("AgentState holds: prompt, spec, case_dir, validation, run report, attempts, status.")
    pdf.bullet("node_extract(): extract_spec()")
    pdf.bullet("node_generate(): generate_case()")
    pdf.bullet("node_validate(): validate_case()")
    pdf.bullet("node_run(): run_solver()")
    pdf.bullet("node_repair(): deterministic fixes")
    pdf.bullet("run_pipeline(state): main while-loop (default path)")
    pdf.bullet("build_graph(): same nodes as LangGraph StateGraph (optional)")
    pdf.ln(1)
    pdf.body("Repair logic (node_repair):")
    pdf.bullet("Mesh failed -> reset to safe 40x40 grid")
    pdf.bullet("Solver diverged (transient) -> reduce deltaT by 5x")
    pdf.bullet("Solver did not converge (steady) -> double max iterations")
    pdf.body("Up to CFD_MAX_REPAIRS attempts (default 3).")

    pdf.sub_title("8. cli.py - how users run it")
    pdf.bullet("cfd-copilot spec: extract_spec() only")
    pdf.bullet("cfd-copilot generate: pipeline, validate only")
    pdf.bullet("cfd-copilot run: full pipeline + solver")
    pdf.bullet("cfd-copilot doctor: check OpenFOAM + Ollama")
    pdf.bullet("cfd-copilot build-rag / ask: RAG over tutorials")
    pdf.bullet("cfd-copilot serve: FastAPI server")

    pdf.sub_title("9. rag.py (optional)")
    pdf.bullet("gather_documents(): scan $FOAM_TUTORIALS for dictionary files")
    pdf.bullet("build_index(): embed with nomic-embed-text + store FAISS index")
    pdf.bullet("answer(question): retrieve snippets + LLM answer")

    pdf.add_page()
    pdf.section_title("Three Supported Case Types")
    pdf.bullet("cavity: lid-driven, laminar, incompressible -> icoFoam")
    pdf.bullet("channel: 2D duct, turbulent RANS -> simpleFoam")
    pdf.bullet("forward_step: supersonic, compressible -> rhoCentralFoam")
    pdf.body("Mach number -> velocity uses ideal gas: v = Mach x sqrt(gamma * R * T)")

    pdf.section_title("Example Output on Disk")
    pdf.code_block(
        """runs/forward_step_mach3_0/
  0/              <- initial fields (U, p, T)
  constant/       <- fluid properties
  system/         <- blockMeshDict, controlDict, fvSchemes, fvSolution
  case_spec.json  <- what the agent understood
  log.blockMesh
  log.checkMesh
  log.rhoCentralFoam
  forward_step_mach3_0.foam"""
    )

    pdf.section_title("30-Second Interview Soundbite")
    pdf.body(
        "\"It is an agentic CFD setup tool. The user describes a flow problem in English. "
        "A local LLM extracts structured parameters into a Pydantic schema - it never writes "
        "OpenFOAM syntax. Jinja2 templates derived from real OpenFOAM tutorials render the "
        "case files. Then we actually run blockMesh, checkMesh, and the solver. If something "
        "fails, a repair loop reads OpenFOAM error output and applies deterministic fixes - "
        "coarser mesh, smaller time step - and retries. Ground truth comes from OpenFOAM, "
        "not from the model guessing.\""
    )

    pdf.section_title("Why Not Just Prompt the LLM to Write OpenFOAM?")
    pdf.body(
        "\"A 7B local model cannot reliably emit valid blockMeshDict / fvSchemes syntax. "
        "Splitting responsibilities - LLM for understanding, templates for syntax, OpenFOAM "
        "for verification - is what makes it actually runnable. Every template was tested "
        "end-to-end on OpenFOAM v2412.\""
    )

    pdf.section_title("Tech Stack")
    pdf.bullet("Python 3.10+, Pydantic, Jinja2, Typer/Rich (CLI)")
    pdf.bullet("Ollama: qwen2.5-coder:7b (chat), nomic-embed-text (embeddings)")
    pdf.bullet("LangChain / LangGraph: structured LLM output + optional graph")
    pdf.bullet("OpenFOAM v2412: meshing and solvers")
    pdf.bullet("FastAPI / Streamlit: API and UI")
    pdf.bullet("pytest: unit tests + integration tests (skip if OpenFOAM missing)")

    pdf.section_title("Quick Start Commands")
    pdf.code_block(
        """cd cfd-case-copilot
source .venv/bin/activate
pip install -e ".[agent,api,dev]"

cfd-copilot doctor
cfd-copilot spec "turbulent channel flow at 10 m/s"
cfd-copilot generate "supersonic flow over a forward step at Mach 3"
cfd-copilot run "lid-driven cavity at 2 m/s, nu 0.01"
paraview runs/forward_step_mach3_0/forward_step_mach3_0.foam"""
    )

    pdf.output(str(output))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "CFD_Case_Copilot_Architecture_Guide.pdf"
    build_pdf(out)
    print(f"Wrote {out}")
