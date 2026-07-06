"""Streamlit UI for the CFD Case Copilot.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.config import settings
from cfd_copilot.openfoam import find_bashrc, openfoam_available

st.set_page_config(page_title="CFD Case Copilot", page_icon="🌀", layout="wide")

EXAMPLES = [
    "Lid-driven cavity at 1 m/s with kinematic viscosity 0.01",
    "Turbulent channel flow at 10 m/s using k-omega SST, Re 50000",
    "Supersonic flow over a forward step at Mach 3, air at 300 K and 101325 Pa",
]

if "prompt" not in st.session_state:
    st.session_state.prompt = EXAMPLES[0]

st.title("🌀 CFD Case Copilot")
st.caption(
    "Describe a flow problem in plain English. The agent extracts a structured "
    "spec, generates a validated OpenFOAM case, meshes it, checks it, and runs the "
    "solver — self-correcting from OpenFOAM's own error output."
)

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use local LLM (Ollama)", value=True, help="Off = deterministic parser")
    out_dir = st.text_input("Output directory", value="runs")
    st.divider()
    of_ok = openfoam_available()
    st.write("OpenFOAM:", "✅ found" if of_ok else "❌ not found")
    if of_ok:
        st.caption(find_bashrc() or "")
    st.caption(f"Chat model: `{settings.chat_model}`  ·  Embed: `{settings.embed_model}`")
    st.divider()
    st.markdown(
        "**How to run**\n\n"
        "1. Pick or type a prompt\n"
        "2. Click **Run simulation**\n\n"
        "Prefer a richer UI? Run `cfd-copilot serve` and open "
        "http://127.0.0.1:8000 in your browser."
    )

st.subheader("Case prompt")
cols = st.columns(len(EXAMPLES))
for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
    label = ["Cavity", "Channel", "Forward step"][i]
    if c.button(label, help=ex, use_container_width=True):
        st.session_state.prompt = ex

st.text_area("Describe your case", key="prompt", height=90)

run_full = st.button("Run simulation", type="primary", help="Generate, mesh, validate, and solve")
mesh_only = st.button("Mesh only", help="Generate + mesh + checkMesh (skip solver)")

if run_full or mesh_only:
    validate_only = mesh_only and not run_full
    state = AgentState(
        prompt=st.session_state.prompt,
        settings=settings,
        use_llm=use_llm,
        validate_only=validate_only,
        run_solver=not validate_only,
        out_dir=Path(out_dir),
    )
    label = "Meshing and validating..." if validate_only else "Running full simulation..."
    with st.status(label, expanded=True) as status:
        run_pipeline(state)
        for line in state.log:
            st.write(("⚠️ " if line.startswith(("repair", "warning")) else "✅ ") + line)
        status.update(
            label=f"Done — status: {state.status}",
            state="complete" if state.status == "success" else "error",
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted spec")
        if state.spec:
            st.json(state.spec.model_dump(mode="json"))
    with right:
        st.subheader("Results")
        if state.validation:
            st.metric("Mesh cells", state.validation.n_cells or "—")
            st.write(state.validation.summary())
        if state.run:
            st.write(state.run.summary())
            if state.run.final_residuals:
                st.bar_chart(
                    {k: v for k, v in state.run.final_residuals.items()},
                    horizontal=True,
                )
        if state.case_dir and state.case_dir.is_dir():
            entries = sorted(
                p.name
                for p in state.case_dir.iterdir()
                if p.is_dir() and (p.name == "0" or p.name.replace(".", "", 1).isdigit())
            )
            if entries:
                st.write("Time folders:", ", ".join(entries))
            elif validate_only:
                st.info(
                    "Only the `0/` folder exists because the solver was skipped. "
                    "Click **Run simulation** to produce time-step results."
                )
            foam = state.case_dir / f"{state.spec.name}.foam"
            st.markdown("**Open in ParaView**")
            st.code(f"paraview {foam}", language="bash")
