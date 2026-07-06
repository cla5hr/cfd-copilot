"""Command-line interface for the CFD Case Copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from cfd_copilot.agent import AgentState, run_pipeline
from cfd_copilot.config import settings as default_settings
from cfd_copilot.llm import extract_spec
from cfd_copilot.openfoam import find_bashrc, openfoam_available

app = typer.Typer(
    add_completion=False,
    help="Turn natural-language flow descriptions into validated OpenFOAM cases.",
)
console = Console()


def _settings(no_llm: bool, model: Optional[str], out: Optional[Path]):
    s = default_settings
    if model:
        s = s.with_overrides(chat_model=model)
    return s


@app.command()
def spec(
    prompt: str = typer.Argument(..., help="Natural-language case description"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use the rule-based parser only"),
    model: Optional[str] = typer.Option(None, help="Override the Ollama chat model"),
):
    """Show the structured CaseSpec extracted from a prompt (no files written)."""
    s = _settings(no_llm, model, None)
    case = extract_spec(prompt, s, use_llm=not no_llm)
    console.print(Panel.fit(f"[bold]{case.case_type.value}[/]  ->  solver [cyan]{case.solver.value}[/]"))
    console.print(Syntax(case.model_dump_json(indent=2), "json", word_wrap=True))


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Natural-language case description"),
    out: Path = typer.Option(Path("runs"), help="Output directory"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use the rule-based parser only"),
    model: Optional[str] = typer.Option(None, help="Override the Ollama chat model"),
):
    """Generate (and validate) a case, but do not run the solver."""
    _pipeline(prompt, out, no_llm, model, validate_only=True)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Natural-language case description"),
    out: Path = typer.Option(Path("runs"), help="Output directory"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use the rule-based parser only"),
    no_solver: bool = typer.Option(False, "--no-solver", help="Mesh+validate, skip solver"),
    model: Optional[str] = typer.Option(None, help="Override the Ollama chat model"),
):
    """Run the full agent loop: generate -> validate -> (repair) -> solve."""
    _pipeline(prompt, out, no_llm, model, validate_only=False, run_solver=not no_solver)


@app.command()
def doctor():
    """Check that OpenFOAM (and optionally Ollama) are reachable."""
    of = openfoam_available()
    console.print(f"OpenFOAM available: {'[green]yes[/]' if of else '[red]no[/]'}")
    if of:
        console.print(f"  bashrc: {find_bashrc()}")
    else:
        console.print("  [yellow]Source your OpenFOAM etc/bashrc or set FOAM_BASHRC.[/]")
    try:
        import langchain_ollama  # noqa: F401

        console.print("Ollama client lib: [green]installed[/]")
    except Exception:
        console.print("Ollama client lib: [yellow]not installed[/] (rule-based parser still works)")


@app.command("build-rag")
def build_rag(
    tutorials: Optional[Path] = typer.Option(None, help="Path to OpenFOAM tutorials"),
):
    """Build the RAG index over OpenFOAM tutorial dictionaries (needs Ollama)."""
    from cfd_copilot.rag import build_index

    with console.status("Indexing OpenFOAM tutorials..."):
        n = build_index(default_settings, tutorials_dir=tutorials)
    console.print(f"[green]Indexed {n} dictionary files[/] -> {default_settings.rag_index_dir}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="An OpenFOAM question"),
    k: int = typer.Option(4, help="Number of snippets to retrieve"),
):
    """Answer an OpenFOAM question grounded in the indexed tutorials (needs Ollama)."""
    from cfd_copilot.rag import answer

    console.print(Panel(answer(question, k=k, settings=default_settings), title="Answer"))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
):
    """Launch the FastAPI server."""
    import uvicorn

    uvicorn.run("cfd_copilot.api:app", host=host, port=port, reload=False)


def _pipeline(prompt, out, no_llm, model, validate_only=False, run_solver=True):
    s = _settings(no_llm, model, out)
    state = AgentState(
        prompt=prompt,
        settings=s,
        use_llm=not no_llm,
        validate_only=validate_only,
        run_solver=run_solver,
        out_dir=Path(out),
    )
    run_pipeline(state)

    for line in state.log:
        prefix = "[yellow]·[/]" if line.startswith(("repair", "warning")) else "[green]✓[/]"
        console.print(f"{prefix} {line}")

    color = "green" if state.status == "success" else "red"
    console.print(Panel.fit(f"[{color}]status: {state.status}[/]"))
    if state.case_dir:
        console.print(f"Case directory: [bold]{state.case_dir}[/]")
        console.print(
            "Post-process with: "
            f"[cyan]paraview {state.case_dir}/{state.spec.name}.foam[/]"
        )
    if state.status != "success":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
