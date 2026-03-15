# kg/commands/flow.py
# Phase 4b: export citation flow JSON and open the Research River visualization in the browser
# Usage: kg flow "2106.09685"

import json
import os
import typer
import webbrowser
from pathlib import Path
from rich.console import Console

app = typer.Typer()
console = Console()

VIZ_DIR = Path(__file__).parent.parent / "visualization"


@app.callback(invoke_without_command=True)
def flow(
    query: str = typer.Argument(..., help="arXiv ID or title fragment"),
    depth: int = typer.Option(2, "--depth", "-d", help="Citation traversal depth"),
):
    """Export citation flow tree and open Research River visualization in the browser."""
    from kg.flow.citation_flow import build_citation_tree, detect_divergence, export_flow_json
    from kg.commands.trace import _resolve

    arxiv_id = _resolve(query)
    if not arxiv_id:
        console.print(f"\n[red]Paper not found:[/red] {query}\n")
        raise typer.Exit(1)

    console.print(f"\n[dim]Building citation flow for {arxiv_id} (depth={depth})...[/dim]")

    root = build_citation_tree(arxiv_id, depth=depth)
    if root is None:
        console.print(f"\n[yellow]Paper {arxiv_id} not found in graph.[/yellow]\n")
        raise typer.Exit()

    detect_divergence(root)
    flow_data = export_flow_json(root)

    # Write river_data.json alongside the HTML file
    out_path = VIZ_DIR / "river_data.json"
    out_path.write_text(json.dumps(flow_data, indent=2))
    console.print(f"  Flow data written: [cyan]{out_path}[/cyan]")

    river_html = VIZ_DIR / "river.html"
    if river_html.exists():
        url = river_html.resolve().as_uri()
        console.print(f"  Opening: [cyan]{url}[/cyan]\n")
        webbrowser.open(url)
    else:
        console.print(f"[yellow]river.html not found at {river_html}[/yellow]\n")
