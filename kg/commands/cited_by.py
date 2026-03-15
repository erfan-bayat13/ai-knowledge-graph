# kg/commands/cited_by.py
# Phase 1: show which papers in the graph cite a given arXiv ID

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def cited_by(
    arxiv_id: str = typer.Argument(..., help="arXiv ID, e.g. '2106.09685'"),
    limit:    int = typer.Option(20, "--limit", "-n"),
):
    """Show which papers in the graph cite the given paper."""
    db = Neo4jClient()
    db.connect()
    results = db.get_cited_by(arxiv_id, limit=limit)
    db.close()

    if not results:
        console.print(f"\n[yellow]No papers in the graph cite {arxiv_id}[/yellow]\n")
        raise typer.Exit()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Score",  style="green", width=7)
    table.add_column("Date",   style="dim",   width=12)
    table.add_column("Title",  min_width=45)
    table.add_column("arXiv",  style="cyan",  width=16)

    for r in results:
        score = r.get("rank_score")
        table.add_row(
            f"{score:.2f}" if score is not None else "—",
            (r.get("published_date") or "")[:10],
            r.get("title") or "",
            r.get("arxiv_id") or "",
        )

    console.print(f"\n[bold]Papers citing {arxiv_id}[/bold]  ({len(results)} found)\n")
    console.print(table)
