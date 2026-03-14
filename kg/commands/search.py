# kg/commands/search.py

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer()
console = Console()

@app.command()
def search(
    query: str = typer.Argument(..., help="Search term, e.g. 'transformers'"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
):
    """Search papers by keyword (title + abstract)."""
    db = Neo4jClient()
    db.connect()
    results = db.search_papers(query, limit=limit)
    db.close()

    if not results:
        console.print(f"\n[yellow]No papers found for:[/yellow] {query}\n")
        raise typer.Exit()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Date", style="dim", width=12)
    table.add_column("Title", min_width=40)
    table.add_column("arXiv ID", style="cyan", width=16)

    for r in results:
        table.add_row(
            r.get("published_date", "")[:10],
            r.get("title", ""),
            r.get("arxiv_id", ""),
        )

    console.print(f"\n[bold]Results for:[/bold] {query}  ({len(results)} papers)\n")
    console.print(table)