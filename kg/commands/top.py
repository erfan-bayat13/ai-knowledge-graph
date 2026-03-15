# kg/commands/top.py
# Phase 1: show top-ranked papers by rank_score
# rank_score = log(citations+1) + 1.5*velocity + 0.8*recency + 0.3*author_influence

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def top(n: int = typer.Option(20, "--n", "-n", help="Number of papers to show")):
    """Show top papers by rank_score (requires enrichment to have run)."""
    db = Neo4jClient()
    db.connect()
    results = db.get_top_papers(limit=n)
    db.close()

    if not results:
        console.print("\n[yellow]No ranked papers yet. Run: kg enrich[/yellow]\n")
        raise typer.Exit()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#",        style="dim",  width=4)
    table.add_column("Score",    style="green", width=7)
    table.add_column("Citations", style="cyan", width=10)
    table.add_column("Date",     style="dim",  width=12)
    table.add_column("Title",    min_width=45)
    table.add_column("arXiv",    style="cyan", width=16)

    for i, r in enumerate(results, 1):
        score = r.get("rank_score")
        score_str = f"{score:.2f}" if score is not None else "—"
        cit = r.get("citation_count")
        cit_str = str(cit) if cit is not None else "—"
        table.add_row(
            str(i),
            score_str,
            cit_str,
            (r.get("published_date") or "")[:10],
            r.get("title") or "",
            r.get("arxiv_id") or "",
        )

    console.print(f"\n[bold]Top {len(results)} papers by rank score[/bold]\n")
    console.print(table)
