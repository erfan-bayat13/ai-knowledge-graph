import typer
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime, timedelta

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer(
    name="kg",
    help="AI research knowledge graph — search papers, explore trends.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search term, e.g. 'LoRA'"),
    limit: int = typer.Option(10, "--limit", "-n"),
):
    """Search papers by keyword (matches title + abstract)."""
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
    table.add_column("arXiv", style="cyan", width=16)

    for r in results:
        table.add_row(
            (r.get("published_date") or "")[:10],
            r.get("title") or "",
            r.get("arxiv_id") or "",
        )

    console.print(f"\n[bold]Results for:[/bold] '{query}'  ({len(results)} papers)\n")
    console.print(table)


@app.command()
def trends(
    days: int = typer.Option(30, "--days", "-d"),
    limit: int = typer.Option(15, "--limit", "-n"),
):
    """Show trending methods and datasets in recent papers."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    db = Neo4jClient()
    db.connect()

    methods = db.run_query("""
        MATCH (p:Paper)-[:PROPOSES]->(m:Method)
        WHERE p.published_date >= $since
        RETURN m.name AS name, count(p) AS count
        ORDER BY count DESC LIMIT $limit
    """, {"since": since, "limit": limit})

    datasets = db.run_query("""
        MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset)
        WHERE p.published_date >= $since
        RETURN d.name AS name, count(p) AS count
        ORDER BY count DESC LIMIT $limit
    """, {"since": since, "limit": limit})

    db.close()

    console.print(f"\n[bold]Trending (last {days} days)[/bold]\n")

    if methods:
        console.print("[bold cyan]Methods being proposed[/bold cyan]")
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("Method", min_width=30)
        t.add_column("Papers", style="dim", width=8)
        for r in methods:
            t.add_row(r["name"], str(r["count"]))
        console.print(t)

    if datasets:
        console.print("[bold cyan]Datasets / benchmarks[/bold cyan]")
        t = Table(box=box.SIMPLE, show_header=False)
        t.add_column("Dataset", min_width=30)
        t.add_column("Papers", style="dim", width=8)
        for r in datasets:
            t.add_row(r["name"], str(r["count"]))
        console.print(t)

    if not methods and not datasets:
        console.print("[yellow]No enriched data yet. Run the enrichment pipeline first.[/yellow]\n")


@app.command()
def status():
    """Show how many papers, methods, and edges are in the graph."""
    db = Neo4jClient()
    db.connect()
    papers   = db.run_query("MATCH (p:Paper)   RETURN count(p) AS n")[0]["n"]
    methods  = db.run_query("MATCH (m:Method)  RETURN count(m) AS n")[0]["n"]
    datasets = db.run_query("MATCH (d:Dataset) RETURN count(d) AS n")[0]["n"]
    edges    = db.run_query("MATCH ()-[r:PROPOSES|EVALUATED_ON]->() RETURN count(r) AS n")[0]["n"]
    db.close()

    console.print("\n[bold]Graph status[/bold]\n")
    console.print(f"  Papers:   [cyan]{papers}[/cyan]")
    console.print(f"  Methods:  [cyan]{methods}[/cyan]")
    console.print(f"  Datasets: [cyan]{datasets}[/cyan]")
    console.print(f"  Edges:    [cyan]{edges}[/cyan]\n")


if __name__ == "__main__":
    app()