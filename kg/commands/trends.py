import typer
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime, timedelta

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer(help="Show trending methods and topics")
console = Console()


@app.callback(invoke_without_command=True)
def trends(
    days: int = typer.Option(30, "--days", "-d", help="Lookback window in days"),
    limit: int = typer.Option(15, "--limit", "-n", help="Number of results"),
):
    """Show which methods and topics are trending in recent papers."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    db = Neo4jClient()
    db.connect()

    # Trending proposed methods
    methods = db.run_query("""
        MATCH (p:Paper)-[:PROPOSES]->(m:Method)
        WHERE p.published_date >= $since
        RETURN m.name AS name, count(p) AS count
        ORDER BY count DESC
        LIMIT $limit
    """, {"since": since, "limit": limit})

    # Trending datasets
    datasets = db.run_query("""
        MATCH (p:Paper)-[:EVALUATED_ON]->(d:Dataset)
        WHERE p.published_date >= $since
        RETURN d.name AS name, count(p) AS count
        ORDER BY count DESC
        LIMIT $limit
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
        console.print(f"[yellow]No enriched data yet. Run the enrichment pipeline first.[/yellow]\n")