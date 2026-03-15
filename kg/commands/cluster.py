# kg/commands/cluster.py
# Phase 3: CLI commands for topic clustering and topic exploration

import typer
from rich.console import Console
from rich.table import Table
from rich import box
import logging

app = typer.Typer()
console = Console()


@app.command("run")
def run_clustering(
    min_cluster_size: int  = typer.Option(10,    "--min-size", help="HDBSCAN min_cluster_size"),
    dry_run:          bool = typer.Option(False, "--dry-run",  help="Print results, no DB writes"),
):
    """Run UMAP + HDBSCAN clustering and write Topic nodes to Neo4j."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    from kg.clustering.cluster import run_clustering as _run
    console.print("\n[bold]Running topic clustering...[/bold]\n")
    result = _run(min_cluster_size=min_cluster_size, dry_run=dry_run)

    if not result:
        console.print("[red]Clustering failed or no embeddings found.[/red]\n")
        raise typer.Exit(1)

    console.print(f"\n  Papers clustered: [cyan]{result.get('n_papers')}[/cyan]")
    console.print(f"  Clusters found:   [cyan]{result.get('n_clusters')}[/cyan]")
    console.print(f"  Noise points:     [dim]{result.get('n_noise')}[/dim]")

    names = result.get("cluster_names", {})
    if names:
        console.print("\n  [bold]Topics discovered:[/bold]")
        for cid, name in sorted(names.items()):
            console.print(f"    [{cid}] {name}")
    console.print()


@app.command("topic")
def show_topic(
    name:  str = typer.Argument(..., help="Topic name, e.g. 'LLM Agents'"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """Show papers in a specific topic cluster."""
    from kg.graph.neo4j_client import Neo4jClient

    db = Neo4jClient()
    db.connect()
    results = db.run_query("""
        MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($name)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.published_date AS published_date, p.rank_score AS rank_score,
               t.name AS topic
        ORDER BY p.rank_score DESC NULLS LAST
        LIMIT $limit
    """, {"name": name, "limit": limit})
    db.close()

    if not results:
        console.print(f"\n[yellow]No papers found for topic: '{name}'[/yellow]\n")
        raise typer.Exit()

    topic_name = results[0].get("topic", name)
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

    console.print(f"\n[bold]Topic:[/bold] {topic_name}  ({len(results)} papers)\n")
    console.print(table)
