# kg/commands/cluster.py
# Phase 3: CLI commands for topic clustering and topic exploration
#
# Usage:
#   kg cluster run                              # coarse topics only
#   kg cluster run --sub-cluster               # coarse + fine sub-topics
#   kg cluster run --min-size 5 --sub-cluster  # more granular
#   kg cluster topic "LLM Agents"
#   kg cluster topic "Reinforcement Learning > Offline RL Methods"

import typer
from rich.console import Console
from rich.table import Table
from rich import box
import logging

app = typer.Typer()
console = Console()


@app.command("run")
def run_clustering(
    min_size: int = typer.Option(
        10, "--min-size",
        help="HDBSCAN min_cluster_size for coarse pass (lower = more clusters)"
    ),
    sub_cluster: bool = typer.Option(
        False, "--sub-cluster",
        help="Run a second fine-grained clustering pass within each coarse topic"
    ),
    sub_min_size: int = typer.Option(
        3, "--sub-min-size",
        help="HDBSCAN min_cluster_size for the sub-cluster pass (default 3)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print results without writing to Neo4j"
    ),
):
    """Run UMAP + HDBSCAN clustering and write Topic nodes to Neo4j.

    \b
    Examples:
      kg cluster run                             # broad topics
      kg cluster run --sub-cluster               # broad + detailed sub-topics
      kg cluster run --min-size 5 --sub-cluster  # more granular, with sub-topics
    """
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)-8s %(message)s")
    from kg.clustering.cluster import run_clustering as _run

    mode = "coarse + sub-cluster" if sub_cluster else "coarse only"
    console.print(f"\n[bold]Running topic clustering[/bold] ({mode}, min-size={min_size})\n")

    result = _run(
        min_cluster_size=min_size,
        sub_cluster=sub_cluster,
        sub_min_size=sub_min_size,
        dry_run=dry_run,
    )

    if not result:
        console.print("[red]Clustering failed or no embeddings found.[/red]\n")
        raise typer.Exit(1)

    console.print(f"  Papers clustered : [cyan]{result.get('n_papers')}[/cyan]")
    console.print(f"  Topics found     : [cyan]{result.get('n_clusters')}[/cyan]")
    console.print(f"  Noise points     : [dim]{result.get('n_noise')}[/dim]")

    names = result.get("cluster_names", {})
    if names:
        console.print("\n  [bold]Topics discovered:[/bold]")
        # Sort: coarse topics first, then sub-topics (contain " > ")
        coarse = sorted([(k, v) for k, v in names.items() if " > " not in v], key=lambda x: x[1])
        fine   = sorted([(k, v) for k, v in names.items() if " > " in v],   key=lambda x: x[1])

        for _, name in coarse:
            console.print(f"    [bold]{name}[/bold]")
        if fine:
            console.print()
            for _, name in fine:
                console.print(f"      [dim]↳[/dim] {name}")

    console.print()


@app.command("topic")
def show_topic(
    name:  str = typer.Argument(..., help="Topic name, e.g. 'LLM Agents' or 'RL > Offline RL Methods'"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max papers to show"),
):
    """Show papers in a specific topic cluster."""
    from kg.graph.neo4j_client import Neo4jClient

    db = Neo4jClient()
    db.connect()

    papers = db.run_query("""
        MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($name)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.published_date AS date, p.rank_score AS rank_score,
               t.name AS topic
        ORDER BY p.rank_score DESC
        LIMIT $limit
    """, {"name": name, "limit": limit})

    db.close()

    if not papers:
        console.print(f"\n[yellow]No papers found for topic '{name}'.[/yellow]")
        console.print("[dim]Tip: use a partial match, e.g. kg cluster topic 'offline rl'[/dim]\n")
        return

    # Show matched topic name
    matched_topic = papers[0].get("topic", name)
    console.print(f"\n[bold]{matched_topic}[/bold]  ({len(papers)} papers)\n")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Score",  style="green", width=8)
    t.add_column("Date",   style="dim",   width=12)
    t.add_column("Title",  min_width=50)
    t.add_column("arXiv",  style="cyan",  width=14)

    for r in papers:
        score = f"{r.get('rank_score') or 0:.2f}"
        t.add_row(score, (r.get("date") or "")[:10], r.get("title") or "", r.get("arxiv_id") or "")

    console.print(t)
    console.print()