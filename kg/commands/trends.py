# CHANGED (Phase 3): rewrote trends to use Topic nodes + trend_score instead of Method/Dataset nodes
# Old queries for PROPOSES/EVALUATED_ON replaced with BELONGS_TO Topic queries

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime, timedelta

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer(help="Show trending research topics")
console = Console()

_ARROWS = {0: "", 1: "↑", 2: "↑↑", 3: "↑↑↑"}


@app.callback(invoke_without_command=True)
def trends(
    days:  int = typer.Option(7,  "--days", "-d", help="Lookback window in days (default 7)"),
    limit: int = typer.Option(15, "--limit", "-n", help="Number of topics to show"),
):
    """Show which research topics are trending (requires clustering to have run)."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    db = Neo4jClient()
    db.connect()

    # Topics ranked by trend_score = new_papers × avg_citation_velocity
    topics = db.run_query("""
        MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic)
        WHERE p.published_date >= $since
        RETURN t.name AS name, count(p) AS new_papers, t.trend_score AS trend_score,
               t.paper_count AS total_papers
        ORDER BY trend_score DESC NULLS LAST, new_papers DESC
        LIMIT $limit
    """, {"since": since, "limit": limit})

    db.close()

    console.print(f"\n[bold]Trending topics[/bold] (last {days} days)\n")

    if not topics:
        console.print("[yellow]No topic data yet. Run: kg cluster --run[/yellow]\n")
        return

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("#",            style="dim",   width=4)
    t.add_column("Topic",        min_width=35)
    t.add_column("New papers",   style="cyan",  width=12)
    t.add_column("Total papers", style="dim",   width=13)
    t.add_column("Trend",        style="green", width=8)

    for i, r in enumerate(topics, 1):
        score = r.get("trend_score") or 0
        arrow_idx = min(int(score / 0.5), 3)  # rough bucketing; tune with real data
        t.add_row(
            str(i),
            r.get("name") or "",
            str(r.get("new_papers", 0)),
            str(r.get("total_papers", "—")),
            _ARROWS.get(arrow_idx, "↑↑↑"),
        )

    console.print(t)
    console.print()