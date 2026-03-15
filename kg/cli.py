# kg/cli.py
# CHANGED: rewrote to include all commands from Phases 1-4
# CHANGED (Phase 1): added enrich, top, cited-by
# CHANGED (Phase 2): added embed (run, search, similar)
# CHANGED (Phase 3): added cluster (run, topic), updated trends to topic-based
# CHANGED (Phase 4): added trace, flow, visualize
# CHANGED: status command updated to reflect new node types (Topic, Institution, CITES edges)

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime, timedelta

from kg.graph.neo4j_client import Neo4jClient

app = typer.Typer(
    name="kg",
    help="AI Research Intelligence Platform — discover, explore, and map research papers.",
    no_args_is_help=True,
)

console = Console()

# ── Import sub-apps ────────────────────────────────────────────────────────────

from kg.commands.search    import app as search_app
from kg.commands.trends    import app as trends_app
from kg.commands.top       import app as top_app
from kg.commands.enrich    import app as enrich_app
from kg.commands.cited_by  import app as cited_by_app
from kg.commands.embed     import app as embed_app
from kg.commands.cluster   import app as cluster_app
from kg.commands.trace     import app as trace_app
from kg.commands.flow      import app as flow_app
from kg.commands.visualize import app as visualize_app

app.add_typer(search_app,    name="search",    help="Search papers by keyword")
app.add_typer(trends_app,    name="trends",    help="Show trending research topics")
app.add_typer(top_app,       name="top",       help="Top papers by rank score")
app.add_typer(enrich_app,    name="enrich",    help="Enrich papers via Semantic Scholar + OpenAlex")
app.add_typer(cited_by_app,  name="cited-by",  help="Show papers that cite a given paper")
app.add_typer(embed_app,     name="embed",     help="SPECTER2 paper-level embeddings + semantic search")
app.add_typer(cluster_app,   name="cluster",   help="Topic clustering with UMAP + HDBSCAN")
app.add_typer(trace_app,     name="trace",     help="Print citation ancestry tree in terminal")
app.add_typer(flow_app,      name="flow",      help="Open Research River citation flow visualization")
app.add_typer(visualize_app, name="visualize", help="Export UMAP scatter data and open landscape view")


# ── Status command (standalone, not a sub-app) ─────────────────────────────────

@app.command()
def status():
    """Show graph counts: papers, authors, topics, institutions, and edges."""
    db = Neo4jClient()
    db.connect()

    papers       = db.run_query("MATCH (p:Paper)       RETURN count(p) AS n")[0]["n"]
    authors      = db.run_query("MATCH (a:Author)      RETURN count(a) AS n")[0]["n"]
    topics       = db.run_query("MATCH (t:Topic)       RETURN count(t) AS n")[0]["n"]
    institutions = db.run_query("MATCH (i:Institution) RETURN count(i) AS n")[0]["n"]
    cites_edges  = db.run_query("MATCH ()-[r:CITES]->()       RETURN count(r) AS n")[0]["n"]
    belongs_to   = db.run_query("MATCH ()-[r:BELONGS_TO]->()  RETURN count(r) AS n")[0]["n"]
    enriched     = db.run_query("MATCH (p:Paper) WHERE p.rank_score IS NOT NULL RETURN count(p) AS n")[0]["n"]
    embedded     = db.run_query("MATCH (p:Paper) WHERE p.embedding  IS NOT NULL RETURN count(p) AS n")[0]["n"]

    db.close()

    console.print("\n[bold]Graph status[/bold]\n")
    console.print(f"  Papers:       [cyan]{papers}[/cyan]  (enriched: {enriched}, embedded: {embedded})")
    console.print(f"  Authors:      [cyan]{authors}[/cyan]")
    console.print(f"  Topics:       [cyan]{topics}[/cyan]")
    console.print(f"  Institutions: [cyan]{institutions}[/cyan]")
    console.print(f"  CITES edges:  [cyan]{cites_edges}[/cyan]")
    console.print(f"  BELONGS_TO:   [cyan]{belongs_to}[/cyan]\n")


if __name__ == "__main__":
    app()
