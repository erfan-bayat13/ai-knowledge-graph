# kg/commands/visualize.py
# Phase 4a: export UMAP 2D coordinates + graph edges to JSON, open scatter plot in browser
# Usage: kg visualize

import json
import webbrowser
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

VIZ_DIR = Path(__file__).parent.parent / "visualization"


@app.callback(invoke_without_command=True)
def visualize():
    """Export UMAP scatter data to JSON and open the landscape visualization in the browser."""
    from kg.graph.neo4j_client import Neo4jClient

    db = Neo4jClient()
    db.connect()

    # Fetch all papers with UMAP coordinates
    papers_raw = db.run_query("""
        MATCH (p:Paper)
        WHERE p.umap_x IS NOT NULL AND p.umap_y IS NOT NULL
        OPTIONAL MATCH (p)-[:BELONGS_TO]->(t:Topic)
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.published_date AS date, p.rank_score AS rank_score,
               p.umap_x AS x, p.umap_y AS y,
               p.cluster_id AS cluster_id,
               t.name AS topic
    """)

    # Fetch CITES edges between papers in the graph
    cites_edges = db.run_query("""
        MATCH (a:Paper)-[:CITES]->(b:Paper)
        WHERE a.umap_x IS NOT NULL AND b.umap_x IS NOT NULL
        RETURN a.arxiv_id AS source, b.arxiv_id AS target, 'cites' AS type
        LIMIT 5000
    """)

    db.close()

    if not papers_raw:
        console.print("\n[yellow]No UMAP coordinates found. Run: kg cluster --run first.[/yellow]\n")
        raise typer.Exit()

    graph_data = {
        "papers": [
            {
                "arxiv_id":   r["arxiv_id"],
                "title":      r["title"] or "",
                "date":       r["date"] or "",
                "rank_score": round(r["rank_score"] or 0, 4),
                "x":          r["x"],
                "y":          r["y"],
                "cluster_id": r["cluster_id"],
                "topic":      r["topic"] or "Unknown",
            }
            for r in papers_raw
        ],
        "edges": [
            {"source": e["source"], "target": e["target"], "type": e["type"]}
            for e in cites_edges
        ],
    }

    out_path = VIZ_DIR / "graph_data.json"
    out_path.write_text(json.dumps(graph_data, indent=2))
    console.print(f"\n  Graph data written: [cyan]{out_path}[/cyan]")
    console.print(f"  Papers: [cyan]{len(graph_data['papers'])}[/cyan]  "
                  f"Edges: [cyan]{len(graph_data['edges'])}[/cyan]")

    graph_html = VIZ_DIR / "graph.html"
    if graph_html.exists():
        url = graph_html.resolve().as_uri()
        console.print(f"  Opening: [cyan]{url}[/cyan]\n")
        webbrowser.open(url)
    else:
        console.print(f"[yellow]graph.html not found at {graph_html}[/yellow]\n")
