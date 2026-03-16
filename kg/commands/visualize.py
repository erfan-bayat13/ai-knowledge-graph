# kg/commands/visualize.py
# Phase 4a: export UMAP 2D coordinates + graph edges to JSON, open scatter plot in browser
# Serves the visualization over a local HTTP server to avoid browser CORS restrictions
# Usage: kg visualize

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()

VIZ_DIR  = Path(__file__).parent.parent / "visualization"
PORT     = 8765


@app.callback(invoke_without_command=True)
def visualize(
    port: int = typer.Option(PORT, "--port", "-p", help="Local server port (default 8765)"),
):
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

    # Fetch CITES edges between papers in the graph (empty until S2 key restored)
    cites_edges = db.run_query("""
        MATCH (a:Paper)-[:CITES]->(b:Paper)
        WHERE a.umap_x IS NOT NULL AND b.umap_x IS NOT NULL
        RETURN a.arxiv_id AS source, b.arxiv_id AS target, 'cites' AS type
        LIMIT 5000
    """)

    db.close()

    if not papers_raw:
        console.print("\n[yellow]No UMAP coordinates found. Run: kg cluster run[/yellow]\n")
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
    if not graph_html.exists():
        console.print(f"[red]graph.html not found at {graph_html}[/red]\n")
        raise typer.Exit(1)

    # ── Spin up a local HTTP server in a background thread ────────────────────
    # Needed because browsers block fetch() on file:// URLs (CORS).
    # SimpleHTTPRequestHandler serves from VIZ_DIR, so graph_data.json is reachable.

    class _QuietHandler(SimpleHTTPRequestHandler):
        """Suppress per-request log lines in the terminal."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(VIZ_DIR), **kwargs)

        def log_message(self, format, *args):  # noqa: A002
            pass  # silence access logs

    server = HTTPServer(("127.0.0.1", port), _QuietHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/graph.html"
    console.print(f"  Serving at: [cyan]{url}[/cyan]")
    console.print(f"  [dim]Press Ctrl+C to stop the server.[/dim]\n")
    webbrowser.open(url)

    try:
        # Keep the main thread alive so the server stays up while the browser loads.
        # daemon=True means it dies automatically when the process exits.
        thread.join()
    except KeyboardInterrupt:
        console.print("\n  [dim]Server stopped.[/dim]\n")
        server.shutdown()