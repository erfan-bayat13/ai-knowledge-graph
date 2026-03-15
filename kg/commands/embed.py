# kg/commands/embed.py
# Phase 2: CLI commands for paper-level SPECTER2 embeddings, semantic search, and similar papers

import typer
from rich.console import Console
from rich.table import Table
from rich import box
import logging

app = typer.Typer()
console = Console()


@app.command()
def run(
    limit:   int  = typer.Option(500,  "--limit",  "-n", help="Max papers to embed"),
    dry_run: bool = typer.Option(False, "--dry-run",      help="Print only, no writes"),
):
    """Embed papers without vectors using SPECTER2 (paper-level)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    from kg.nlp.embedder import run_embedding_pipeline
    console.print(f"\n[bold]Embedding papers[/bold] (limit={limit})\n")
    result = run_embedding_pipeline(limit=limit, dry_run=dry_run)
    console.print(f"\n  Embedded: [cyan]{result['papers_embedded']}[/cyan]  Failed: [red]{result['failed']}[/red]\n")


@app.command()
def search(
    query: str = typer.Argument(..., help="Semantic query, e.g. 'mechanistic interpretability transformers'"),
    limit: int = typer.Option(10, "--limit", "-n"),
):
    """Semantic search: find papers by meaning, not just keyword."""
    from kg.nlp.embedder import embed_query
    from kg.graph.neo4j_client import Neo4jClient

    console.print(f"\n[dim]Embedding query...[/dim]")
    query_vec = embed_query(query)
    if query_vec is None:
        console.print("[red]Failed to embed query.[/red]")
        raise typer.Exit(1)

    db = Neo4jClient()
    db.connect()
    results = db.vector_search_papers(query_vec, top_k=limit)
    db.close()

    if not results:
        console.print(f"\n[yellow]No semantic matches found. Run: kg embed run[/yellow]\n")
        raise typer.Exit()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Score",  style="green", width=7)
    table.add_column("Date",   style="dim",   width=12)
    table.add_column("Title",  min_width=45)
    table.add_column("arXiv",  style="cyan",  width=16)

    for r in results:
        table.add_row(
            f"{r.get('score', 0):.3f}",
            (r.get("published_date") or "")[:10],
            r.get("title") or "",
            r.get("arxiv_id") or "",
        )

    console.print(f"\n[bold]Semantic search:[/bold] '{query}'\n")
    console.print(table)


@app.command()
def similar(
    arxiv_id: str = typer.Argument(..., help="arXiv ID of the seed paper"),
    limit:    int = typer.Option(10, "--limit", "-n"),
):
    """Find papers semantically similar to a given paper."""
    from kg.graph.neo4j_client import Neo4jClient

    db = Neo4jClient()
    db.connect()

    # Fetch the paper's embedding
    rows = db.run_query(
        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.embedding AS emb, p.title AS title",
        {"id": arxiv_id}
    )
    if not rows or rows[0]["emb"] is None:
        console.print(f"\n[yellow]Paper {arxiv_id} has no embedding yet. Run: kg embed run[/yellow]\n")
        db.close()
        raise typer.Exit()

    emb   = [float(x) for x in rows[0]["emb"]]
    title = rows[0]["title"]

    results = db.vector_search_papers(emb, top_k=limit + 1)  # +1 to exclude self
    db.close()

    # Filter out the seed paper itself
    results = [r for r in results if r.get("arxiv_id") != arxiv_id][:limit]

    if not results:
        console.print(f"\n[yellow]No similar papers found.[/yellow]\n")
        raise typer.Exit()

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Score",  style="green", width=7)
    table.add_column("Date",   style="dim",   width=12)
    table.add_column("Title",  min_width=45)
    table.add_column("arXiv",  style="cyan",  width=16)

    for r in results:
        table.add_row(
            f"{r.get('score', 0):.3f}",
            (r.get("published_date") or "")[:10],
            r.get("title") or "",
            r.get("arxiv_id") or "",
        )

    console.print(f"\n[bold]Papers similar to:[/bold] {arxiv_id} — {title}\n")
    console.print(table)
