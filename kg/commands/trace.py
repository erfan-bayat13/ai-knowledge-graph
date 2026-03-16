# kg/commands/trace.py
# Phase 4b: print citation ancestry tree to terminal
# Usage: kg trace "2106.09685"
#        kg trace "LoRA" --depth 3   (search by title, traverse deeper)

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def trace(
    query: str = typer.Argument(..., help="arXiv ID or paper title fragment"),
    depth: int = typer.Option(2, "--depth", "-d", help="Citation traversal depth (default 2)"),
):
    """Print the citation ancestry tree for a paper in the terminal."""
    from kg.flow.citation_flow import build_citation_tree, detect_divergence, render_tree
    from kg.graph.neo4j_client import Neo4jClient

    # Resolve title → arxiv_id if user passed a title fragment
    arxiv_id = _resolve(query)
    if not arxiv_id:
        console.print(f"\n[red]Paper not found:[/red] {query}\n")
        raise typer.Exit(1)

    console.print(f"\n[dim]Building citation tree for {arxiv_id} (depth={depth})...[/dim]")

    root = build_citation_tree(arxiv_id, depth=depth)
    if root is None:
        console.print(f"\n[yellow]Paper {arxiv_id} not found in graph.[/yellow]\n")
        raise typer.Exit()

    detect_divergence(root)

    tree_str = render_tree(root)
    console.print(f"\n[bold]Citation ancestry:[/bold] {arxiv_id}\n")
    console.print(tree_str)

    # Summary stats
    def count_nodes(n):
        return 1 + sum(count_nodes(c) for c in n.children)
    console.print(f"[dim]{count_nodes(root)} nodes, depth={depth}[/dim]\n")


def _resolve(query: str) -> str:
    """
    If query looks like an arXiv ID (digits.digits), return as-is.
    Otherwise search Neo4j for a paper with matching title.
    """
    import re
    if re.match(r"^\d{4}\.\d{4,5}$", query.strip()):
        return query.strip()

    from kg.graph.neo4j_client import Neo4jClient
    db = Neo4jClient()
    db.connect()
    results = db.run_query(
        "MATCH (p:Paper) WHERE toLower(p.title) CONTAINS toLower($q) "
        "RETURN p.arxiv_id AS arxiv_id ORDER BY p.rank_score DESC LIMIT 1",
        {"q": query}
    )
    db.close()
    return results[0]["arxiv_id"] if results else ""
