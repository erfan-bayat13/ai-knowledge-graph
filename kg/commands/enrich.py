# kg/commands/enrich.py
# Phase 1: CLI command to run Semantic Scholar + OpenAlex enrichment pipeline

import typer
import logging
from rich.console import Console

app = typer.Typer()
console = Console()


@app.callback(invoke_without_command=True)
def enrich(
    limit:    int   = typer.Option(100,  "--limit",  "-n", help="Max papers to enrich per run"),
    dry_run:  bool  = typer.Option(False, "--dry-run",      help="Print what would be written, no DB writes"),
    rate:     float = typer.Option(0.5,  "--rate",         help="Seconds between API calls"),
):
    """Enrich papers with citation data from Semantic Scholar + OpenAlex."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    from kg.enrichment.runner import run_enrichment
    console.print(f"\n[bold]Running enrichment[/bold] (limit={limit}, dry_run={dry_run})\n")
    run_enrichment(limit=limit, dry_run=dry_run, rate_limit_s=rate)
