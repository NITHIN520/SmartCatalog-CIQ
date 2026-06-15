"""
Category Tagging Automation – Main CLI
========================================
Commands:
  seed        Bootstrap the dataset from built-in or custom seed data
  predict     Predict category for one or more sub_categories
  batch       Predict from a CSV file
  correct     Apply a human correction
  retrain     Force retrain the ML model
  stats       Show dataset statistics
  status      Show system readiness
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="category-tagger",
    help="Self-learning product category tagging system.",
    add_completion=False,
)
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Built-in seed data (the example from the user) ────────────────────────────
DEFAULT_SEED_DATA = [
    {"sub_category": "Craft Beer", "category": "Beer"},
    {"sub_category": "Variety Packs", "category": "Beer"},
    {"sub_category": "Classic Lager", "category": "Beer"},
    {"sub_category": "Light Lager", "category": "Beer"},
    {"sub_category": "Cider", "category": "Hard Beverage"},
    {"sub_category": "Hard Seltzers", "category": "Hard Beverage"},
    {"sub_category": "Ready to Drink", "category": "Hard Beverage"},
    {"sub_category": "Flavored Hard Beverages", "category": "Hard Beverage"},
    {"sub_category": "Non-alcoholic Beer", "category": "Non-alcoholic Beer"},
]


# ── seed ─────────────────────────────────────────────────────────────────────

@app.command()
def seed(
    json_file: Optional[Path] = typer.Option(
        None,
        "--file", "-f",
        help="Path to a JSON file with [{sub_category, category}] records. "
             "Uses built-in beverage data if omitted.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Bootstrap (or expand) the dataset and train the first ML model."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if json_file:
        records = json.loads(Path(json_file).read_text())
        console.print(f"[cyan]Seeding from {json_file} ({len(records)} records) …[/cyan]")
    else:
        records = DEFAULT_SEED_DATA
        console.print(f"[cyan]Seeding with {len(records)} built-in records …[/cyan]")

    from src.pipeline.trainer import bootstrap
    metrics = bootstrap(records)

    console.print(f"[green]✓ Seeded & trained.[/green]  {metrics}")


# ── predict ───────────────────────────────────────────────────────────────────

@app.command()
def predict(
    sub_categories: list[str] = typer.Argument(..., help="One or more sub-category strings."),
    no_store: bool = typer.Option(False, "--no-store", help="Don't save prediction to dataset."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Predict category for one or more sub-categories."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.pipeline.predictor import predict as do_predict

    table = Table(title="Predictions", show_lines=True)
    table.add_column("Product / Input", style="cyan")
    table.add_column("Category", style="green bold")
    table.add_column("Sub-Category", style="blue bold")
    table.add_column("Cat Conf", justify="right")
    table.add_column("Sub Conf", justify="right")
    table.add_column("Source", style="magenta")

    for sc in sub_categories:
        result = do_predict(sc, store_result=not no_store)
        cat_conf = f"{result.confidence:.1%}"
        sub_conf = f"{result.sub_confidence:.1%}" if result.sub_confidence else "-"
        if result.low_confidence:
            cat_conf = f"[yellow]{cat_conf}[/yellow]"
        if result.sub_confidence < 0.60:
            sub_conf = f"[yellow]{sub_conf}[/yellow]"
        table.add_row(
            result.sub_category,
            result.category,
            result.predicted_sub_category or "-",
            cat_conf,
            sub_conf,
            f"{result.source} / {result.sub_source}",
        )

    console.print(table)


# ── batch ─────────────────────────────────────────────────────────────────────

@app.command()
def batch(
    input_file: Path = typer.Argument(..., help="CSV with a 'sub_category' column."),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o"),
    no_store: bool = typer.Option(False, "--no-store"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Batch-predict from a CSV file. Results are written row-by-row as each product is processed."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    import csv
    import pandas as pd
    from src.pipeline.predictor import predict

    # In-batch cache for near-duplicate products (avoids redundant LLM calls)
    from rapidfuzz import process as fz_process, fuzz
    from src.pipeline.text_preprocessor import clean_product_name
    from src.pipeline.predictor import _enrich_with_sub_category, PredictionResult
    from src.data import dataset_manager

    df_in = pd.read_csv(input_file, dtype=str)

    # Accept either 'sub_category' or 'product_name' as the input column
    if "sub_category" in df_in.columns:
        input_col = "sub_category"
    elif "product_name" in df_in.columns:
        input_col = "product_name"
    else:
        console.print("[red]CSV must have a 'product_name' or 'sub_category' column.[/red]")
        raise typer.Exit(1)

    product_names = df_in[input_col].fillna("").tolist()
    total = len(product_names)
    console.print(f"[cyan]Processing {total} rows — writing results as each completes …[/cyan]")

    # Carry through any extra columns from input (e.g. product_id) to output
    extra_cols = [c for c in df_in.columns if c != input_col]
    extra_rows = df_in[extra_cols].fillna("").values.tolist() if extra_cols else [[] for _ in product_names]

    # Output column order: extra input cols first, then prediction cols
    out_columns = extra_cols + [
        input_col,
        "predicted_category",
        "predicted_sub_category",
        "category_confidence",
        "sub_category_confidence",
        "category_source",
        "sub_category_source",
        "needs_review",   # YES if confidence is low or source is llm/fallback
    ]

    # Thresholds below which a prediction is flagged for human review
    _REVIEW_CAT_THRESHOLD = 0.75
    _REVIEW_SUB_THRESHOLD = 0.50   # sub-category confidence below this → review
    _REVIEW_SOURCES = {"llm", "fallback", "batch_cache"}

    # ── Open output file and write header immediately ─────────────────────────
    out_fp = open(output_file, "w", newline="", encoding="utf-8") if output_file else None
    writer = csv.DictWriter(out_fp, fieldnames=out_columns) if out_fp else None
    if writer:
        writer.writeheader()
        out_fp.flush()

    batch_cache: dict[str, PredictionResult] = {}

    try:
        for idx, (pname, extra_vals) in enumerate(zip(product_names, extra_rows), 1):
            pname = pname.strip()

            # Check in-batch cache for near-duplicates
            result = None
            if batch_cache:
                best = fz_process.extractOne(
                    pname, list(batch_cache.keys()), scorer=fuzz.token_sort_ratio
                )
                if best and best[1] >= 92:
                    cached = batch_cache[best[0]]
                    result = _enrich_with_sub_category(PredictionResult(
                        sub_category=pname,
                        category=cached.category,
                        confidence=cached.confidence,
                        source="batch_cache",
                        steps_tried=["batch_cache"],
                    ), clean_product_name(pname))
                    if not no_store:
                        dataset_manager.append_prediction(
                            pname, result.category, "batch_cache", result.confidence,
                            predicted_sub_category=result.predicted_sub_category,
                            sub_confidence=result.sub_confidence,
                        )

            if result is None:
                result = predict(pname, store_result=not no_store)
                batch_cache[pname] = result

            # ── Write this row to CSV immediately ─────────────────────────────
            needs_review = (
                result.confidence < _REVIEW_CAT_THRESHOLD
                or result.sub_confidence < _REVIEW_SUB_THRESHOLD
                or result.source in _REVIEW_SOURCES
            )

            row_data = {col: val for col, val in zip(extra_cols, extra_vals)}
            row_data.update({
                input_col: pname,
                "predicted_category": result.category,
                "predicted_sub_category": result.predicted_sub_category,
                "category_confidence": round(result.confidence, 4),
                "sub_category_confidence": round(result.sub_confidence, 4),
                "category_source": result.source,
                "sub_category_source": result.sub_source,
                "needs_review": "YES" if needs_review else "NO",
            })

            if writer:
                writer.writerow(row_data)
                out_fp.flush()   # flush to disk so file updates in real time
            else:
                # No output file: print each row to console as it's done
                console.print(
                    f"[bold]{idx}/{total}[/bold] {pname[:60]:<60} "
                    f"→ [green]{result.category}[/green] / {result.predicted_sub_category} "
                    f"([dim]{result.source}[/dim])"
                )

            console.print(
                f"  [[cyan]{idx}/{total}[/cyan]] {pname[:70]} "
                f"→ [green]{result.category}[/green] / {result.predicted_sub_category}"
            )

    finally:
        if out_fp:
            out_fp.close()

    if output_file:
        console.print(f"\n[green]✓ All {total} results saved to {output_file}[/green]")


# ── correct ───────────────────────────────────────────────────────────────────

@app.command()
def correct(
    sub_category: str = typer.Argument(..., help="The sub-category to correct."),
    correct_category: str = typer.Argument(..., help="The correct category label."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Apply a human correction and immediately retrain the model."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.pipeline.trainer import apply_correction
    result = apply_correction(sub_category, correct_category)

    if result.get("correction_applied"):
        console.print(
            f"[green]✓ Correction applied:[/green] "
            f"'{sub_category}' → '{correct_category}'. Model retrained."
        )
    else:
        console.print("[yellow]Correction not applied (sub_category not found).[/yellow]")


# ── retrain ───────────────────────────────────────────────────────────────────

@app.command()
def retrain(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Force retrain the ML model from the full dataset."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from src.pipeline.trainer import force_retrain
    metrics = force_retrain()
    console.print(f"[green]✓ Retrained.[/green]  {metrics}")


# ── stats ─────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show dataset and model statistics."""
    from src.data.dataset_manager import dataset_stats, load
    from src.ml.classifier import is_trained

    s = dataset_stats()
    df = load()

    table = Table(title="System Statistics", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green bold")

    table.add_row("Total rows in dataset", str(s["total_rows"]))
    table.add_row("Unique categories", str(s["unique_categories"]))
    table.add_row("Human corrections", str(s["human_corrections"]))
    table.add_row("ML model trained", "[green]Yes[/green]" if is_trained() else "[red]No[/red]")

    for src, cnt in s.get("source_counts", {}).items():
        table.add_row(f"  source={src}", str(cnt))

    console.print(table)

    if not df.empty:
        console.print("\n[bold]Category distribution:[/bold]")
        dist_table = Table(show_lines=False)
        dist_table.add_column("Category", style="cyan")
        dist_table.add_column("Count", justify="right")
        for cat, cnt in df["category"].value_counts().items():
            dist_table.add_row(cat, str(cnt))
        console.print(dist_table)


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Check overall system readiness."""
    from src.ml.classifier import is_trained
    from src.data.dataset_manager import load
    from src.config import OPENAI_API_KEY, ML_CONFIDENCE_THRESHOLD, FUZZY_MATCH_THRESHOLD

    df = load()
    trained = is_trained()
    has_data = not df.empty
    has_openai = bool(OPENAI_API_KEY)

    console.print("\n[bold]Category Tagger – System Status[/bold]\n")
    console.print(f"  Dataset loaded    : {'[green]✓[/green]' if has_data else '[red]✗[/red]'} ({len(df)} rows)")
    console.print(f"  ML model trained  : {'[green]✓[/green]' if trained else '[yellow]✗ (run: python main.py seed)[/yellow]'}")
    console.print(f"  LLM available     : {'[green]✓[/green]' if has_openai else '[yellow]⚠ No OPENAI_API_KEY (LLM fallback disabled)[/yellow]'}")
    console.print(f"  ML conf threshold : {ML_CONFIDENCE_THRESHOLD:.0%}")
    console.print(f"  Fuzzy threshold   : {FUZZY_MATCH_THRESHOLD}")

    if not has_data:
        console.print("\n[yellow]→ Run '[bold]python main.py seed[/bold]' to get started.[/yellow]")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
