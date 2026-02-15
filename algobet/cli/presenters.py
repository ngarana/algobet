"""CLI presenters for formatting and displaying output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from algobet.predictions.evaluation import EvaluationResult


def display_backtest_results(result: EvaluationResult) -> None:
    """Display backtest results in a formatted table."""
    click.echo("=" * 60)
    click.echo("BACKTEST RESULTS")
    click.echo("=" * 60)

    # Classification metrics
    click.echo("\n📊 Classification Metrics:")
    click.echo("-" * 40)
    cm = result.classification
    metrics = [
        ("Accuracy", f"{cm.accuracy:.2%}", "≥50%"),
        ("Log Loss", f"{cm.log_loss:.3f}", "≤0.95"),
        ("Brier Score", f"{cm.brier_score:.3f}", "≤0.20"),
        ("F1 (Macro)", f"{cm.f1_macro:.3f}", "≥0.45"),
        ("Top-2 Accuracy", f"{cm.top_2_accuracy:.2%}", "≥75%"),
        ("Cohen's Kappa", f"{cm.cohen_kappa:.3f}", "≥0.30"),
    ]

    for name, value, target in metrics:
        click.echo(f"  {name:15s}: {value:>8s} (target: {target})")

    # Per-class metrics
    click.echo("\n📈 Per-Class Performance:")
    click.echo("-" * 40)
    click.echo(f"  {'Outcome':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    outcome_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    for outcome in ["H", "D", "A"]:
        name = outcome_names[outcome]
        prec = cm.per_class_precision.get(outcome, 0)
        rec = cm.per_class_recall.get(outcome, 0)
        f1 = cm.per_class_f1.get(outcome, 0)
        click.echo(f"  {name:<10} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")

    # Betting metrics
    if result.betting:
        click.echo("\n💰 Betting Simulation:")
        click.echo("-" * 40)
        bm = result.betting
        betting_metrics = [
            ("Total Bets", f"{bm.total_bets:,}"),
            ("Win Rate", f"{bm.win_rate:.1%}"),
            ("ROI", f"{bm.profit_loss:+.2f} ({bm.roi_percent:+.1f}%)"),
            ("Yield", f"{bm.yield_percent:+.2f}%"),
            ("Max Drawdown", f"{bm.max_drawdown:.1%}"),
            ("Sharpe Ratio", f"{bm.sharpe_ratio:.3f}"),
            ("Avg Winning Odds", f"{bm.average_winning_odds:.2f}"),
        ]
        for name, value in betting_metrics:
            click.echo(f"  {name:18s}: {value}")

    # Calibration
    click.echo("\n🎯 Calibration:")
    click.echo("-" * 40)
    click.echo(f"  Expected Calibration Error: {result.expected_calibration_error:.4f}")
    click.echo(f"  Maximum Calibration Error: {result.maximum_calibration_error:.4f}")

    click.echo("\n" + "=" * 60)


def display_value_bets(value_bets: list[dict], min_ev: float) -> None:
    """Display value bets in a formatted table."""
    click.echo(f"\n{'=' * 80}")
    click.echo(f"VALUE BETS (EV ≥ {min_ev:.1%})")
    click.echo(f"{'=' * 80}\n")

    for vb in value_bets:
        click.echo(f"📅 {vb['date']} | {vb['tournament']}")
        click.echo(f"⚽ {vb['home_team']} vs {vb['away_team']}")
        click.echo(f"🎯 {vb['outcome_name']} @ {vb['market_odds']:.2f}")
        click.echo(
            f"   Predicted: {vb['predicted_prob']:.1%} | "
            f"EV: {vb['expected_value']:+.1%} | "
            f"Kelly: {vb['kelly_fraction']:.1%}"
        )
        click.echo("-" * 80)

    # Summary
    total_bets = len(value_bets)
    avg_ev = sum(vb["expected_value"] for vb in value_bets) / total_bets
    avg_odds = sum(vb["market_odds"] for vb in value_bets) / total_bets

    click.echo("\n📊 Summary:")
    click.echo(f"   Total value bets: {total_bets}")
    click.echo(f"   Average EV: {avg_ev:+.1%}")
    click.echo(f"   Average odds: {avg_odds:.2f}")


def display_calibration_improvement(
    raw_metrics: dict, cal_metrics: dict, metric_names: list[str]
) -> None:
    """Display calibration improvement metrics."""
    click.echo("\n📊 Calibration Improvement:")
    click.echo("-" * 50)
    click.echo(f"{'Metric':<25} {'Before':>10} {'After':>10} {'Δ':>10}")
    click.echo("-" * 50)

    for metric in metric_names:
        before = raw_metrics[metric]
        after = cal_metrics[metric]
        delta = after - before
        sign = "+" if delta > 0 else ""

        # For ECE, MCE, Brier, Log Loss - lower is better
        improvement = "✓" if after < before else "✗"
        click.echo(
            f"{metric:<25} {before:>10.4f} {after:>10.4f} "
            f"{sign}{delta:>9.4f} {improvement}"
        )
