# Selective Market Mediation Model

## Summary

Build a dual-lane production system, but change the objective: the model no longer tries to force a prediction on every match. It learns when AlgoBet has measurable edge against the market and returns `ABSTAIN` otherwise.

“Certainty” is defined as deployment certainty, not guaranteed match prediction: no model can be guaranteed to beat football markets, but production activation can be guaranteed to fail unless the system proves positive classical CLV and robust walk-forward performance.

## Key Changes

- Add real opening/closing odds support instead of treating `odds_home/draw/away` as a closing-line proxy.
- Keep the pure lane as a pre-market prior and explanation source, but do not expect it alone to beat the market.
- Add `model_type="market_mediation"` with two heads:
  - probability residual head: corrects opening/current market probabilities only when pure features add signal;
  - CLV gate head: predicts whether taking the current price will beat the closing price.
- Add a selective production policy:
  - emit `BET_CANDIDATE` only when the lower confidence bound of expected CLV is positive;
  - otherwise emit `ABSTAIN`.
- Closing odds are labels/evaluation data only. They must never be available as inference features.

## Implementation

- Schema/import:
  - Add Alembic migration plus `Match` columns for canonical opening and closing 1X2 odds.
  - Add optional opening/closing Asian handicap and over/under fields when Football-Data exposes `C`-suffix columns.
  - Update `FDImporter` so opening maps from non-`C` odds columns and closing maps from `B365CH/B365CD/B365CA`, `AvgCH/AvgCD/AvgCA`, `MaxCH/MaxCD/MaxCA`, and equivalent closing AH/O/U columns when present.
  - Keep existing `odds_home/draw/away` for backward compatibility, but stop using it for true CLV claims.

- Features/model:
  - Add `market_mediation` feature group using opening/current implied probabilities, overround, entropy, bookmaker disagreement, AH/O/U context, pure-lane probability residuals, and non-market football features.
  - Implement `MarketMediationPredictor` as a saved artifact containing the pure prior model, residual head, CLV head, thresholds, and abstention policy.
  - Train residuals against `actual_one_hot - opening_market_probability`.
  - Train CLV targets per outcome as `opening_odds / closing_odds - 1`.

- Evaluation/activation:
  - Backtest with classical CLV when both opening and closing odds exist.
  - Report opening-market, closing-market, pure-lane, residual-lane, and selected-bet metrics separately.
  - Activation gates:
    - closing-odds coverage >= 80% in every test fold;
    - selected bets >= 200 pooled across folds;
    - mean selected-bet CLV > 0 in at least 3 of 4 walk-forward folds;
    - pooled bootstrap 95% lower bound for selected-bet CLV > 0;
    - final probability log loss must not be worse than opening market log loss;
    - production artifact must include saved feature pipeline, thresholds, fold metrics, CLV CI, selected/abstained counts, and odds coverage report.

## API And Payload

Add request fields:

```json
{
  "model_type": "market_mediation",
  "production_lane": "dual",
  "taken_odds_snapshot": "opening",
  "closing_odds_required": true,
  "min_expected_clv": 0.005,
  "min_positive_clv_probability": 0.55,
  "activate": false
}
```

First production candidate payload:

```json
{
  "model_type": "market_mediation",
  "description": "Top5 selective market mediation with true closing-line CLV",
  "activate": false,
  "tournament_ids": [359, 545, 98, 28, 123],
  "start_date": "2012-07-01",
  "end_date": "2025-06-30T23:59:59",
  "split_strategy": "walk_forward",
  "train_seasons": 8,
  "val_seasons": 1,
  "test_seasons": 1,
  "feature_groups": [
    "team_form",
    "head_to_head",
    "temporal",
    "standings",
    "enriched_stats",
    "draw_signals",
    "matchup_interaction",
    "player_quality",
    "market_mediation"
  ],
  "closing_odds_required": true,
  "min_expected_clv": 0.005,
  "min_positive_clv_probability": 0.55
}
```

## Test Plan

- Unit tests:
  - importer maps opening and closing columns correctly;
  - `market_mediation` features exclude all closing odds;
  - CLV target calculation is correct for H/D/A;
  - predictor returns `ABSTAIN` when thresholds fail.
- Integration tests:
  - `/api/v1/ml/train` accepts `market_mediation`;
  - saved artifact reloads with pure model, residual head, CLV head, and thresholds;
  - backtest uses opening odds as taken price and closing odds for classical CLV.
- Live verification:
  - run migration;
  - re-import/backfill top-5 Football-Data odds;
  - run closing-odds coverage audit;
  - train with `activate=false`;
  - activate only if all gates pass.

## Assumptions

- Production betting decisions are allowed after an opening/current market price exists.
- If true closing odds are missing, the system must not claim production readiness.
- If no statistically positive CLV is found, the correct production behavior is abstention, not another forced classifier.
