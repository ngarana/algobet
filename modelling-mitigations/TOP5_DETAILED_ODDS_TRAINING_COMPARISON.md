# Top-5 Detailed Odds Training Comparison

Date: 2026-05-17

## Scope

Two XGBoost models were trained on the imported top-5 league data to isolate
the impact of the newly exposed `detailed_odds` feature group.

Common configuration:

| Setting | Value |
|---|---|
| Tournament IDs | `[359, 545, 98, 28, 123]` |
| Excluded tournament | Duplicate Bundesliga id `360` |
| Date range | `2012-07-01` to `2025-06-30` |
| Split | `walk_forward`, 8 train seasons, 1 validation season, 1 test season |
| Walk-forward folds | 4 |
| Model | `xgboost` |
| Tuning | Disabled |
| Feature selection | Enabled, threshold `0.005`, min samples per feature `40` |
| Calibration | Disabled for this comparison |
| Activation | Not activated |

## Runs

| Run | Model Version | DB ID | Feature Groups |
|---|---:|---:|---|
| Baseline | `xgboost_20260517_170218` | 231 | default football groups |
| Detailed odds | `xgboost_20260517_172949` | 232 | default football groups + `detailed_odds` |

Default football groups:

```text
team_form, head_to_head, temporal, standings, enriched_stats,
draw_signals, matchup_interaction
```

## Walk-Forward Test Metrics

Lower is better for Brier, ECE, max prediction share, and market probability
MAE. Higher is better for accuracy and F1.

| Metric | Baseline | + Detailed Odds | Delta |
|---|---:|---:|---:|
| Accuracy | 0.3276 | 0.3127 | -0.0149 |
| F1 macro | 0.3235 | 0.2978 | -0.0257 |
| Brier score | 0.2224 | 0.2227 | +0.0003 |
| Expected calibration error | 0.2064 | 0.2137 | +0.0073 |
| Max prediction share | 0.4110 | 0.5251 | +0.1141 |
| Market probability MAE | 0.1333 | 0.1339 | +0.0006 |
| Market log loss | 0.9656 | 0.9656 | 0.0000 |

## Final Split Test Metrics

| Metric | Baseline | + Detailed Odds | Delta |
|---|---:|---:|---:|
| Accuracy | 0.3394 | 0.3338 | -0.0056 |
| F1 macro | 0.3357 | 0.3309 | -0.0048 |
| Brier score | 0.2222 | 0.2223 | +0.0001 |
| Expected calibration error | 0.1696 | 0.1696 | +0.0000 |
| Max prediction share | 0.4116 | 0.4023 | -0.0094 |
| Market probability MAE | 0.1358 | 0.1359 | +0.0001 |

## Feature Selection

| Run | Selected Feature Count | Selected Detailed Odds Features |
|---|---:|---|
| Baseline | 114 | 0 |
| Detailed odds | 101 | 6 |

Selected detailed odds features:

```text
avg_implied_prob_draw
odds_disagreement_away
odds_disagreement_home
ou_vs_1x2_total_diff
over_under_25_over_odds
over_under_25_under_odds
```

## Conclusion

Adding `detailed_odds` did not improve this top-5 walk-forward run. It reduced
average test accuracy by 1.49 percentage points, reduced macro F1 by 2.57
points, slightly worsened Brier score and ECE, and made predictions more
concentrated by max class share. Neither comparison model was activated.

The metric serialization path was also fixed after this run so future
walk-forward summaries retain numpy scalar `log_loss` values. These two saved
comparison runs do not include model log-loss keys in their stored metrics, so
the comparison above uses the retained metrics from the completed runs.
