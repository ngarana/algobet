# Feature Ablation Study Summary

**Date**: 2026-05-12
**Objective**: Identify optimal feature subset for EPL match prediction model

## Methodology

Comprehensive ablation testing conducted across 5 feature groups:
- `team_form` (99 features): Rolling performance metrics
- `enriched_stats` (139 features): Detailed match statistics
- `temporal` (12 features): Time-based features
- `head_to_head` (60 features): Historical matchups
- `standings` (24 features): League position indicators

Each feature group was tested independently and in combination via stratified 5-fold cross-validation.

## Key Findings

### 1. Team Form Features Are Optimal
**Result**: `team_form` alone achieved **val_log_loss: 0.9649** (12% improvement vs baseline)

All other feature groups showed no significant improvement:
| Feature Group | Val Log Loss | vs Team Form |
|--------------|--------------|--------------|
| team_form | 0.9649 | baseline |
| team_form + temporal | 1.0673 | +10.4% worse |
| team_form + head_to_head | 1.0949 | +13.5% worse |
| team_form + enriched_stats | 1.0670 | +10.5% worse |
| team_form + standings | N/A | model collapse |

### 2. Standings Features Cause Model Collapse
When `standings` added to any other feature group:
- Training produces degenerate predictions
- Model predicts only 2 classes instead of 3
- Action required: **exclude standings entirely**

### 3. Enriched Stats Are Redundant
Of 139 `enriched_stats` features:
- Only 1 feature (home team shots on target) shows any predictive value
- Remaining 138 are either redundant proxies or noise
- Creates multicollinearity with `team_form` signals

## Model Deployment

### Final Model
- **ID**: `xgboost_20260511_153157`
- **Features**: 99 team_form features only
- **Architecture**: XGBoost classifier
- **Backtest ROI**: 9.69%

### Prediction Characteristics
- Draw recall: ~3% (critically low)
- Overround expectation: ~4.5% for value bets
- ECE (calibration error): 0.086 (acceptable)

## Limitations

### 1. Retrospective Feature Nature
**Team form features are purely historical indicators**
- Based on actual performance, not market expectations
- No implied probabilities from betting markets
- Cannot calibrate to market pricing inefficiencies

**Consequences**:
- Model predictions may diverge from efficient market prices
- Value bet identification relies on model's unique signal, not market mispricing
- No guarantee that historical patterns persist in future matches

### 2. No Market-Based Signals
- Absence of odds, line movements, or betting volume data
- Cannot identify contrarian opportunities where market overreacts
- No sentiment or behavioral bias exploitation

### 3. Draw Prediction Weakness
- Model severely under-predicts draws (3% recall)
- Risk of systematic underestimation in portfolio
- May miss value opportunities in draw markets

### 4. Temporal Assumptions
- Assumes recent form patterns persist
- No accounting for fixture congestion effects
- Squad changes/transfers not explicitly modeled

### 5. Feature Stability
- Rolling windows (3/5/10 matches) may not capture optimal periods
- No automated feature selection within team_form suite
- Static feature set requires manual re-evaluation

### 6. Model Calibration
- Training on historical outcomes may not reflect true probabilities
- No recalibration for different competitive contexts (relegation battles, title races)
- Fixed probability thresholds may not optimize expected value

## Recommendations

1. **Immediate**: Deploy `team_form` only model for production predictions
2. **Short-term**: Implement bet sizing based on model confidence intervals
3. **Medium-term**:
   - Improve draw prediction via targeted feature engineering

## Files Referenced

- `/algobet/predictions/features/team_form_generator.py` - Team form feature definitions
- `/algobet/models/backtest_results.csv` - Backtest performance data
- `/algobet/models/xgboost_20260511_153157/` - Final model artifacts
