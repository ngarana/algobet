# AlgoBet Machine Learning Model Design Document

## Executive Summary

This document presents a comprehensive machine learning architecture for the AlgoBet football match prediction engine. The system is designed to predict match outcomes (home win, draw, away win) with calibrated probability estimates, leveraging historical match data, team statistics, and betting market information.

**Key Design Principles:**
- **Temporal Integrity**: Strict time-based data splits prevent data leakage
- **Hybrid Architecture**: Ensemble of gradient boosting and probabilistic models
- **Interpretability**: SHAP-based explanations for all predictions
- **Production-Ready**: Automated retraining, monitoring, and fallback mechanisms

**Expected Performance Benchmarks:**
| Metric | Target | Baseline (Bookmaker) |
|--------|--------|---------------------|
| Log Loss | < 0.95 | ~1.05 |
| Brier Score | < 0.20 | ~0.21 |
| Top-1 Accuracy | 50-55% | ~48% |
| ROI (Value Bets) | 5-10% | 0% (vig-adjusted) |

---

## Table of Contents

1. [Target Variable Design](#1-target-variable-design)
2. [Feature Engineering Strategy](#2-feature-engineering-strategy)
3. [Model Architecture](#3-model-architecture)
4. [Training Strategy](#4-training-strategy)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Model Performance Expectations](#6-model-performance-expectations)
7. [SQL Feature Extraction](#7-sql-feature-extraction)
8. [Edge Cases & Risk Mitigation](#8-edge-cases--risk-mitigation)

---

## 1. Target Variable Design

### 1.1 Primary Prediction Targets

The model produces predictions for multiple target variables to support various betting markets:

#### 1.1.1 1X2 Match Outcome (Primary Target)
```
Target Encoding:
- H (Home Win): home_score > away_score  → [1, 0, 0]
- D (Draw):     home_score = away_score  → [0, 1, 0]
- A (Away Win): home_score < away_score  → [0, 0, 1]

Class Distribution (Historical Baseline):
- Home Win: ~45-48%
- Draw:     ~25-27%
- Away Win: ~27-30%

Imbalanced nature requires:
- Stratified sampling in train/test splits
- Class weight adjustment during training
- Focal loss or similar techniques for gradient boosting
```

#### 1.1.2 Score Predictions (Secondary Target)
```
Targets:
- home_score: Discrete integer (Poisson/Negative Binomial)
- away_score: Discrete integer (Poisson/Negative Binomial)
- total_goals: home_score + away_score (Over/Under markets)

Applications:
- Both Teams To Score (BTTS) market
- Over/Under 2.5 goals market
- Correct score market (derived)
```

#### 1.1.3 Derived Market Predictions
```
Derived from primary probabilities:
- Double Chance: P(H) + P(D), P(D) + P(A), P(H) + P(A)
- Asian Handicap: Adjusted probabilities based on goal expectancy
- BTTS: P(home_score > 0) × P(away_score > 0)
- Over/Under: Poisson-derived cumulative probabilities
```

### 1.2 Class Imbalance Considerations

```python
# Class weights for imbalanced learning
# Inverse frequency weighting
class_weights = {
    'H': 1.0,    # Most frequent
    'D': 1.8,    # Least frequent (~2x weight)
    'A': 1.6     # Medium frequency
}

# Alternative: Effective number weighting
# beta = 0.9999
# weight = (1 - beta) / (1 - beta^n_c)
```

**Implications:**
- Draw prediction is most challenging due to rarity (~26% of matches)
- Model must be calibrated to avoid over-predicting home wins
- Evaluation must weight all outcomes equally (macro metrics)

---

## 2. Feature Engineering Strategy

### 2.1 Feature Priority Ranking

Based on historical predictive power in football modeling:

| Rank | Feature Category | Expected Importance | Notes |
|------|------------------|---------------------|-------|
| 1 | Betting Odds | 35-40% | Market efficiency captures most information |
| 2 | Team Form (Home/Away) | 20-25% | Recent performance is highly predictive |
| 3 | Head-to-Head | 10-15% | Style matchups matter |
| 4 | Tournament Context | 8-12% | Season stage, motivation |
| 5 | Rest Days | 5-8% | Fatigue effects are real |
| 6 | Historical Season Data | 5-8% | Long-term team quality |

### 2.2 Feature Categories

#### 2.2.1 Team Form Features

**Basic Form Metrics (Last N Matches)**
```python
# For windows N ∈ {3, 5, 10}
form_features = {
    # Points-based (3 for win, 1 for draw, 0 for loss)
    f'home_points_last_{N}': float,      # Average points per game
    f'away_points_last_{N}': float,
    
    # Win/Draw/Loss rates
    f'home_win_rate_{N}': float,         # Wins / matches
    f'home_draw_rate_{N}': float,        # Draws / matches
    f'home_loss_rate_{N}': float,        # Losses / matches
    
    # Goal metrics
    f'home_goals_scored_avg_{N}': float,
    f'home_goals_conceded_avg_{N}': float,
    f'home_goal_diff_avg_{N}': float,
    
    # Same for away team...
}
```

**Form Momentum (Weighted Recent Performance)**
```python
# Exponentially weighted form (recent matches count more)
weights = [0.5, 0.7, 0.85, 0.95, 1.0]  # For last 5 matches

form_momentum = sum(w * points_i for w, points_i in zip(weights, recent_points))

# Form trend (improving vs declining)
form_trend = points_last_3 - points_matches_4_to_6
```

**Home/Away Specific Form**
```python
# Separate tracking for home and away performance
home_advantage_features = {
    'home_team_home_form_5': float,      # Home team's record at home
    'away_team_away_form_5': float,      # Away team's record away
    'home_advantage_index': float,       # Relative home strength
}
```

#### 2.2.2 Head-to-Head Features

```python
h2h_features = {
    # Last 5 meetings between these teams
    'h2h_matches_count': int,            # How many historical meetings
    'h2h_home_wins': int,                # Home team wins in H2H
    'h2h_draws': int,
    'h2h_away_wins': int,
    
    # Goal trends in H2H
    'h2h_avg_total_goals': float,
    'h2h_home_avg_goals': float,
    'h2h_away_avg_goals': float,
    
    # Recent H2H form (last 3 meetings)
    'h2h_recent_home_wins': int,
    
    # Tournament-specific H2H
    'h2h_same_tournament_home_wins': int,
}
```

#### 2.2.3 Tournament & Season Context

```python
tournament_features = {
    # Season stage
    'season_progress': float,            # Match number / total expected matches
    'season_stage': str,                 # 'early' (0-20%), 'mid' (20-75%), 'late' (>75%)
    
    # League table position context
    'home_league_position': int,
    'away_league_position': int,
    'position_diff': int,                # home_pos - away_pos
    
    # Points context
    'home_points_total': int,
    'away_points_total': int,
    'points_diff': int,
    
    # Relegation/Promotion/Championship implications
    'home_relegation_risk': float,       # Distance from drop zone
    'home_title_race': float,            # Distance from top
    'away_relegation_risk': float,
    'away_title_race': float,
}
```

#### 2.2.4 Betting Odds Features

**Implied Probabilities**
```python
# Convert odds to probabilities
implied_prob_home = 1 / odds_home
implied_prob_draw = 1 / odds_draw
implied_prob_away = 1 / odds_away

# Normalize to account for bookmaker margin
margin = implied_prob_home + implied_prob_draw + implied_prob_away - 1
normalized_prob_home = implied_prob_home / (1 + margin)
```

**Value Detection Features**
```python
odds_features = {
    # Raw implied probabilities
    'implied_prob_home': float,
    'implied_prob_draw': float,
    'implied_prob_away': float,
    
    # Bookmaker margin
    'bookmaker_margin': float,
    'num_bookmakers': int,               # Confidence in odds quality
    
    # Market consensus (if multiple bookmakers)
    'odds_home_std': float,              # Standard deviation of home odds
    'odds_consensus_strength': float,    # Inverse of coefficient of variation
    
    # Value indicators (to be compared with model predictions)
    'favorite_outcome': str,             # Which outcome has lowest odds
    'favorite_implied_prob': float,
    'odds_home_away_ratio': float,       # Home vs Away odds ratio
}
```

#### 2.2.5 Temporal Features

```python
temporal_features = {
    # Days since last match (fatigue/recovery)
    'home_rest_days': int,
    'away_rest_days': int,
    'rest_days_diff': int,               # Advantage to more rested team
    
    # Fixture congestion
    'home_matches_last_14_days': int,
    'away_matches_last_14_days': int,
    'home_fixture_density': float,       # Matches per week
    
    # Calendar effects
    'day_of_week': int,                  # 0=Monday, 6=Sunday
    'month': int,                        # Seasonal patterns
    'is_weekend': bool,
    'days_from_season_start': int,
}
```

### 2.3 Feature Engineering Pipeline

```
Raw Data → Feature Generators → Transformers → Feature Store

1. Data Retrieval (SQL)
   ├── Match history
   ├── Team records
   └── Odds data

2. Feature Generation
   ├── TeamFormGenerator (rolling windows)
   ├── HeadToHeadGenerator (historical matchups)
   ├── TournamentContextGenerator (league tables)
   ├── OddsFeatureGenerator (probability conversion)
   └── TemporalFeatureGenerator (dates, rest days)

3. Feature Transformation
   ├── StandardScaler (numerical features)
   ├── OneHotEncoder (categorical)
   └── PCA (optional dimensionality reduction)

4. Storage
   └── PostgreSQL (model_features table as JSONB)
```

---

## 3. Model Architecture

### 3.1 Proposed Architecture: Hybrid Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE INPUT LAYER                          │
│  [Team Form] [H2H] [Tournament] [Odds] [Temporal]              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   XGBoost     │    │   LightGBM    │    │    Random     │
│  Classifier   │    │  Classifier   │    │    Forest     │
│               │    │               │    │               │
│  - High perf  │    │  - Fast train │    │  - Robust     │
│  - Handles    │    │  - Large data │    │  - Low corr   │
│    missing    │    │  - Native cat │    │    w/ GBM     │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                     │                     │
        │    P(H), P(D), P(A) │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     META-CLASSIFIER (LR)      │
              │   Logistic Regression Stack   │
              │   - Calibrates probabilities  │
              │   - Combines model outputs    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   PROBABILITY CALIBRATION     │
              │      (Isotonic Regression)    │
              │   - Ensures calibrated probs  │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     FINAL OUTPUT LAYER        │
              │  P(Home Win), P(Draw), P(Away)│
              │  Confidence Score, SHAP Values│
              └───────────────────────────────┘
```

### 3.2 Model Selection Justification

#### 3.2.1 Primary Model: XGBoost Classifier

**Why XGBoost:**
- **Handles tabular data excellently**: Football features are inherently tabular
- **Missing value handling**: Historical data has gaps (odds, early season)
- **Feature importance**: Built-in importance scores for interpretability
- **Regularization**: Built-in L1/L2 prevents overfitting on limited data
- **Industry standard**: Proven track record in Kaggle competitions and production systems

**Hyperparameter Space:**
```python
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'max_depth': [4, 6, 8],              # Tree complexity
    'learning_rate': [0.01, 0.05, 0.1],  # Shrinkage
    'n_estimators': [200, 500, 1000],    # Number of trees
    'subsample': [0.6, 0.8, 1.0],        # Row sampling
    'colsample_bytree': [0.6, 0.8, 1.0], # Feature sampling
    'min_child_weight': [1, 3, 5],       # Minimum leaf samples
    'gamma': [0, 0.1, 0.2],              # Minimum loss reduction
    'reg_alpha': [0, 0.1, 1.0],          # L1 regularization
    'reg_lambda': [1, 2, 5],             # L2 regularization
}
```

#### 3.2.2 Secondary Model: LightGBM

**Why LightGBM:**
- **Training speed**: 3-5x faster than XGBoost
- **Memory efficient**: Handles large datasets with less RAM
- **Native categorical**: Better handling of categorical features
- **Leaf-wise growth**: Can capture complex patterns

**Use Case**: Fast iteration during development and as ensemble diversity source.

#### 3.2.3 Tertiary Model: Random Forest

**Why Random Forest:**
- **Low correlation with GBM**: Uses different optimization approach
- **Robust to outliers**: Less sensitive to extreme values
- **No hyperparameter sensitivity**: More stable with default settings
- **Bagging approach**: Reduces variance in ensemble

**Role**: Provides diversity in ensemble, improving overall robustness.

### 3.3 Meta-Model: Logistic Regression Stacking

```python
# Stacking architecture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

estimators = [
    ('xgb', xgb_model),
    ('lgb', lgb_model),
    ('rf', rf_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        C=1.0
    ),
    cv=5,
    stack_method='predict_proba'
)
```

**Benefits of Stacking:**
- Calibrates base model probabilities
- Learns optimal weighting of base models
- Reduces overfitting through cross-validation

### 3.4 Score Prediction Model (Poisson Regression)

For goal prediction and derived markets:

```python
# Independent Poisson models for home and away goals
from sklearn.linear_model import PoissonRegressor

# Home goals model
home_goal_model = PoissonRegressor(alpha=0.1, max_iter=1000)
home_goal_model.fit(X, home_goals)

# Away goals model  
away_goal_model = PoissonRegressor(alpha=0.1, max_iter=1000)
away_goal_model.fit(X, away_goals)

# Predict expected goals
lambda_home = home_goal_model.predict(X_new)
lambda_away = away_goal_model.predict(X_new)

# Derived predictions
btts_prob = (1 - poisson.pmf(0, lambda_home)) * (1 - poisson.pmf(0, lambda_away))
over_2_5_prob = 1 - poisson.cdf(2, lambda_home + lambda_away)
```

### 3.5 Model Comparison

| Aspect | XGBoost | LightGBM | Random Forest | Neural Network |
|--------|---------|----------|---------------|----------------|
| **Accuracy** | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **Training Speed** | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| **Interpretability** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| **Missing Data** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| **Small Data** | ★★★★★ | ★★★★☆ | ★★★★★ | ★★☆☆☆ |
| **Feature Importance** | Built-in | Built-in | Built-in | SHAP required |

**Recommendation**: Start with XGBoost as primary, add LightGBM and RF for ensemble diversity. Neural networks require more data and tuning for marginal gains in this domain.

---

## 4. Training Strategy

### 4.1 Temporal Data Splitting

**Critical Rule**: Never use future data to predict past matches.

```
Timeline-based Split Strategy:

2020-01-01 ────────────────────────────────────────── 2024-12-31
│←───────── Train (60%) ─────────→│←─ Val (20%) ─→│←─ Test (20%) ─→│
│                                 │               │               │
2020-01-01                    2022-12-31      2023-12-31      2024-12-31
```

**Implementation:**
```python
def temporal_split(df, train_end, val_end):
    """Split data temporally to prevent data leakage."""
    train = df[df['match_date'] < train_end]
    val = df[(df['match_date'] >= train_end) & (df['match_date'] < val_end)]
    test = df[df['match_date'] >= val_end]
    return train, val, test

# Example splits
train_end = '2022-12-31'
val_end = '2023-12-31'
train, val, test = temporal_split(df, train_end, val_end)
```

### 4.2 Time-Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Expanding window cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold}: Validation score = {score:.4f}")
```

### 4.3 Hyperparameter Tuning

**Bayesian Optimization with Optuna:**
```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    
    model = XGBClassifier(**params)
    
    # Time-series cross-validation
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)
        scores.append(log_loss(y_val, preds))
    
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 4.4 Class Balancing Strategy

```python
# Method 1: Class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

model = XGBClassifier(
    scale_pos_weight=class_weight_dict,
    # ... other params
)

# Method 2: SMOTE (Synthetic Minority Oversampling)
# Note: Only apply within training set, not validation/test
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 4.5 Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling (sigmoid calibration)
calibrated_clf = CalibratedClassifierCV(
    base_estimator=model,
    method='sigmoid',  # or 'isotonic'
    cv=5
)
calibrated_clf.fit(X_train, y_train)

# Predict calibrated probabilities
calibrated_probs = calibrated_clf.predict_proba(X_test)
```

---

## 5. Evaluation Metrics

### 5.1 Classification Metrics

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| **Accuracy** | (TP+TN)/Total | > 50% | Overall correct predictions |
| **Log Loss** | -Σ(y·log(p)) | < 0.95 | Penalizes confident wrong predictions |
| **Brier Score** | Σ(p - y)² | < 0.20 | Probability calibration quality |
| **F1 (Macro)** | 2·P·R/(P+R) | > 0.45 | Balanced precision/recall per class |
| **ROC-AUC** | AUC of ROC curve | > 0.65 | Ranking quality |

### 5.2 Betting-Specific Metrics

**Return on Investment (ROI):**
```python
def calculate_roi(predictions, odds, actual_results, stake=1.0):
    """
    Calculate ROI from predictions.
    
    Strategy: Bet when predicted probability > implied probability + margin
    """
    total_staked = 0
    total_return = 0
    
    for pred, odd, actual in zip(predictions, odds, actual_results):
        implied_prob = 1 / odd
        model_prob = pred[actual]  # Predicted probability for actual outcome
        
        # Value bet: Model probability exceeds implied probability
        if model_prob > implied_prob * 1.05:  # 5% edge threshold
            total_staked += stake
            if predicted_outcome == actual:
                total_return += stake * odd
    
    roi = (total_return - total_staked) / total_staked if total_staked > 0 else 0
    return roi
```

**Yield:**
```
Yield = (Total Profit / Total Staked) × 100%

Target: > 5% yield over 500+ bets for statistical significance
```

**Kelly Criterion for Staking:**
```python
def kelly_stake(bankroll, prob, odds, fraction=0.25):
    """
    Calculate Kelly stake (fractional for risk management).
    
    f* = (bp - q) / b
    where: b = odds - 1, p = probability, q = 1 - p
    """
    b = odds - 1
    p = prob
    q = 1 - p
    
    kelly_fraction = (b * p - q) / b
    stake = bankroll * kelly_fraction * fraction  # Quarter-Kelly for safety
    
    return max(0, stake)  # Don't bet if negative expected value
```

### 5.3 Calibration Metrics

**Expected Calibration Error (ECE):**
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Measure how well predicted probabilities match actual outcomes."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece
```

**Interpretation:**
- ECE < 0.05: Well-calibrated
- ECE 0.05-0.1: Acceptable
- ECE > 0.1: Poor calibration, needs adjustment

### 5.4 Ranking Metrics

```python
# Rank probability score - how well model ranks teams
from sklearn.metrics import ndcg_score

# For predicting which matches are most likely to be home wins
y_true_relevance = (y_test == 'H').astype(int)
y_scores = predictions_proba[:, 0]  # Home win probabilities

ndcg = ndcg_score([y_true_relevance], [y_scores])
```

---

## 6. Model Performance Expectations

### 6.1 Realistic Accuracy Benchmarks

**Historical Context:**
- **Naive Baseline** (Always predict home win): ~45% accuracy
- **Bookmaker Odds** (Favorite outcome): ~48-50% accuracy
- **Simple Elo Model**: ~50-52% accuracy
- **Advanced ML Models**: ~52-56% accuracy
- **Theoretical Maximum** (given football randomness): ~60-65%

**AlgoBet Targets:**

| Metric | Conservative | Target | Optimistic |
|--------|-------------|--------|------------|
| **Top-1 Accuracy** | 50% | 53% | 56% |
| **Top-2 Accuracy** | 72% | 76% | 80% |
| **Log Loss** | 1.00 | 0.95 | 0.90 |
| **Brier Score** | 0.22 | 0.20 | 0.18 |
| **ROI (Value Bets)** | 3% | 7% | 12% |
| **Calibration ECE** | 0.08 | 0.05 | 0.03 |

### 6.2 Per-Outcome Performance

| Outcome | Expected Precision | Expected Recall | F1-Score |
|---------|-------------------|-----------------|----------|
| Home Win | 55-60% | 65-70% | 0.60-0.65 |
| Draw | 30-35% | 20-25% | 0.25-0.30 |
| Away Win | 45-50% | 50-55% | 0.48-0.52 |

**Note:** Draws are inherently hardest to predict due to class imbalance and higher variance.

### 6.3 Bookmaker Comparison

**Market Efficiency Test:**
```
Hypothesis: Bookmaker odds are efficient (impossible to beat consistently)

Test: Compare model predictions vs closing odds

If model_roc_auc > 0.52 vs bookmaker implied probabilities:
    → Potential edge exists
    
If model_roc_auc > 0.55:
    → Statistically significant edge
    
If model_roi > 5% over 1000+ bets:
    → Profitable in practice
```

**Expected vs Bookmaker:**
- Model should slightly outperform naive odds conversion
- 2-5% accuracy improvement is realistic
- ROI of 5-10% on value bets is achievable with discipline

---

## 7. SQL Feature Extraction

### 7.1 Rolling Form Features

```sql
-- Team form calculation using window functions
WITH team_matches AS (
    SELECT 
        m.id AS match_id,
        m.match_date,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        CASE 
            WHEN m.home_score > m.away_score THEN 'H'
            WHEN m.home_score < m.away_score THEN 'A'
            ELSE 'D'
        END AS result,
        -- Points for home team
        CASE 
            WHEN m.home_score > m.away_score THEN 3
            WHEN m.home_score = m.away_score THEN 1
            ELSE 0
        END AS home_points,
        -- Points for away team
        CASE 
            WHEN m.away_score > m.home_score THEN 3
            WHEN m.away_score = m.home_score THEN 1
            ELSE 0
        END AS away_points
    FROM matches m
    WHERE m.status = 'FINISHED'
),
-- Unpivot to get team-centric view
team_performance AS (
    SELECT 
        match_id,
        match_date,
        home_team_id AS team_id,
        home_points AS points,
        home_score AS goals_for,
        away_score AS goals_against,
        'H' AS venue
    FROM team_matches
    UNION ALL
    SELECT 
        match_id,
        match_date,
        away_team_id AS team_id,
        away_points AS points,
        away_score AS goals_for,
        home_score AS goals_against,
        'A' AS venue
    FROM team_matches
),
-- Calculate rolling averages
rolling_form AS (
    SELECT 
        match_id,
        team_id,
        venue,
        AVG(points) OVER (
            PARTITION BY team_id 
            ORDER BY match_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_points_last_5,
        AVG(goals_for) OVER (
            PARTITION BY team_id 
            ORDER BY match_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_goals_for_last_5,
        AVG(goals_against) OVER (
            PARTITION BY team_id 
            ORDER BY match_date 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS avg_goals_against_last_5
    FROM team_performance
)
SELECT * FROM rolling_form;
```

### 7.2 Head-to-Head Features

```sql
-- Head-to-head statistics
WITH h2h_stats AS (
    SELECT 
        m1.home_team_id,
        m1.away_team_id,
        m1.id AS match_id,
        m1.match_date,
        -- H2H record (last 5 meetings, excluding current)
        (
            SELECT COUNT(*)
            FROM matches m2
            WHERE m2.status = 'FINISHED'
                AND ((m2.home_team_id = m1.home_team_id AND m2.away_team_id = m1.away_team_id)
                     OR (m2.home_team_id = m1.away_team_id AND m2.away_team_id = m1.home_team_id))
                AND m2.match_date < m1.match_date
                AND m2.match_date >= m1.match_date - INTERVAL '3 years'
            ORDER BY m2.match_date DESC
            LIMIT 5
        ) AS h2h_matches_count,
        -- Home team wins in H2H
        (
            SELECT COUNT(*)
            FROM matches m2
            WHERE m2.status = 'FINISHED'
                AND m2.home_team_id = m1.home_team_id 
                AND m2.away_team_id = m1.away_team_id
                AND m2.home_score > m2.away_score
                AND m2.match_date < m1.match_date
            ORDER BY m2.match_date DESC
            LIMIT 5
        ) AS h2h_home_wins,
        -- Average goals in H2H
        (
            SELECT AVG(m2.home_score + m2.away_score)
            FROM matches m2
            WHERE m2.status = 'FINISHED'
                AND ((m2.home_team_id = m1.home_team_id AND m2.away_team_id = m1.away_team_id)
                     OR (m2.home_team_id = m1.away_team_id AND m2.away_team_id = m1.home_team_id))
                AND m2.match_date < m1.match_date
            ORDER BY m2.match_date DESC
            LIMIT 5
        ) AS h2h_avg_goals
    FROM matches m1
    WHERE m1.status = 'SCHEDULED'
)
SELECT * FROM h2h_stats;
```

### 7.3 League Table Context

```sql
-- League table position calculation
WITH season_matches AS (
    SELECT 
        m.*,
        t.name AS tournament_name,
        s.name AS season_name
    FROM matches m
    JOIN tournaments t ON m.tournament_id = t.id
    JOIN seasons s ON m.season_id = s.id
    WHERE m.status = 'FINISHED'
),
team_records AS (
    SELECT 
        tournament_id,
        season_id,
        home_team_id AS team_id,
        CASE WHEN home_score > away_score THEN 3
             WHEN home_score = away_score THEN 1
             ELSE 0 END AS points,
        home_score AS gf,
        away_score AS ga
    FROM season_matches
    UNION ALL
    SELECT 
        tournament_id,
        season_id,
        away_team_id AS team_id,
        CASE WHEN away_score > home_score THEN 3
             WHEN away_score = home_score THEN 1
             ELSE 0 END AS points,
        away_score AS gf,
        home_score AS ga
    FROM season_matches
),
league_standings AS (
    SELECT 
        tournament_id,
        season_id,
        team_id,
        SUM(points) AS total_points,
        SUM(gf) AS goals_for,
        SUM(ga) AS goals_against,
        SUM(gf) - SUM(ga) AS goal_diff,
        RANK() OVER (PARTITION BY tournament_id, season_id ORDER BY SUM(points) DESC, SUM(gf) - SUM(ga) DESC) AS position
    FROM team_records
    GROUP BY tournament_id, season_id, team_id
)
SELECT * FROM league_standings;
```

### 7.4 Training Dataset Construction

```sql
-- Complete training dataset query
WITH features AS (
    SELECT 
        m.id AS match_id,
        m.match_date,
        m.tournament_id,
        m.season_id,
        m.home_team_id,
        m.away_team_id,
        m.home_score,
        m.away_score,
        m.odds_home,
        m.odds_draw,
        m.odds_away,
        -- Target variable
        CASE 
            WHEN m.home_score > m.away_score THEN 'H'
            WHEN m.home_score < m.away_score THEN 'A'
            ELSE 'D'
        END AS result,
        -- Odds implied probabilities
        1.0 / m.odds_home AS implied_prob_home,
        1.0 / m.odds_draw AS implied_prob_draw,
        1.0 / m.odds_away AS implied_prob_away,
        -- Temporal features
        EXTRACT(DOW FROM m.match_date) AS day_of_week,
        EXTRACT(MONTH FROM m.match_date) AS month,
        -- Days since last match (would need LAG window function)
        m.match_date - LAG(m.match_date) OVER (
            PARTITION BY m.home_team_id ORDER BY m.match_date
        ) AS home_rest_days
    FROM matches m
    WHERE m.status = 'FINISHED'
        AND m.home_score IS NOT NULL
        AND m.odds_home IS NOT NULL  -- Require odds for training
)
SELECT * FROM features
ORDER BY match_date;
```

---

## 8. Edge Cases & Risk Mitigation

### 8.1 Edge Case Handling

#### Newly Promoted Teams

**Problem:** Limited historical data in top division.

**Mitigation:**
```python
# Feature: Is newly promoted
is_newly_promoted = team_not_in_last_season_top_division

# Fallback: Use second-division form with downgrade factor
second_div_form *= 0.85  # Expected performance reduction

# Increase uncertainty for these predictions
if is_newly_promoted:
    confidence_score *= 0.8
    # Wider prediction intervals
```

#### Derby Matches

**Problem:** Form often irrelevant; intensity changes dynamics.

**Detection:**
```python
# Identify derby by: same city, historical rivalry, close league positions
derby_indicators = {
    'same_city': teams_in_same_city(home_team, away_team),
    'historical_rivalry': match_id in rivalry_database,
    'close_positions': abs(home_position - away_position) <= 3,
    'high_attendance_expected': avg_attendance > 1.5 * stadium_capacity
}

is_derby = sum(derby_indicators.values()) >= 2
```

**Adjustment:**
```python
if is_derby:
    # Reduce form feature weights
    form_weight *= 0.7
    # Increase draw probability (derbies are tighter)
    draw_prob_boost = 0.05
    # Reduce confidence
    confidence *= 0.85
```

#### End-of-Season Matches

**Problem:** Motivation varies greatly (chasing titles, avoiding relegation, mid-table nothing to play for).

**Detection:**
```python
season_progress = matches_played / total_expected_matches
is_late_season = season_progress > 0.75

motivation_context = {
    'home_title_race': home_position <= 3 and season_progress > 0.7,
    'home_relegation_battle': home_position >= num_teams - 3,
    'home_mid_table': not (title_race or relegation_battle),
    # Same for away...
}
```

**Adjustment:**
```python
if is_late_season:
    if motivation_context['home_title_race'] and motivation_context['away_mid_table']:
        home_prob_boost = 0.08
    elif motivation_context['home_mid_table'] and motivation_context['away_relegation_battle']:
        home_prob_penalty = 0.05
```

### 8.2 Data Quality Checks

```python
# Critical validation rules
DATA_QUALITY_RULES = {
    # Odds validation
    'odds_range': lambda o: 1.01 <= o <= 100,
    'odds_margin': lambda h, d, a: 1.0 < (1/h + 1/d + 1/a) < 1.3,
    
    # Score validation
    'score_non_negative': lambda s: s >= 0,
    'reasonable_score': lambda s: s <= 15,  # Unlikely to see more
    
    # Date validation
    'date_in_past': lambda d: d <= datetime.now(),
    'date_not_too_old': lambda d: d > datetime(1990, 1, 1),
    
    # Team validation
    'different_teams': lambda h, a: h != a,
    'teams_exist': lambda h, a: h in team_database and a in team_database,
}

def validate_match_data(match_row):
    """Run all validation rules on match data."""
    errors = []
    
    # Check odds
    if not DATA_QUALITY_RULES['odds_range'](match_row['odds_home']):
        errors.append("Invalid home odds")
    
    # Check margin
    margin = sum(1/o for o in [match_row['odds_home'], 
                               match_row['odds_draw'], 
                               match_row['odds_away']])
    if not (1.0 < margin < 1.3):
        errors.append(f"Suspicious odds margin: {margin:.3f}")
    
    return errors
```

### 8.3 Confidence Scoring System

```python
def calculate_confidence_score(probabilities, feature_completeness, edge_cases):
    """
    Calculate overall prediction confidence (0.0 to 1.0).
    
    Factors:
    1. Prediction entropy (lower = more confident)
    2. Feature completeness
    3. Data quality
    4. Edge case penalties
    """
    # Entropy-based confidence
    probs = np.array(probabilities)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(3)  # Maximum for 3 classes
    entropy_confidence = 1 - (entropy / max_entropy)
    
    # Feature completeness
    expected_features = 50  # Total expected features
    completeness_factor = feature_completeness / expected_features
    
    # Edge case penalties
    penalty = 1.0
    if edge_cases['is_newly_promoted']:
        penalty *= 0.85
    if edge_cases['is_derby']:
        penalty *= 0.90
    if edge_cases['is_late_season']:
        penalty *= 0.95
    
    # Combined confidence
    confidence = entropy_confidence * 0.5 + completeness_factor * 0.3 + penalty * 0.2
    
    return min(1.0, max(0.0, confidence))
```

### 8.4 Fallback Mechanisms

```python
def generate_prediction(match_features, ml_model, fallback_rules):
    """
    Generate prediction with fallback for low confidence.
    """
    # Try ML model first
    ml_prediction = ml_model.predict_proba(match_features)
    ml_confidence = calculate_confidence_score(ml_prediction, ...)
    
    # Fallback decision
    if ml_confidence < 0.4:
        # Use rules-based fallback
        return rules_based_prediction(match_features, fallback_rules)
    elif ml_confidence < 0.6:
        # Blend ML and rules
        rules_pred = rules_based_prediction(match_features, fallback_rules)
        blended = 0.7 * ml_prediction + 0.3 * rules_pred
        return blended
    else:
        # Use ML prediction
        return ml_prediction

def rules_based_prediction(features, rules):
    """
    Simple rules-based fallback when ML confidence is low.
    """
    probs = [0.33, 0.33, 0.33]  # Start uniform
    
    # Home advantage
    probs[0] += 0.10  # Home win boost
    probs[2] -= 0.05  # Away win penalty
    
    # Form adjustment
    if features['home_form_5'] > features['away_form_5']:
        probs[0] += 0.05
        probs[2] -= 0.05
    
    # Normalize
    total = sum(probs)
    return [p/total for p in probs]
```

### 8.5 Model Monitoring & Drift Detection

```python
# Performance drift detection
class ModelMonitor:
    def __init__(self, baseline_metrics, threshold=0.1):
        self.baseline = baseline_metrics
        self.threshold = threshold
        
    def check_drift(self, recent_metrics):
        """Check if model performance has degraded."""
        alerts = []
        
        # Log loss drift
        if recent_metrics['log_loss'] > self.baseline['log_loss'] * (1 + self.threshold):
            alerts.append("LOG_LOSS_DRIFT: Model calibration degraded")
        
        # Accuracy drift
        if recent_metrics['accuracy'] < self.baseline['accuracy'] * (1 - self.threshold):
            alerts.append("ACCURACY_DRIFT: Prediction accuracy dropped")
        
        # ROI drift
        if recent_metrics['roi'] < self.baseline['roi'] - 0.05:
            alerts.append("ROI_DRIFT: Betting performance declined")
        
        return alerts
    
    def check_feature_drift(self, recent_features, baseline_distribution):
        """Detect if input feature distributions have shifted."""
        # KS test for continuous features
        # Chi-square for categorical
        pass
```

---

## Appendix A: Production Deployment Checklist

### A.1 Pre-deployment Requirements

- [ ] Model achieves target log loss (< 0.95) on holdout test set
- [ ] Calibration ECE < 0.05
- [ ] ROI positive on 500+ bet backtest
- [ ] All edge cases have handling logic
- [ ] Fallback mechanism tested
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds set

### A.2 Model Refresh Schedule

| Model Component | Refresh Frequency | Trigger Condition |
|-----------------|-------------------|-------------------|
| Feature Store | Daily | New match results |
| Base Models | Weekly | Performance drift detected |
| Full Retraining | Monthly | Or accuracy drop > 5% |
| Hyperparameters | Quarterly | Or on major schema change |

### A.3 Minimum Data Requirements

For reliable predictions on a new match:
- Home team: Minimum 5 matches in database
- Away team: Minimum 5 matches in database
- H2H history: Nice to have, not required
- Odds data: Required for market features

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **1X2** | Three-way betting market: 1=Home Win, X=Draw, 2=Away Win |
| **Brier Score** | Mean squared error of probability predictions |
| **Calibration** | How well predicted probabilities match actual frequencies |
| **ECE** | Expected Calibration Error |
| **H2H** | Head-to-Head (historical matchups between teams) |
| **Kelly Criterion** | Formula for optimal bet sizing based on edge |
| **Log Loss** | Cross-entropy loss for classification |
| **ROI** | Return on Investment |
| **SHAP** | SHapley Additive exPlanations (feature importance) |
| **Yield** | Profit as percentage of total staked |

---

*Document Version: 1.0*
*Author: AlgoBet Data Science Team*
*Last Updated: 2026-01-31*
