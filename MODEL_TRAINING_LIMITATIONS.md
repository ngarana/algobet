# Model Training Data Selection Limitations

**Last Updated:** 2026-04-25
**Status:** Documented - Requires Implementation

## Overview

The current model training interface in AlgoBet provides **severely limited control** over which data is used for training. Users can only configure temporal and volume parameters, but cannot select specific teams, tournaments, match types, or feature groups.

---

## Current Capabilities ✅

| Parameter | UI Location | Description |
|-----------|-------------|-------------|
| **Start Date** | Data Range Section | Earliest match date to include |
| **End Date** | Data Range Section | Latest match date to include |
| **Min Matches** | Data Range Section | Minimum matches required (100-10,000) |
| **Train Ratio** | Data Split Section | % of data for training (default 70%) |
| **Val Ratio** | Data Split Section | % of data for validation (default 15%) |
| **Test Ratio** | Data Split Section | % of data for testing (default 15%) |
| **Model Type** | Basic Settings | XGBoost, LightGBM, or Random Forest |
| **Description** | Basic Settings | Free-text model description |
| **Activate on Train** | Basic Settings | Auto-activate model after training |
| **Hyperparameter Tuning** | Basic Settings | Enable/disable Optuna optimization |
| **Tuning Trials** | Training Settings (Advanced) | Number of Optuna trials (default 50), shown when tuning is enabled |
| **Calibrate Probabilities** | Basic Settings | Enable/disable probability calibration |
| **Calibration Method** | Training Settings (Advanced) | Isotonic or sigmoid calibration method |
| **Random Seed** | Training Settings (Advanced) | Reproducibility seed (default 42) |
| **Early Stopping Rounds** | Training Settings (Advanced) | Stop training if no improvement after N rounds (default 50) |

---

## Missing Capabilities ❌

### 1. Team Selection

**Impact:** Cannot train models focused on specific teams or exclude problematic teams

| Feature | Status | Description |
|---------|--------|-------------|
| Include specific teams (home) | ❌ Not available | Cannot limit training to matches involving specific teams |
| Include specific teams (away) | ❌ Not available | Cannot filter by away team participation |
| Exclude specific teams | ❌ Not available | Cannot exclude teams with poor data quality |
| Home-only matches filter | ❌ Not available | Cannot train only on home matches for analysis |
| Away-only matches filter | ❌ Not available | Cannot train only on away matches for analysis |

**Example Use Case:**
A user wants to train a model specifically on Premier League matches but exclude Leeds United (due to relegation/promotion data quality issues). Currently impossible.

---

### 2. Tournament/League Selection

**Impact:** Cannot create specialized models for specific competitions

| Feature | Status | Description |
|---------|--------|-------------|
| Include specific tournaments | ❌ Not available | Cannot filter by league/cup |
| Multi-tournament selection | ❌ Not available | Cannot combine specific competitions |
| Exclude specific tournaments | ❌ Not available | Cannot exclude low-quality leagues |

**Example Use Case:**
A user wants to train separate models:
- Model A: Premier League only (high-quality data)
- Model B: Championship only (different play style)
- Model C: Champions League only (elite teams)

Currently requires manual database queries or training on all tournaments together.

---

### 3. Match Quality Filters

**Impact:** Cannot ensure high-quality training data

| Feature | Status | Description |
|---------|--------|-------------|
| Minimum total goals | ❌ Not available | Cannot filter low-scoring matches |
| Maximum total goals | ❌ Not available | Cannot filter outlier matches |
| Odds availability required | ⚠️ Partial | Hardcoded `True` in backtest only; training does NOT require odds, but neither is configurable |
| Odds confidence threshold | ❌ Not available | Cannot filter by bookmaker consensus |
| Match importance | ❌ Not available | Cannot filter by tournament stage (finals, group stage) |

**Example Use Case:**
A user wants to train only on "high-quality" matches:
- Total goals ≥ 1.5
- Both teams in top division
- Odds available from ≥5 bookmakers

Currently impossible.

---

### 4. Feature Group Selection

**Impact:** Cannot control model complexity or focus areas

| Feature Group | Generator Class | Status | Description |
|---------------|-----------------|--------|-------------|
| Temporal features | `TemporalFeatureGenerator` | ❌ Fixed | Day of week, month, season start, rest days, fixture density (9 features) |
| Team form (recent) | `TeamFormGenerator` | ❌ Fixed | Points, win rates, goals for/against per window of 3/5/10 matches, venue form, momentum (33 features) |
| Head-to-head history | `HeadToHeadGenerator` | ❌ Fixed | H2H win rates, avg goals, recent form (9 features) |
| Market odds features | `OddsFeatureGenerator` | ❌ Fixed | Implied probabilities, margins, favorite detection, odds quality score (8 features) |
| Custom feature groups | — | ❌ Not available | Cannot select subsets or toggle individual generators |

**Example Use Case:**
A user wants to test which features matter most:
- Model A: Only odds + team form
- Model B: Odds + H2H + temporal
- Model C: All features

Currently impossible without code changes to the feature pipeline.

---

### 5. Data Stratification

**Impact:** Cannot ensure balanced training data

| Feature | Status | Description |
|---------|--------|-------------|
| Outcome balancing | ⚠️ Partial | Inverse-frequency class weights are auto-applied during training (`get_class_weights()` in `split.py`), but user cannot control the strategy or disable/override it |
| Temporal stratification | ❌ Not available | Cannot ensure equal representation across seasons |
| Tournament stratification | ❌ Not available | Cannot control tournament representation |

**Example Use Case:**
Dataset has 60% Home wins, 25% Draws, 15% Away wins. User wants to:
- Override auto-weights to balance to 40/30/30 for better generalization
- Or disable auto-weighting and use oversampling instead

Currently auto-weighting is always applied with no user control.

---

## Technical Limitations

### Backend (`ml_operations.py`)

```python
class TrainModelRequest(BaseModel):
    # Existing fields (omitted for brevity — see Current Capabilities table)
    hyperparameters: dict[str, Any] = {}  # API accepts custom hyperparams, but no UI to set them
    # Missing fields that should exist:
    tournament_ids: list[int] | None = None # NOT IMPLEMENTED
    team_ids: list[int] | None = None # NOT IMPLEMENTED
    min_total_goals: float | None = None # NOT IMPLEMENTED
    max_total_goals: float | None = None # NOT IMPLEMENTED
    feature_groups: list[str] | None = None # NOT IMPLEMENTED
```

### MatchRepository (`queries.py`)

```python
def get_historical_matches(
    self,
    min_date: datetime | None = None,
    max_date: datetime | None = None,
    tournament_id: int | None = None, # Single tournament only!
    require_results: bool = True,
) -> list[Match]:
    # Missing:
    # - tournament_ids: list[int] (for multiple tournaments)
    # - team_ids: list[int]
    # - min_goals/max_goals
    # - venue_filter (home/away/both)
    # - require_odds: bool (odds filtering is only in backtest, not training)
```

### Frontend (`types.ts`)

```typescript
interface TrainingConfig {
    // Existing fields (omitted for brevity — see Current Capabilities table)
    // Missing:
    tournamentIds?: number[]; // NOT IMPLEMENTED
    teamIds?: number[]; // NOT IMPLEMENTED
    minGoals?: number; // NOT IMPLEMENTED
    maxGoals?: number; // NOT IMPLEMENTED
    featureGroups?: string[]; // NOT IMPLEMENTED
    outcomeBalance?: boolean; // NOT IMPLEMENTED
}
```

---

## Hidden / Unexposed Capabilities

Several capabilities exist in the backend codebase but are **not exposed** in the API or frontend UI:

| Capability | Location | Status | Description |
|------------|----------|--------|-------------|
| **Ensemble training** | `pipeline.py` (TrainingConfig), `cli/commands/train.py` | CLI only | `use_ensemble` and `ensemble_types` fields exist in the internal `TrainingConfig` dataclass and CLI, but are absent from the API schema and frontend |
| **Auto class weighting** | `split.py:420-436` | Always-on, no user control | `get_class_weights()` computes inverse-frequency H/D/A weights automatically during training. Users cannot configure, override, or disable this |
| **Feature caching** | `pipeline.py` (TrainingConfig), `features/store.py` | Internal only | `use_feature_cache: bool = True` in internal config. Feature store with schema versioning exists but is not user-configurable |
| **Feature schema versioning** | `pipeline.py` (TrainingConfig) | Internal only | `feature_schema_version: str = "v1.0"` in internal config. Not exposed in API or UI |
| **Temporal split gap days** | `split.py:71` (TemporalSplitter) | Internal only | `gap_days` parameter prevents data leakage at train/val/test boundaries. Default is 0, not user-configurable |
| **Expanding window splitter** | `split.py:201-276` | Code exists, unused | `ExpandingWindowSplitter` class implemented but not wired into the training pipeline or API |
| **Season-aware splitter** | `split.py:279-387` | Code exists, unused | `SeasonAwareSplitter` class implemented but not wired into the training pipeline or API |
| **Custom hyperparameters** | `ml_operations.py:67` | API accepts, no UI | Backend accepts `hyperparameters: dict[str, Any]` and frontend Zod schema has it, but UI always sends `{}` |
| **Model tags** | `pipeline.py:79` (TrainingConfig) | Internal only | `tags: dict[str, str]` exists in internal config for model versioning/metadata, not exposed in API or UI |

---

## Consequences

### 1. Model Quality Issues
- **Overfitting to dominant patterns:** Cannot exclude outlier tournaments/teams that behave differently
- **Data leakage risk:** Cannot properly isolate competitions for cross-validation
- **Poor generalization:** Models trained on all data may not perform well on specific subsets

### 2. Research Limitations
- **Cannot perform ablation studies:** Cannot test which features/filters improve performance
- **No comparative analysis:** Cannot compare models trained on different subsets
- **No domain adaptation:** Cannot train specialized models for specific contexts

### 3. Operational Issues
- **Manual workarounds required:** Users must export data and train externally to get proper control
- **Wasted training time:** Must train on irrelevant data (e.g., lower divisions when interested only in top tier)
- **Version control problems:** Cannot document which data subsets were used for each model

---

## Recommended Priority Fixes

### Phase 1: Critical (High Impact, Low Effort)
1. **Multi-tournament filtering**
   - Extend `tournament_id` → `tournament_ids: list[int]`
   - Update UI to show tournament multi-select

2. **Team filtering**
   - Add `team_ids: list[int]` for matches involving specific teams
   - Add venue filter (home/away/both)

### Phase 2: Important (Medium Impact, Medium Effort)
3. **Goal-based filtering**
   - Add `min_total_goals` and `max_total_goals`
   - Filter on `home_score + away_score`

4. **Odds quality filter**
   - Make `require_odds` configurable
   - Add minimum bookmaker count

### Phase 3: Valuable (Lower Impact, Higher Effort)
5. **Feature group selection**
   - Expose feature pipeline configuration
   - Allow toggling feature groups on/off

6. **Outcome balancing controls**
    - Expose `get_class_weights()` as a configurable option (auto/custom/disabled)
    - Add option to oversample minority classes

---

## Workarounds (Current)

To achieve similar results today, users must:

1. **Manual database queries:** Export specific subsets to CSV, train externally
2. **Post-filtering:** Train on all data, then evaluate on specific subsets only
3. **Custom scripts:** Write Python scripts using the internal APIs directly
4. **Multiple models:** Train one model per tournament, manually exclude data

None of these are user-friendly or efficient.

---

## Conclusion

The current model training interface provides **only 30-40% of the necessary data selection capabilities** for professional ML workflows. While the basic temporal, volume, and model configuration controls are adequate for initial exploration, the lack of team, tournament, and feature selection severely limits:

- Model specialization for specific contexts
- Research into feature importance
- Quality control of training data
- Operational flexibility

**Recommendation:** Prioritize Phase 1 fixes (tournament and team filtering) as these provide the highest immediate value with relatively low implementation complexity.

---

## Related Files

- [`algobet/api/routers/ml_operations.py`](algobet/api/routers/ml_operations.py) - Backend training API
- [`algobet/predictions/data/queries.py`](algobet/predictions/data/queries.py) - Match data access
- [`algobet/predictions/training/pipeline.py`](algobet/predictions/training/pipeline.py) - Internal training pipeline & `TrainingConfig` dataclass
- [`algobet/predictions/training/split.py`](algobet/predictions/training/split.py) - Data splitting & class weights
- [`algobet/predictions/features/generators.py`](algobet/predictions/features/generators.py) - Feature generators (TeamForm, H2H, Odds, Temporal)
- [`algobet/predictions/features/store.py`](algobet/predictions/features/store.py) - Feature caching & schema versioning
- [`algobet/cli/commands/train.py`](algobet/cli/commands/train.py) - CLI training command (has ensemble options)
- [`frontend/components/models/TrainModelCard.tsx`](frontend/components/models/TrainModelCard.tsx) - Training UI
- [`frontend/components/models/types.ts`](frontend/components/models/types.ts) - Frontend `TrainingConfig` interface
- [`frontend/components/models/DataRangeSection.tsx`](frontend/components/models/DataRangeSection.tsx) - Date range controls
- [`frontend/components/models/DataSplitSection.tsx`](frontend/components/models/DataSplitSection.tsx) - Split ratio controls
- [`frontend/components/models/BasicSettings.tsx`](frontend/components/models/BasicSettings.tsx) - Model type, description, activate, tuning, calibration toggle
- [`frontend/components/models/TrainingSettingsSection.tsx`](frontend/components/models/TrainingSettingsSection.tsx) - Advanced settings: seed, early stopping, tuning trials, calibration method
