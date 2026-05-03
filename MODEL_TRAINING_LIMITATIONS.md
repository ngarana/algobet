# Model Training Limitations

> **Status Update**: As of recent development, the vast majority of historical limitations in model training capabilities have been **fully resolved** both in the backend API and the frontend UI. This document serves to track the history of these capabilities.

## Currently Supported Capabilities

The model training pipeline and UI now provide a comprehensive set of configuration options, including all items previously listed as limitations:

### Data Selection & Filtering
- **Multi-Tournament Filtering**: Train on matches from specific tournaments or combinations of tournaments.
- **Team Filtering**: Filter training data to only include matches involving specific teams.
- **Venue Filtering**: Train exclusively on home matches, away matches, or both.
- **Goal Range Filters**: Restrict training data to matches with total goals within a specific min/max range.
- **Odds Requirement**: Filter to include only matches with betting odds available.
- **Data Range Settings**: Control minimum match requirements and explicit date ranges.

### Model Architecture
- **Multiple Algorithms**: XGBoost, LightGBM, and Random Forest.
- **Ensemble Training**: Combine multiple model types (e.g., XGBoost + LightGBM) to create an `EnsemblePredictor` that averages predictions for better robustness.
- **Hyperparameter Tuning**: Automated search using Optuna.
- **Custom Hyperparameters**: Manual configuration of algorithm-specific parameters (e.g., `max_depth`, `learning_rate` for XGBoost).

### Data Splitting Strategies
The pipeline supports three distinct data splitting strategies:
1. **Temporal (Default)**: Chronological split with configurable `gap_days` to prevent data leakage between train/val/test sets.
2. **Expanding Window**: Rolling window cross-validation (configurable `min_train_size`, `step_size`, etc.).
3. **Season-Aware**: Splitting by complete football seasons to ensure no partial seasons span across sets.

### Feature Engineering
- **Feature Group Selection**: Toggle specific groups of features (Team Form, Head-to-Head, Market Odds, Temporal).
- **Outcome Balancing**: Enable inverse-frequency class weighting to apply higher weights to minority outcomes (like draws or away wins).

### Organization
- **Model Tags**: Attach arbitrary key-value metadata tags to trained models.
- **Descriptions**: Attach human-readable descriptions to model runs.

---

## Historical Resolved Items

The following items were historically tracked as limitations but have since been resolved:

1. **Tournament selection**: Fully supported.
2. **Team filtering**: Fully supported.
3. **Ensemble models**: Fully exposed via API and frontend UI.
4. **Alternative splitters**: `ExpandingWindowSplitter` and `SeasonAwareSplitter` are fully integrated and selectable.
5. **Hyperparameter UI**: Model-aware forms exposed in frontend.
6. **Feature sub-selection**: Configurable via `FeatureGroupsSection`.
7. **Balancing control**: Toggleable in frontend.
8. **Tags**: Sent via API and stored.
