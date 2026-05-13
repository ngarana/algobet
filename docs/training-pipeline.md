# Model Training Pipeline

This document describes the execution sequence and logic of the Algobet model training pipeline, specifically focusing on the feature selection and model training orchestration.

## Execution Sequence

When a training payload is issued, the pipeline executes in a strict order defined in `feature_selection.py` and `feature_selection_pipeline.py`.

### 1. Initial XGBoost Probe
The pipeline first probes an XGBoost model trained on all raw features (approximately 167 features). This stage produces a **Gain Importance** dictionary used for initial ranking.

### 2. Correlation Pruning
Features are pruned based on correlation to reduce redundancy.
- **Default Threshold:** `max_correlation=0.90`.
- **Behavior:** If two features have a correlation $|r| \geq 0.90$, the one with lower importance is dropped in favor of its higher-variance peer.

### 3. Threshold Filtering
A secondary filter is applied based on normalized gain.
- **Criteria:** Retain features where `normalized_gain >= 0.01`.

### 4. Sample-to-Feature Capping
To prevent overfitting on small datasets, a cap is applied based on the number of training samples.
- **Rule:** `min_samples_per_feature = 75`.
- **Example:** With 3,040 training rows, the model is capped at a maximum of 40 features.

### 5. Family-Retention Guards
Certain "critical" features are force-injected into the final set regardless of their importance score to ensure balanced predictive signals. These guards inject up to:
- **Draw Features (min: 3):** Matches "draw_rate", "h2h_draw", etc.
- **Away Features (min: 3):** Matches "away_win", "away_away", etc.
- **Enriched Coverage (min: 5):** Statistics like xG, PPDA, deep completions, etc.
- **Low Scoring (min: 2):** Features related to low-scoring match contexts.

> [!NOTE]
> There are a minimum of 13 forced slots reserved for these family-retention guards before natural importance rankings are even considered.

### 6. Pipeline Refit
Once the final feature set is selected, the transformers are rebuilt on the surviving features, and the final production model is trained.

---

## Interpreting Feature Importance

It is important to distinguish between **Probe Importance** and **Post-Selection Importance**.

### Post-Selection Importance
The importance extracted from the final `model.pkl` reflects the gain *after* the model has been refit on the reduced feature set (e.g., the surviving 14 features).

Because XGBoost spreads gain across the available features, these surviving features often show a "uniform-ish" importance (e.g., ~0.07 normalized gain each).

### Probe Importance
The "true" ranking signal used to decide which features to keep lives in the **Probe Model**, which is used during step 1 but is typically not persisted as the primary model artifact.

---

## Code References

- **Selection Logic:** [feature_selection.py:183](file:///home/arch/Coding/algobet/algobet/ml/feature_selection.py#L183)
- **Pipeline Orchestration:** [feature_selection_pipeline.py:113](file:///home/arch/Coding/algobet/algobet/ml/feature_selection_pipeline.py#L113)
- **Retention Guards:** [feature_selection.py:251-297](file:///home/arch/Coding/algobet/algobet/ml/feature_selection.py#L251-L297)
