# Feature Importance & Orthogonality Review

**Date:** 2026-05-12
**Scope:** 8 XGBoost models from the 2026-05-12 debug/feature-engineering session.
**Method:** gain importance per model → cross-model stability → Pearson correlation
on a 3,802-match EPL panel (2014-01 → 2024-08) generated through the winner's
feature pipeline (`xgboost_20260512_201906`).

Backing artifacts in `scripts/throw-away/`:
`_aggregate_gain.csv`, `_survival.csv`, `_classification.csv`,
`_correlation_pairs.csv` (winner-14), `_raw_correlation_pairs.csv` (raw-167),
`_edge_partners.csv`.

---

## TL;DR

- **12 features carry the model's signal.** They survive feature selection in 6–8
  of 8 runs, sit in the top-quartile of gain, and have low collinearity inside
  the surviving set.
- **Within the winner's 14-feature set, only one pair has |r| ≥ 0.70** (clean-sheet
  3-window vs 10-window, r=0.76). No hard collinearity (|r| ≥ 0.85). Selection
  did its job.
- **In the *raw* 167-feature panel there are 52 hard-collinear pairs.** This is
  why dropping selection (models `194212`, `195921`) tanks Kappa — feature
  selection is doing structural deduplication, not just noise filtering.
- **The new `player_quality` group is collinear, not orthogonal.** All
  `xi_stability_*` variants pair-correlate at r ≈ 0.89, and the diff/team forms
  redundantly cover the same signal. None survived selection in the winner.

---

## 1. The 12 Edge features

Top-quartile mean gain across the 8 models, surviving in ≥6 of 8 runs.

| Feature | Survives | Mean gain | CV | In winner |
|---|---:|---:|---:|:---:|
| `away_ppda_against_avg_3` | 8/8 | 0.0577 | 0.55 | ✓ |
| `draw_rate_diff_season` | 8/8 | 0.0574 | 0.55 | ✓ |
| `home_draw_rate_season` | 8/8 | 0.0563 | 0.54 | ✓ |
| `is_season_mid` | 7/8 | 0.0548 | 0.59 | ✓ |
| `home_xg_for_avg_3` | 7/8 | 0.0542 | 0.59 | ✓ |
| `home_deep_completions_for_avg_3` | 7/8 | 0.0537 | 0.59 | — |
| `h2h_away_win_rate` | 8/8 | 0.0528 | 0.55 | ✓ |
| `low_scoring_matchup_5` | 7/8 | 0.0519 | 0.60 | — |
| `away_win_rate_season` | 6/8 | 0.0513 | 0.66 | ✓ |
| `home_clean_sheet_rate_10` | 7/8 | 0.0508 | 0.61 | ✓ |
| `away_draw_rate_season` | 6/8 | 0.0504 | 0.67 | — |
| `away_clean_sheet_rate_10` | 6/8 | 0.0491 | 0.67 | — |

**Reading the columns.** `Survives` counts the number of the 8 runs where the
feature has non-zero gain (note: 2 of the 8 — `194212` and `195921` — were
trained without feature selection, so they include nearly all 167 features at
low gain; the meaningful filter is "survives in the 6 selection runs"). `CV =
std/mean` of normalised gain across models; CV in the 0.5–0.7 range here just
reflects the different selection counts (zero gain in two recipes drags both
mean and std), not instability.

### Signal these features capture

| Group | What it measures | Why orthogonal |
|---|---|---|
| `*_draw_rate_season`, `draw_rate_diff_season` | Per-team season DRAW frequency | Only feature group that targets the DRAW class directly — without it, the model collapses to HOME |
| `away_ppda_against_avg_3` | Pressing intensity opponent applied to away team in last 3 | Defensive-pressure signal not captured by form or xG |
| `home_xg_for_avg_3` | Recent attacking quality | Quality (not result) — diverges from form |
| `home_deep_completions_for_avg_3` | Penetrating passes per match | Style signal — uncorrelated with goals |
| `h2h_away_win_rate` | Head-to-head, away perspective | Stable matchup memory |
| `is_season_mid` | Calendar phase indicator | Captures the EPL mid-season congestion / form-volatility regime |
| `low_scoring_matchup_5` | Recent low-scoring tendency | Predictive of DRAW outcomes |
| `*_clean_sheet_rate_10` | Defensive solidity over 10 | Defence axis |
| `away_win_rate_season` | Season-level strength proxy | Strength axis |

The four axes — **draw propensity, pressing/defensive style, attacking style,
season-level strength** — each have at least one strong feature carrying it,
which is what the model needs.

---

## 2. Collinearity inside the winner (14 features)

Only **one** pair above the soft threshold:

| A | B | r |
|---|---|---:|
| `home_clean_sheet_rate_5` | `home_clean_sheet_rate_10` | 0.758 |

No |r| ≥ 0.85 in the selected set. **This is exactly what we want from
feature selection** — the residual r = 0.76 is XGBoost using both windows to
read short-term vs long-term defensive trajectory.

---

## 3. Hard-collinear clusters in the *raw* 167-feature panel

These pairs have |r| ≥ 0.85 — selection is the only thing stopping them from
fighting for the same split.

| Cluster (pick one, drop the rest) | Representative pairs | r |
|---|---|---:|
| **Season-progress** (`days_from_season_start`, `season_progress`, `home_matches_played`, `away_matches_played`, `*_goals_against_season`) | days↔season_progress | 0.997 |
| | matches_played↔matches_played | 0.998 |
| **Strength proxies** (`away_win_rate_season`, `away_points_per_game`, `away_league_position`, `away_position_normalized`) | win_rate_season↔points_per_game | 0.977 |
| | league_position↔position_normalized | 1.000 |
| **Tactical-window pairs** (3-game vs 5-game of same metric) | ppda_against_avg_3↔_5 | 0.903 |
| | xg_for_avg_3↔_5 | 0.860 |
| | deep_completions_for_avg_3↔_5 | 0.905 |
| | ppda_for_avg_3↔_5 | 0.858 |
| **Derived duplicates** | `goal_convergence_N` ↔ `offensive_balance_N` | **1.000** |
| **Calendar** | `day_of_week` ↔ `is_weekend` | 0.908 |
| **H2H trio** | `h2h_home_win_rate` ↔ `h2h_recency_weighted_home_points` ↔ `h2h_recent_home_form` | 0.88–0.97 |
| **xi_stability** (player_quality group) | xi_stability_3 ↔ _5 (home, away, diff) | 0.89 each |
| **Position vs Points** | `position_diff` ↔ `points_per_game_diff` ↔ `points_diff` | 0.87–0.92 |
| **Enriched coverage** | `*_enriched_match_coverage_3` ↔ `_5` | 0.94 |
| **Strength parity** | `strength_parity` ↔ `strength_parity_5` ↔ `win_rate_gap_5` | 0.86–0.94 |

The pair `goal_convergence_N` ↔ `offensive_balance_N` with **r = 1.000** is a
code smell — two features almost certainly computed from the same formula.
Worth checking the generators and deleting one.

---

## 4. For each Edge feature: its strongest collinear shadow

This is the "if you dropped this feature, what would the model fall back on?"
view. Anything ≥ 0.85 means the Edge feature has a near-perfect substitute
sitting in the raw panel — dropping it would barely hurt.

| Edge feature | Shadow partner | r | Substitutable? |
|---|---|---:|:---:|
| `away_ppda_against_avg_3` | `away_ppda_against_avg_5` | 0.896 | **Yes** |
| `home_xg_for_avg_3` | `home_xg_for_avg_5` | 0.860 | **Yes** |
| `home_deep_completions_for_avg_3` | `_avg_5` | 0.905 | **Yes** |
| `away_win_rate_season` | `away_points_per_game` | **0.977** | **Yes (basically the same feature)** |
| `away_win_rate_season` | `away_league_position` | 0.875 | Yes |
| `h2h_away_win_rate` | `h2h_goal_diff_avg_from_home_perspective` | 0.816 | Soft |
| `home_clean_sheet_rate_10` | `home_clean_sheet_rate_5` | 0.758 | Soft |
| `away_clean_sheet_rate_10` | `away_clean_sheet_rate_5` | 0.764 | Soft |
| `is_season_mid` | `month` | 0.751 | Soft (binary vs continuous) |
| `low_scoring_matchup_5` | `low_scoring_matchup_10` | 0.727 | Soft |
| `draw_rate_diff_season` | (no |r|≥0.70 partner) | — | **No — unique** |
| `home_draw_rate_season` | (no |r|≥0.70 partner) | — | **No — unique** |

**Truly orthogonal Edge features** (no |r| ≥ 0.70 partner anywhere in the raw
panel): `draw_rate_diff_season`, `home_draw_rate_season`. These two are
load-bearing — the only ones the model literally couldn't substitute. They
are also the only features in the entire set that directly target the DRAW
class. **Losing them would re-trigger class collapse.**

Everything else on the Edge list has at least one moderately strong shadow —
the model could replace any single feature without much pain, but it does need
*one* representative from each axis.

---

## 5. Verdict on the `player_quality` group

Features introduced: `home/away_xi_stability_3`, `home/away_xi_stability_5`,
`xi_stability_diff_3`, `xi_stability_diff_5`, `home/away_starting_pool_5`.

Findings:

1. **Highly collinear internally.** `xi_stability_3 ↔ _5` correlate at
   r ≈ 0.89 for every variant (home/away/diff). The 3 vs 5 window adds nothing.
2. **Did not survive selection in the winner.** None of the `xi_*` features
   appear in 201906's 14-feature set. Only `away_xi_stability_3` appears at
   all, and only in the two no-selection runs at ~12th–13th rank.
3. **Not orthogonal to existing features.** The signal `xi_stability` carries
   — "how much has the lineup changed" — overlaps with rotation-driven dips
   that already show in `*_form_*` and `*_xg_*` features. The information
   isn't new; it's a noisier proxy.

**Recommendation:** drop `player_quality` from the feature_groups list, *or*
reduce it to a single window (only `_5`, drop `_3`) and the diff form only,
and re-test. As specified today it adds 8 collinear columns and zero
discriminative gain in the winning recipe.

---

## 6. Feature taxonomy

(See `_classification.csv` for the full table. Counts reflect ≥1 survival in
any of the 8 models.)

| Label | Count | Meaning |
|---|---:|---|
| **Edge** | 12 | Carries load-bearing signal across runs |
| **Mid** | 4 | Top half of gain, survives 4–5 runs |
| **Recipe-dependent** | 13 | Appears only in some recipes — XGBoost picks them as substitutes for Edge features when selection cuts differently |
| **Noise** | ~135 | Appears only in the two no-selection runs at low gain |

The "Noise" group isn't pure noise — most of those features are *collinear
shadows* of Edge features that XGBoost picks up only when it's forced to use
all 167. They are eliminated cleanly by `min_samples_per_feature=75`.

---

## 7. Recommended feature set for the next run

Drop ~50 columns of duplicates. Either:

**Aggressive (recommended):** restrict `feature_groups` to
```
["team_form", "head_to_head", "temporal", "standings",
 "enriched_stats", "draw_signals", "matchup_interaction"]
```
and remove `player_quality`. This matches the winner's effective set and
avoids selection wasting trials on collinear `xi_stability` features.

**Surgical:** keep all groups but explicitly remove inside the generators:
- `goal_convergence_*` (duplicate of `offensive_balance_*`)
- `*_position_normalized` (perfect mirror of `*_league_position`)
- `*_points_per_game` (r=0.98 with `*_win_rate_season`)
- All `xi_stability_*` and `xi_stability_diff_*` 3-window variants (keep `_5`
  only, or drop entirely)
- `is_weekend` (r=0.91 with `day_of_week`)
- One of `season_progress` / `days_from_season_start` / `*_matches_played`
  (pick one, they're all 0.98+)
- One window of every `_avg_3` / `_avg_5` pair (likely keep the `_5` since
  it survives more often in the unselected models; the winner already prefers
  `_3` so reconsider per metric)

Either route, expect:
- ~110 raw features instead of 167
- Faster tuner (smaller search space)
- More stable selection (less for the threshold to choose between)
- Same or slightly better Kappa / ROI — the information content is unchanged.

---

## 8. Caveats

- Gain importance is a XGBoost-internal metric. SHAP would give a per-prediction
  attribution; a follow-up could confirm these rankings.
- The correlation panel uses pre-2024 EPL training data. Cross-tournament or
  cross-season correlation structure may differ, but since all current models
  are EPL-only this is the right window.
- Two of the 8 models (`194212`, `195921`) were trained *without* feature
  selection, so they inflate the "n_models" count for many low-gain features.
  The Edge label is robust to this — it requires top-quartile *mean* gain,
  which the inflated-low-gain features can't reach.
- **Post-fit vs probe importance.** The importance numbers analysed above
  come from each model's *post-selection* fit (the booster trained on the
  surviving 14 features). The probe model that *drove* selection assigns
  gain on all 167 raw features and is never persisted. Post-fit gain is a
  fair-to-good proxy for relative importance among surviving features but
  cannot rank features that were filtered out.

---

## 9. Stability test: guard-free run (model `xgboost_20260512_211458`)

To validate that the 14 Edge features are XGBoost's natural choice rather
than artifacts of the family-retention guards, a guard-free training run was
executed with `min_draw_features=0`, `min_away_features=0`,
`min_enriched_or_coverage=0`, `min_low_scoring_features=0`. All other
settings matched the winner's recipe.

### Result: feature set is **identical** to the winner

The guard-free selector picked the same 14 features as `201906`:

```
is_season_mid, home_draw_rate_season, draw_rate_diff_season,
h2h_draw_boost, h2h_away_win_rate, away_win_rate_season,
away_away_form_5, home_clean_sheet_rate_5, home_clean_sheet_rate_10,
away_ppda_against_avg_3, home_xg_for_avg_3,
away_enriched_match_coverage_3, away_deep_completions_against_avg_5,
home_ppda_against_avg_5
```

**Implications for §1–§7:**

| Earlier claim | Status |
|---|---|
| 14 Edge features are the model's natural picks | ✅ **Confirmed.** Guards do not change the selected set. |
| Family-retention guards are load-bearing for selection | ❌ **Wrong.** Guards are redundant for this recipe — probe importance + correlation prune + `min_samples_per_feature=75` already produce the same set. |
| `draw_rate_*` features are force-injected by `min_draw_features=3` | ❌ **Wrong.** They survive on probe importance alone. They genuinely carry signal. |
| `player_quality` is collinear and not selected | ✅ Confirmed in the guard-free run too. |

### But: metrics diverge sharply on identical features

| Metric | Winner `201906` (guards on) | Guard-free `211458` (guards off) |
|---|---:|---:|
| Test accuracy | 40.3% | 41.1% |
| Cohen's Kappa | +0.037 | +0.029 |
| **HOME recall** | 47% | **72%** |
| **DRAW recall** | **12.5%** | **5.2%** (5 of 96) |
| AWAY recall | ~40% | 28% |
| Total bets | ~80 | 375 |
| **ROI** | **+4.79%** | **−1.35%** |
| Max drawdown | 1.80 | 1.56 |

Same 14 features, same `outcome_balance=0.3`, same 50 Optuna trials, same
seed (42). The two models' booster hyperparameters end up in different
regions of the search space because Optuna's sampler is non-deterministic
under different objective values, and the validation-loss landscape is
multi-modal enough that the "best trial" is effectively a draw between
several plausible configurations.

### The real variance source: hyperparameter sampling, not features

The session's "180433 vs 184948 vs 201906" comparison was reading
hyperparameter-sampling noise as if it were feature-engineering signal.
Every recipe in the session picked roughly the same 14 features; what
differed was *which Optuna trial won*, which moved DRAW recall by ±10pp
and ROI by ±6pp.

This is a **stability problem, not a feature problem**. The fix is
multi-seed ensembling or a different model architecture (see §10), not
more feature engineering.

---

## 10. Novel approaches to address the instability

Ranked by `expected payoff / engineering effort`. Each addresses a
specific failure mode documented above.

### A. Multi-seed booster ensemble (cheapest stability win)

**What:** Train N=10 boosters with different `random_seed` on the same
14-feature set and recipe. Average predicted probabilities at inference.

**Why it works:** the variance the user just observed (ROI ±6pp on
identical features and recipe) is dominated by per-trial Optuna
randomness. Averaging 10 independent fits cuts that variance by ~√10.

**Effort:** ~1 day. The training runner already supports model
registries; loop the existing pipeline 10× with different seeds and
average at `predict_proba` time. No new model class required.

**Risk:** none — strictly improves stability. May reduce peak performance
(no single booster will be as lucky as `201906`), but eliminates
catastrophic Optuna draws like `211458`.

**Verifiable improvement:** report Sharpe and ROI across 5 held-out folds
of the test season. Multi-seed ensemble's std should drop ~3×.

---

### B. Hierarchical / multi-league model

**What:** Train one model on **all available leagues** (EPL, La Liga,
Bundesliga, Serie A, Ligue 1, etc.) with `league_id` as a categorical
input. Sample size grows from ~3040 EPL matches to 15k+ matches.

**Why it works:** the session summary explicitly flagged
"distribution shift between adjacent EPL seasons is large enough to break
a validation-tuned blend weight". With 3040 training rows and 14
features, the per-season noise floor is high. Pooling leagues with a
league-id feature lets the model share the universal patterns
(home advantage, draw propensity at low-scoring matchups, etc.) while
letting league-specific patterns surface through the categorical split.

**Effort:** ~3–5 days. Requires `MatchRepository.get_historical_matches`
to support `tournament_ids=None` (all leagues), and league-id needs to
become a categorical column the feature pipeline emits. The existing
`SeasonAwareSplitter` already handles multi-tournament splits — just
needs verification.

**Risk:** model may overfit to the largest league. Mitigate with sample
weighting inversely proportional to league size.

**Verifiable improvement:** test on a held-out league the model was
never trained on (e.g., train on all-but-Bundesliga, test on
Bundesliga). Current EPL-only models cannot do this at all.

---

### C. Dixon-Coles as primary model (not as a blend)

**What:** Replace XGBoost as the primary classifier with a Dixon-Coles
score-grid model: per-team attack and defence parameters fit by Poisson
MLE, with a low-score correlation correction. Probabilities for
{H, D, A} come from summing the score-grid.

**Why it works:** the fundamental problem of every model in the session
is that **classification log-loss has no idea what a "DRAW" is** — it
just sees three opaque labels. Dixon-Coles models *goals*, which is the
underlying mechanism. The DRAW probability emerges naturally from low
expected total goals + similar attack/defence — there is no class
collapse possible because the model doesn't see classes.

The session's "Fix 9: DC blend FAILED" tried DC as a calibrator on top
of XGBoost. That's the wrong way around — DC should be the engine, with
XGBoost optionally feeding it engineered features (e.g., expected goals
from xG history).

**Effort:** medium. Codebase already has `dixon_coles_epl.joblib`, so
the model class exists. Wiring it as a primary `MatchPredictor`
implementation alongside `XGBoostPredictor` is ~3 days; harder is
adding hyperparameters around the time-decay (`xi`) and low-score
correction (`rho`) and running a proper temporal CV.

**Risk:** DC is structurally less expressive than XGBoost on tabular
features. Best used as the second model in a 2-model ensemble (DC for
goal-based reasoning, XGBoost for everything else), not as a standalone
replacement.

**Verifiable improvement:** test DRAW recall directly — DC should
deliver 25–30% naturally, vs current 5–17%. Calibration (ECE) should
also drop from 0.22 to <0.10.

---

### D. Conformal prediction wrapper (turn instability into abstention)

**What:** wrap any of the above models in a conformal predictor
calibrated on a held-out validation set. Inference returns a *prediction
set* with calibrated coverage (e.g., "95% sure the outcome is in
{H, D}"). Bet only when the set contains a single outcome with positive
EV.

**Why it works:** the current betting policy bets every match
(`min_edge=0`, single-best-edge). On uncertain matches, this is just
noise. Conformal lets the model say "I don't know" by returning
singleton-or-empty sets at chosen miscoverage `α`. Bet count drops, but
ROI on remaining bets should rise.

**Effort:** ~2 days. Add a `ConformalCalibrator` class (no external
library needed — algorithm is ~30 lines) and plumb `bet_if_singleton`
into the betting metric.

**Risk:** none structural. Worst case: too many singletons, behaviour
matches current. Best case: bet count drops to ~150/season with ROI
~8–10%.

**Verifiable improvement:** compute Sharpe with and without the
abstention rule. Should rise substantially even if total profit falls.

---

### E. Multi-task learning across markets (1X2 + Over/Under + BTTS)

**What:** train a single model to predict three correlated targets
simultaneously: match result (1X2), total goals over/under 2.5, and
both-teams-to-score (BTTS). Use shared-trunk neural net or LightGBM's
multi-output mode.

**Why it works:** the same underlying state (team strengths, recent
form, defensive solidity) drives all three markets. Training on all
three regularises the shared features — `draw_rate_diff_season` should
also predict low-scoring matches (it does in the correlation table).
This is equivalent to ~3× the effective training signal per feature.

**Effort:** medium-high (~5 days). Requires extending the prediction
schema to hold three predictions per match, adding O/U and BTTS labels
to the training data, and either a custom multi-output XGBoost loss or
switching to a small PyTorch MLP.

**Risk:** the codebase is currently 1X2-only. This is a meaningful
architectural change. Worth it if you intend to bet on those markets
anyway; speculative otherwise.

**Verifiable improvement:** 1X2 metrics should not degrade. O/U and
BTTS predictions become available as a side benefit, with calibration
quality comparable to or better than dedicated single-task models.

---

### F. Bayesian booster ensemble (over hyperparameters)

**What:** instead of `tuning_trials=50` followed by "pick the best",
keep all 50 trials and weight each by `exp(-val_log_loss / T)`.
Aggregate predictions across the weighted posterior.

**Why it works:** this is approach A taken further — the 50 Optuna
trials *already* explored hyperparameter space, but only one wins. The
others contain useful diversity that's currently thrown away. Bayesian
model averaging captures the epistemic uncertainty across the trial
distribution.

**Effort:** ~2 days. Modify the tuner to persist all 50 fitted models
(disk cost ~150MB per training run) and add an `EnsemblePredictor`
that loads them and averages at inference.

**Risk:** disk + inference cost. At 50 boosters × 14 features, inference
is still <50ms/match — negligible.

**Verifiable improvement:** same metric structure as approach A but
with strictly more diversity (every Optuna trial is a unique
hyperparameter point, vs identical-recipe-different-seed in A).

---

### G. Direct ROI optimisation via reinforcement learning

**What:** train a policy network to *bet* directly. Reward = realised
profit × Kelly fraction. Use REINFORCE or PPO over a sequence of
historical matches, with the betting decision as the action.

**Why it works:** every model in the session optimises log-loss or
class-weighted cross-entropy. The actual objective is **profit under
uncertainty given fluctuating market odds**. These two objectives can
diverge — a model with worse log-loss can have better ROI if it's
overconfident on the right matches.

**Effort:** high (~2 weeks). Requires designing the action space
(skip/bet, which outcome, stake fraction), the state representation,
and a reward shaping that handles delayed rewards and variance.

**Risk:** high. Sample-inefficient relative to supervised learning.
Likely needs >50k training matches to stabilise — feasible only with
the multi-league dataset from approach B.

**Verifiable improvement:** if it works, +5–10pp ROI over best
supervised approach. If it doesn't, you've spent 2 weeks. **Do this
last**, only after A/B/C are in place.

---

### Recommended sequence

1. **A first** (1 day) — buys stability immediately, makes every
   subsequent comparison meaningful by removing Optuna noise.
2. **B second** (5 days) — addresses the small-sample-per-league
   problem that's the root of distribution shift.
3. **D third** (2 days) — bolts onto A or A+B; gives betting policy a
   clean abstention mechanism.
4. **C fourth** (3 days) — adds an orthogonal signal (goal-based) to the
   tabular signal you already have. Best as a 2-model ensemble vote.
5. **E or F** if budget remains. **G only if you've exhausted everything
   else and have a multi-league dataset.**

After step 1 alone, the report's Edge-feature claims become
re-verifiable on stable metrics; after step 2 the claims become
generalisable across leagues. Those are the two things the current
single-model EPL-only setup cannot give you.
