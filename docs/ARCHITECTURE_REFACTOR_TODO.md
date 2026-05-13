# Architecture Refactor TODO

> Generated from `audit_architecture_antipatterns.py` (min-lines: 500).
> **Guiding principle:** No module is reviewed in isolation. Treat antipatterns as **systemic symptoms** and resolve them through cross-cutting architectural changes rather than local clean-ups.

---

## 1. Extract a Thin Router / Fat Service Boundary (Systemic KISS + God Module)

**Symptoms across routers:**
- `algobet/api/routers/ml_operations.py` — 1,063 lines, 17 top-level defs; `run_backtest` (287 lines), `run_training` (146 lines), `run_calibrate` (165 lines).
- `algobet/api/routers/scraping.py` — 1,012 lines, 23 top-level defs; `run_scraping_job` (162 lines).
- `algobet/api/routers/matches.py` — `list_matches` (101 lines).

**Root cause:** Business logic, orchestration, and HTTP framing are co-located in endpoint handlers. This repeats in every large router.

**Actions:**
- [x] Introduce a strict **Controller → Orchestrator → Domain** rule: routers may only parse input, call an orchestrator/use-case, and shape the response.
- [x] Extract `MLOperationsOrchestrator` from `ml_operations.py` into `algobet/services/ml_ops/` (or similar domain package). Move `run_backtest`, `run_training`, `run_calibrate`, and `get_backtest_detail` logic there.
- [ ] Extract `ScrapingOrchestrator` from `scraping.py` router into `algobet/services/scraping_orchestrator.py`. Move `run_scraping_job`, `scrape_results`, `scrape_by_date` logic there.
- [x] Extract `MatchQueryOrchestrator` from `matches.py` for `list_matches` filtering/pagination logic.
- [x] Add an ARCHITECTURE_DECISION_RECORD (ADR) documenting the router slimness policy to prevent regression.

---

## 2. Decompose God Classes in the Service Layer (Systemic God Class)

**Symptoms across services:**
- `ScrapingService` — 982 lines, 25 methods (`scraping_service.py`).
- `AnalysisService` — 686 lines, 5 methods (`analysis_service.py`); `run_backtest` (220 lines), `find_value_bets` (183 lines), `calibrate_model` (200 lines).
- `OddsPortalScraper` — 767 lines, 19 methods (`scraper.py`).
- `SoccerDataImporter` — 672 lines, 16 methods (`importers/soccerdata_importer.py`).

**Root cause:** Classes accumulate responsibilities because there is no internal packaging boundary beneath the service class.

**Actions:**
- [ ] Split `ScrapingService` into cohesive collaborator classes under `algobet/services/scraping/`:
  - `UpcomingScraper`, `ResultScraper`, `RangeScraper`, `MatchPersister` (extract `_save_upcoming_matches` and `_save_result_matches`).
- [x] Split `AnalysisService` into `algobet/services/analysis/`:
  - `BacktestRunner`, `ValueBetFinder`, `ModelCalibrator`.
- [x] Split `OddsPortalScraper` into `algobet/scraping/` sub-modules:
  - `PageNavigator`, `MatchExtractor`, `UpcomingMatchExtractor`.
- [ ] Split `SoccerDataImporter` into `algobet/importers/soccerdata/`:
  - `ScheduleImporter`, `StatsEnricher`, `DataNormalizer`.
- [ ] For each decomposition, keep a thin facade class that delegates to the new collaborators so existing call-sites can migrate incrementally.

---

## 3. Unify the Scraping Subsystem (Systemic DRY + KISS)

**Symptoms:**
- `scraping_service.py` has **15 repeated 8-line blocks** (highest duplication in scan).
- `algobet/api/routers/scraping.py` has **8 repeated 8-line blocks**.
- Both modules also suffer from deep nesting (depth 5) and God Module/Class issues.

**Root cause:** The scraper service, the scraper class, and the router each re-implement similar fetch-parse-persist sequences without shared abstractions.

**Actions:**
- [ ] Before splitting `ScrapingService`, **first extract shared helpers** for the duplicated 8-line blocks (likely session/response handling, retry loops, or error normalization).
- [ ] Create `algobet/scraping/common.py` with:
  - `fetch_with_retry(session, url, **kwargs)`
  - `parse_odds_html(html) -> Iterator[RawOddsRow]`
  - `normalize_odds_row(row) -> MatchDict`
- [ ] Migrate `scraper.py` (`OddsPortalScraper`) and `scraping_service.py` (`ScrapingService`) to consume these shared primitives.
- [ ] Ensure `algobet/api/routers/scraping.py` delegates entirely to the orchestrator (see #1) so it no longer contains inline scraping logic that duplicates the service layer.

---

## 4. Consolidate Data Access & Repository Pattern (Systemic God Class)

**Symptoms:**
- `MatchRepository` — 588 lines, 12 methods (`predictions/data/queries.py`); `get_historical_matches` (98 lines).
- `ReportGenerator` — 531 lines, 13 methods (`predictions/evaluation/reports.py`).

**Root cause:** Repository classes grow because they handle SQL construction, filtering, pagination, and reporting data aggregation in one place.

**Actions:**
- [x] Decompose `MatchRepository` into:
  - `MatchQueryBuilder` (SQL/filter construction)
  - `HistoricalMatchProvider` (complex `get_historical_matches` logic)
  - Keep `MatchRepository` as a thin coordinator implementing the public interface.
- [ ] Decompose `ReportGenerator` into:
  - `MarkdownReportRenderer`, `HtmlReportRenderer`, `MetricsAggregator`
- [ ] Audit whether `queries.py` and `reports.py` share common aggregation logic; if so, extract a `algobet/predictions/evaluation/metrics_core.py` module.

---

## 5. Simplify Feature & Training Modules (Systemic God Class / God Module)

**Symptoms:**
- `FeaturePipeline` — 436 lines, 22 methods (`predictions/features/pipeline.py`).
- `classifiers.py` — 883 lines, 40 top-level functions/methods; `MatchPredictor` (173 lines, 12 methods); `fit` (91 lines); `compute_adaptive_regularization` (83 lines); 5 repeated 8-line blocks.

**Root cause:** Feature engineering, model fitting, and regularization are packed into single files/classes.

**Actions:**
- [ ] Extract `FeaturePipeline` steps into `algobet/predictions/features/steps/`:
  - `EncodingStep`, `ScalingStep`, `WindowingStep`, `ImputationStep`
- [ ] Split `classifiers.py` into a package `algobet/predictions/training/classifiers/`:
  - `base.py`, `match_predictor.py`, `regularization.py`, `validation.py`
- [ ] Extract the 5 repeated blocks in `classifiers.py` into shared utility functions (likely CV-split or metric-computation snippets).

---

## 6. Frontend Component & Page Decomposition (Systemic God Module)

**Symptoms:**
- `frontend/app/backtest/page.tsx` — 636 lines, 28 function-like blocks.
- `frontend/components/scraping/FetchDialog.tsx` — 665 lines, 35 function-like blocks, 2 repeated 8-line blocks.

**Root cause:** Pages and dialogs inline data fetching, state management, and presentational sub-components.

**Actions:**
- [x] Extract presentational sub-components from `backtest/page.tsx` into `frontend/components/backtest/`:
  - `BacktestForm`, `BacktestResultsPanel`, `BacktestParamsCard`
- [x] Extract custom hooks for data fetching and form state into `frontend/hooks/useBacktest.ts`.
- [x] Extract presentational sub-components from `FetchDialog.tsx` into `frontend/components/scraping/`:
  - `FetchFormSection`, `FetchProgressIndicator`, `FetchConfirmationFooter`
- [ ] Extract the 2 repeated 8-line blocks in `FetchDialog.tsx` into shared helpers (likely form-field definitions or validation snippets).

---

## 7. Systemic DRY Remediation — Shared Utilities (Cross-cutting)

**Symptoms:**
- `matches.py`: 10 repeated 8-line blocks
- `scraping_service.py`: 15 repeated 8-line blocks
- `scraping.py` router: 8 repeated 8-line blocks
- `classifiers.py`: 5 repeated 8-line blocks
- `ml_operations.py`: 2 repeated 8-line blocks
- `FetchDialog.tsx`: 2 repeated 8-line blocks

**Root cause:** Common patterns (error handling, response serialization, pagination, retry loops, form boilerplate) are copy-pasted rather than centralized.

**Actions:**
- [x] Create `algobet/api/common.py` for shared router helpers:
  - `handle_service_error(exc) -> HTTPException`
  - `parse_pagination_params(request)`
  - `serialize_paginated(results, total, page, size)`
- [x] Create `algobet/services/common.py` for shared service primitives:
  - `retry_with_backoff(fn, max_retries)`
  - `batch_persist(session, items, model_class)`
- [x] Create `frontend/lib/form-utils.ts` for repeated form/validation blocks.
- [ ] Run the audit again after helpers are introduced; target **zero high-severity DRY findings** in files >500 lines.

---

## 8. Establish Guardrails to Prevent Regression (Systemic)

**Actions:**
- [x] Add the architecture audit script (`scripts/audit_architecture_antipatterns.py`) to CI with thresholds:
  - `--max-score 5` for any module >500 lines.
  - `--max-god-class-methods 10`.
  - `--max-function-lines 60`.
- [x] Update `AGENTS.md` (or create `CONTRIBUTING.md`) with the **Router Slimness Rule** and **No God Classes** policy.
- [ ] Schedule a quarterly architecture review using the same audit parameters to verify trends improve.

---

## Priority Summary

| Priority | Theme | Rationale |
|----------|-------|-----------|
| **P0** | #1 + #2 + #3 (Router/Service/Scraping decomposition) | Highest scores, most duplication, deepest nesting; changes unlock everything else. |
| **P1** | #7 (Systemic DRY helpers) | Can be done in parallel with P0; reduces duplication before final splits. |
| **P2** | #4 + #5 (Repository & ML modules) | Large classes, but lower duplication; safer to refactor after patterns are established in P0. |
| **P3** | #6 (Frontend pages/components) | Isolated from backend regressions; can follow the same decomposition playbook. |
| **P4** | #8 (Guardrails) | Final step to lock in improvements. |
