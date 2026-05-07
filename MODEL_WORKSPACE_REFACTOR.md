# Models Page Guided Workspace Refactor

## Summary
- Rebuild `/models` as a training-first workspace instead of a stats-plus-table page.
- Keep the existing backend and TanStack Query contracts unchanged; the refactor is a frontend composition and UX rewrite.
- Expose training filters progressively through staged sections, while also making the model registry easier to browse, inspect, activate, and retire.

## Key Changes
- Refactor `frontend/app/models/page.tsx` into a thin orchestration page that owns:
  - fetched data from `useModels`, `useActiveModel`, `useActivateModel`, `useDeleteModel`
  - page-level state for selected model, registry filters, and guided training section expansion
  - a page-level refresh action
  - deep-link selection from `/models?id=<modelId>`, with fallback to the active model, then the first available model
- Replace the current left-card/right-table layout with three coordinated areas:
  - a compact page header with active-model summary and refresh
  - a guided training workspace as the primary content
  - a registry plus selected-model inspector as the secondary content
- Rework the training UX in `frontend/components/models/*` into guided sections:
  - `Basics` open by default: model type, description, activate-after-training, tuning toggle, calibration toggle
  - `Data Scope` collapsed by default: date range, minimum matches, tournaments, teams, venue, goals range
  - `Validation` collapsed by default: split ratios, split strategy, strategy-specific fields, outcome balancing, calibration method
  - `Expert Options` collapsed by default: feature groups, feature selection settings, ensemble settings, custom hyperparameters, seed, early stopping, tuning trials
- Add `TrainingSettingsSection` component consolidating configurable training options with validation for split ratios
- Add `TrainingSummary` component displaying active configuration including feature selection status, tuning status, calibration method, and selected feature groups
- Add a persistent training summary rail/card that shows:
  - selected model type
  - active data filters count
  - split strategy
  - tuning / ensemble / calibration status
  - feature selection status (enabled with threshold)
  - primary train CTA
  - split-ratio validation warning
- Replace the table-first registry experience with a responsive searchable registry:
  - client-side text search across version, description, and algorithm
  - status filter: all, active, inactive
  - algorithm filter chips
  - card/list presentation that works on mobile and desktop
- Add a selected-model inspector panel driven by registry selection:
  - reuse the metrics query for detailed metrics
  - show description, created date, feature schema, hyperparameters, active status
  - keep activate and delete actions in the inspector
- If the current `ModelRow` and `ModelMetricsPanel` abstractions fight the new UX, replace them instead of forcing the new design into the existing table pattern.

## Interfaces
- No backend API changes.
- No query-hook contract changes.
- New client-side deep-link behavior: `/models?id=<modelId>` selects and opens that model in the inspector.
- Keep `TrainingConfig` as the main form state shape unless a small derived-summary helper is added alongside it.

## Test Plan
- Render default page state with training basics visible and deeper sections collapsed.
- Verify progressive disclosure for data, validation, and expert sections.
- Verify conditional controls for tuning, calibration, ensemble, and split strategy.
- Verify registry search, status filtering, and algorithm filtering.
- Verify model selection opens the inspector and loads metrics.
- Verify `/models?id=<validId>` preselects the model and `/models?id=<invalidId>` falls back safely.
- Verify activate and delete still call the existing mutations.
- Verify training submit preserves the current request payload mapping and blocks invalid split totals.
- Verify loading and empty states remain usable for both registry and inspector.

## Assumptions
- Scope includes adding or refactoring supporting components under `frontend/components/models`, not only editing `page.tsx`.
- “Expose all filter in progressive way” applies primarily to training configuration, with secondary progressive filters for browsing saved models.
- This refactor does not add multi-model comparison or backend-side registry filtering.
- Visual style stays within the existing Tailwind/shadcn system and app tokens rather than introducing a new design system.
