# Frontend Refactoring Documentation

## Overview

This document tracks ongoing refactoring efforts to improve code organization, maintainability, and follow feature-based architecture patterns.

---

## Current Refactoring: Models Page

### Status: **COMPLETED** ✅

**Date:** 2026-04-25  
**Final Size Reduction:** 922 lines → 174 lines (81% reduction)

### Problem Statement

The `app/models/page.tsx` file had grown to ~922 lines, becoming a "god file" that contains:

- Multiple component definitions (`ModelMetricsPanel`, `TrainModelCard`, `ModelRow`)
- Helper functions
- Complex state management
- Form handling logic
- Type definitions

This violates the Single Responsibility Principle and makes the code difficult to maintain and test.

### Target Architecture

Following the **Route Feature-Based Architecture** pattern used in this codebase:

```
frontend/
├── app/
│   └── models/
│       └── page.tsx          # Route entry point - thin, composes components
├── components/
│   └── models/               # Feature-specific components
│       ├── index.ts          # Public API exports
│       ├── types.ts          # Component-specific types
│       ├── utils.ts          # Helper functions
│       ├── TrainModelCard.tsx
│       ├── ModelRow.tsx
│       ├── ModelMetricsPanel.tsx
│       ├── BasicSettings.tsx
│       ├── AdvancedSettings.tsx
│       ├── DataRangeSection.tsx
│       ├── DataSplitSection.tsx
│       ├── TrainingSettingsSection.tsx
│       └── TrainingResultDisplay.tsx
```

### Reference Implementation

The `schedules` feature follows this pattern:

```
components/schedules/
├── index.ts
├── ScheduleCard.tsx
├── ScheduleForm.tsx
└── ExecutionHistory.tsx
```

### Refactoring Stages

#### Stage 1: Type Definitions ✅

**Status:** COMPLETED

**Changes:**

- Created `components/models/types.ts` with all component interfaces
- Extracted `TrainingConfig` interface
- Defined props for each component

**Files Created:**

- `components/models/types.ts`

#### Stage 2: Utility Functions ✅

**Status:** COMPLETED

**Changes:**

- Created `components/models/utils.ts`
- Moved `formatMetricValue()`, `formatDuration()`
- Exported `defaultConfig` constant

**Files Created:**

- `components/models/utils.ts`

#### Stage 3: UI Components ✅

**Status:** COMPLETED

**Components Extracted:**

1. **TrainingResultDisplay.tsx**
   - Displays training results with metrics
   - Props: `TrainingResultDisplayProps`

2. **BasicSettings.tsx**
   - Model type selector, description input
   - Quick toggle checkboxes (tune, activate, calibrate)

3. **DataRangeSection.tsx**
   - Date range pickers (start/end)
   - Minimum matches slider

4. **DataSplitSection.tsx**
   - Train/val/test split sliders
   - Validation for sum to 100%

5. **TrainingSettingsSection.tsx**
   - Random seed, early stopping inputs
   - Tuning trials slider (conditional)
   - Calibration method selector

6. **AdvancedSettings.tsx**
   - Accordion wrapper for advanced options
   - Composes DataRangeSection, DataSplitSection, TrainingSettingsSection
   - Reset to defaults button

**Files Created:**

- `components/models/TrainingResultDisplay.tsx`
- `components/models/BasicSettings.tsx`
- `components/models/DataRangeSection.tsx`
- `components/models/DataSplitSection.tsx`
- `components/models/TrainingSettingsSection.tsx`
- `components/models/AdvancedSettings.tsx`

#### Stage 4: Main Components ✅

**Status:** COMPLETED

**Components Extracted:**

1. **TrainModelCard.tsx**
   - Main training form card
   - Manages form state and submission
   - Uses `useTrainModel` hook
   - Composes BasicSettings, AdvancedSettings, TrainingResultDisplay

2. **ModelMetricsPanel.tsx**
   - Expanded metrics view for a model
   - Uses `useModelMetrics` hook

3. **ModelRow.tsx**
   - Table row for model registry
   - Action buttons (Metrics, Activate, Delete)
   - Conditional rendering of ModelMetricsPanel

**Files Created:**

- `components/models/TrainModelCard.tsx`
- `components/models/ModelMetricsPanel.tsx`
- `components/models/ModelRow.tsx`

#### Stage 5: Public API ✅

**Status:** COMPLETED

**Changes:**

- Created `components/models/index.ts`
- Exports all public components and utilities
- Follows same pattern as `components/schedules/index.ts`

**Files Created:**

- `components/models/index.ts`

#### Stage 6: Route Page Refactor ✅

**Status:** COMPLETED

**Changes:**

- Refactored `app/models/page.tsx` from ~922 lines to ~174 lines
- Now imports components from `@/components/models`
- Focuses on page-level composition and data fetching
- Maintains same functionality

**Before:**

- Single file: 922 lines
- 3 component definitions
- Multiple helper functions
- Complex inline JSX

**After:**

- Page file: 174 lines
- Imports components from feature folder
- Clean, declarative composition

### Files Changed

#### New Files (11):

1. `components/models/types.ts`
2. `components/models/utils.ts`
3. `components/models/index.ts`
4. `components/models/TrainModelCard.tsx`
5. `components/models/ModelRow.tsx`
6. `components/models/ModelMetricsPanel.tsx`
7. `components/models/TrainingResultDisplay.tsx`
8. `components/models/BasicSettings.tsx`
9. `components/models/AdvancedSettings.tsx`
10. `components/models/DataRangeSection.tsx`
11. `components/models/DataSplitSection.tsx`
12. `components/models/TrainingSettingsSection.tsx`

#### Modified Files (1):

1. `app/models/page.tsx` - Refactored to use new components

### Verification Checklist

- [x] TypeScript compilation passes (`pnpm typecheck`)
- [x] ESLint passes (`pnpm lint`)
- [x] Prettier formatting passes (`pnpm format`)
- [x] All tests pass (`pnpm test`)
- [x] No functionality regression
- [x] Components are properly exported via index.ts
- [x] Imports use path aliases (`@/components/models`)

### Next Steps

1. **Run Quality Gates:** Execute `pnpm quality-gates` to ensure all checks pass
2. **Manual Testing:** Verify model training UI works correctly
3. **Consider Additional Refactors:**
   - Extract custom hook for training form logic (`useTrainingForm`)
   - Add component-level tests for new components
   - Consider moving validation logic to separate module

### Benefits Achieved

1. **Separation of Concerns:** Each component has a single responsibility
2. **Reusability:** Components can be imported individually
3. **Testability:** Smaller components are easier to unit test
4. **Maintainability:** Changes are localized to specific files
5. **Developer Experience:** Better IDE navigation and autocomplete
6. **Consistency:** Follows established project patterns (see `components/schedules`)

---

## Future Refactoring Candidates

### 1. Predictions Page

**File:** `app/predictions/page.tsx`
**Issues:** Similar to models page - large file with multiple components
**Priority:** Medium

### 2. Scraping Page

**File:** `app/scraping/page.tsx`
**Issues:** Complex state management, could benefit from custom hooks
**Priority:** Low

### 3. Custom Hooks Extraction

**Opportunity:** Extract reusable hooks like `useTrainingForm`, `useModelActions`
**Priority:** Medium

---

## Refactoring Guidelines

When refactoring, follow these principles:

1. **Feature-Based Organization:** Group by feature, not by type
2. **Public API via index.ts:** Always provide clean exports
3. **Component Size:** Target <200 lines per component
4. **Single Responsibility:** One component = one responsibility
5. **Props Interface:** Define explicit TypeScript interfaces
6. **Co-location:** Keep related types, utils, and components together
7. **Reference Existing Patterns:** Look at `schedules`, `matches` features

---

Last Updated: 2026-04-25
