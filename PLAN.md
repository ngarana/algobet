# Predictions Page Refactoring Plan

## Current State Analysis

The current `frontend/app/predictions/page.tsx` is a 542-line monolithic component with several issues:

### Problems Identified

1. **Poor Code Organization**
   - Three components (`PredictionsSummary`, `PredictionRow`, `PredictionControls`) defined in same file
   - All state management, API calls, and UI logic mixed in single component
   - Difficult to test, maintain, or reuse components

2. **Generic UI Design**
   - Basic table view not optimized for betting predictions
   - No visual representation of prediction confidence/probabilities
   - Missing key betting metrics (expected value, Kelly criterion, value indicators)
   - No model performance comparison

3. **Missing Features**
   - No sorting/filtering beyond basic upcoming/history toggle
   - No search functionality
   - No bulk actions (e.g., export predictions)
   - No prediction confidence visualization
   - No odds comparison with market odds
   - No model accuracy metrics display
   - No date range picker for custom history views

4. **Limited Data Presentation**
   - Probabilities shown as raw numbers without visual aids
   - No sparklines or trend indicators
   - Missing match context (form, H2H, etc.)
   - No quick action buttons (e.g., view match details)

## Proposed Architecture

### New File Structure

```
frontend/app/predictions/
├── page.tsx                          # Main page (simplified, orchestrates components)
├── loading.tsx                        # Loading state
├── error.tsx                          # Error state
└── components/
    ├── index.ts                       # Barrel exports
    ├── PredictionDashboard.tsx         # Main dashboard layout
    ├── PredictionControls.tsx          # Model selection, generation, activation
    ├── PredictionStats.tsx             # Summary statistics cards
    ├── PredictionFilters.tsx           # Filters: search, date range, confidence, outcome
    ├── PredictionTable.tsx             # Main table with sorting/pagination
    ├── PredictionCard.tsx             # Card view for mobile/alternative view
    ├── PredictionRow.tsx               # Individual row component
    ├── PredictionDetailModal.tsx        # Modal for detailed prediction view
    ├── ConfidenceIndicator.tsx         # Visual confidence/probability display
    ├── ModelPerformanceCard.tsx        # Model metrics display
    ├── ValueBetIndicator.tsx           # Shows if prediction is a value bet
    ├── GenerationResultCard.tsx        # Displays generation run results
    └── ExportButton.tsx                # Export predictions to CSV/JSON
```

### Component Responsibilities

#### 1. `PredictionDashboard.tsx` (New)
- **Purpose**: Main layout component that orchestrates all prediction page sections
- **Responsibilities**:
  - Renders header with title and view toggles
  - Manages layout (controls, stats, table/filters)
  - Handles error display
  - Coordinates data fetching state

#### 2. `PredictionControls.tsx` (Refactored)
- **Purpose**: Model selection and prediction generation
- **Improvements**:
  - Add model performance metrics (accuracy, last trained date)
  - Show model status more prominently
  - Add generation progress indicator
  - Include tournament filter for generation
  - Add tooltip explanations for controls

#### 3. `PredictionStats.tsx` (Enhanced)
- **Current**: Basic 4-card summary
- **New Features**:
  - Total predictions count
  - Average confidence with trend
  - Win rate (for historical)
  - Value bets count
  - ROI percentage (for historical)
  - Model accuracy indicator
  - Sparkline mini-charts for trends

#### 4. `PredictionFilters.tsx` (New)
- **Purpose**: Advanced filtering and search
- **Features**:
  - Search by team name
  - Date range picker
  - Confidence range slider (min/max)
  - Outcome filter (Home/Draw/Away)
  - Tournament filter
  - Model version filter
  - Only show value bets toggle
  - Reset filters button
  - Active filter count badge

#### 5. `PredictionTable.tsx` (Major Refactor)
- **Current**: Basic table with minimal features
- **New Features**:
  - Sortable columns (click to sort)
  - Pagination (server-side or client-side)
  - Row selection for bulk actions
  - Toggle between table/card view
  - Column visibility toggle
  - Visual confidence bars
  - Value bet highlighting
  - Quick actions per row (view details, export)
  - Responsive design (horizontal scroll on mobile)

#### 6. `PredictionRow.tsx` (Enhanced)
- **Current**: Basic row with 5 columns
- **New Features**:
  - Visual probability bars (home/draw/away)
  - Confidence indicator component
  - Value bet badge with expected value
  - Match status indicator
  - Quick action buttons
  - Expandable row for match details
  - Odds comparison display

#### 7. `ConfidenceIndicator.tsx` (New)
- **Purpose**: Visual representation of prediction confidence
- **Features**:
  - Horizontal bar with color gradient (red → yellow → green)
  - Percentage display
  - Tooltip with probability breakdown
  - Size variants (sm, md, lg)
  - Animated on generation

#### 8. `ValueBetIndicator.tsx` (New)
- **Purpose**: Highlight predictions with positive expected value
- **Features**:
  - Green/red badge based on expected value
  - Show expected value percentage
  - Kelly criterion fraction
  - Market odds vs predicted probability comparison
  - Tooltip with calculation details

#### 9. `ModelPerformanceCard.tsx` (New)
- **Purpose**: Display model metrics and performance
- **Features**:
  - Accuracy rate
  - Total predictions
  - ROI (if applicable)
  - Last training date
  - Feature importance preview
  - Link to full model page

#### 10. `PredictionDetailModal.tsx` (New)
- **Purpose**: Show detailed prediction information
- **Features**:
  - Full match details
  - All probabilities with visualizations
  - Model information
  - Historical performance for this matchup
  - Head-to-head record
  - Recent form for both teams
  - Export single prediction

#### 11. `PredictionCard.tsx` (New)
- **Purpose**: Card view for alternative layout (mobile-friendly)
- **Features**:
  - Compact prediction display
  - Visual indicators
  - Quick actions
  - Expandable for more details

#### 12. `ExportButton.tsx` (New)
- **Purpose**: Export predictions data
- **Features**:
  - Export to CSV
  - Export to JSON
  - Select fields to export
  - Filter-aware export (exports current filtered view)

## Data Enhancements

### Add to API Calls
1. **Server-side Filtering**
   - Update `usePredictions` to support all filter parameters
   - Add pagination support to API if not present

2. **New Query Hooks**
   ```
   usePredictionStats()           # Aggregate statistics
   useModelPerformance()         # Model-specific metrics
   useValueBets()               # Filter for value bets only
   ```

3. **Enhanced Types**
   - Add `expected_value` to Prediction type if not present
   - Add `kelly_fraction` to Prediction type
   - Add `market_comparison` object

## UI/UX Improvements

### Visual Design
1. **Color Coding**
   - Home win: Blue gradient
   - Draw: Amber/Gold
   - Away win: Red gradient
   - Confidence: Green (high) → Yellow (med) → Red (low)
   - Value bets: Green highlight

2. **Layout**
   - Sticky filters bar when scrolling
   - Collapsible stats section
   - Responsive grid for stats cards
   - Smooth transitions on data changes

3. **Interactions**
   - Click row to expand details
   - Hover for quick actions
   - Optimistic UI updates
   - Toast notifications for actions
   - Confirmation dialogs for destructive actions

### User Experience
1. **Empty States**
   - Better illustrations
   - Clear call-to-action buttons
   - Help text explaining how to generate predictions

2. **Loading States**
   - Skeleton loaders matching final layout
   - Progressive loading for stats vs table
   - Inline loading for actions (generate, activate)

3. **Error Handling**
   - Inline field errors
   - Retry buttons
   - Error boundaries for components
   - User-friendly error messages

## Implementation Order

### Phase 1: Foundation (Week 1)
1. Create new directory structure
2. Extract existing components into separate files
3. Create barrel exports (`index.ts`)
4. Build `PredictionDashboard` as orchestrator
5. Refactor `PredictionControls` with improvements
6. Enhance `PredictionStats` with new metrics

### Phase 2: Core Features (Week 2)
1. Build `PredictionFilters` component
2. Refactor `PredictionTable` with sorting/pagination
3. Enhance `PredictionRow` with visual indicators
4. Create `ConfidenceIndicator` component
5. Add server-side filtering support

### Phase 3: Advanced Features (Week 3)
1. Build `ValueBetIndicator` component
2. Create `ModelPerformanceCard`
3. Implement `PredictionDetailModal`
4. Add `PredictionCard` for alternative view
5. Implement `ExportButton`

### Phase 4: Polish (Week 4)
1. Add animations and transitions
2. Improve responsive design
3. Add keyboard shortcuts
4. Implement virtual scrolling for large datasets
5. Performance optimization
6. Comprehensive testing

## Technical Considerations

### State Management
- Use Zustand store for UI state (filters, view mode, selection)
- Keep TanStack Query for server state
- Consider URL state sync for filters (shareable links)

### Performance
- Memoize expensive calculations
- Virtual scrolling for 100+ predictions
- Debounce filter changes
- Paginate server-side when possible
- Lazy load modal/detail components

### Testing
- Unit tests for each new component
- Integration tests for filter interactions
- E2E tests for generation flow
- Visual regression tests for charts/indicators

### Accessibility
- ARIA labels for all interactive elements
- Keyboard navigation support
- Screen reader friendly tables
- Focus management in modals
- Color contrast compliance

## Success Metrics

After refactoring, the page should achieve:
- ✅ Component files under 200 lines each
- ✅ 90%+ test coverage for new components
- ✅ Lighthouse accessibility score > 90
- ✅ Time to interactive < 2s
- ✅ Smooth 60fps scrolling with 1000+ predictions
- ✅ All user flows completable via keyboard

## Appendix: API Changes Needed

If certain data isn't available, coordinate with backend:

1. **Prediction Statistics Endpoint**
   ```
   GET /api/v1/predictions/stats
   Query: { model_version_id?, from_date?, to_date? }
   Returns: { total, avg_confidence, win_rate, roi, value_bets_count }
   ```

2. **Bulk Export Endpoint**
   ```
   GET /api/v1/predictions/export
   Query: { format: 'csv' | 'json', filters... }
   Returns: File download
   ```

3. **Value Bets Filter**
   - Add `min_expected_value` parameter to predictions filter
   - Backend calculates expected value based on odds and probabilities
