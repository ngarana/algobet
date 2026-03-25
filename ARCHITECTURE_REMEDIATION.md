# Feature-Root Architecture Violations & Remediation Guide

This document identifies all feature-root architecture violations in the AlgoBet codebase and provides detailed remediation strategies.

---

## Executive Summary

**Project**: AlgoBet Football Match Prediction System
**Current Architecture**: Hybrid (partially feature-based with heavy technical-layer contamination)
**Target Architecture**: Feature-Root (modular, feature-co-located, low coupling)

---

## Table of Contents

1. [Architecture Principles](#architecture-principles)
2. [Current State Analysis](#current-state-analysis)
3. [Violation Categories](#violation-categories)
4. [Remediation Roadmap](#remediation-roadmap)
5. [Specific Remediation Guides](#specific-remediation-guides)

---

## Architecture Principles

### What is Feature-Root Architecture?

Feature-root architecture (also called "vertical slicing" or "feature-based organization") organizes code by business functionality rather than technical layer.

**Core Principles:**

1. **Feature Co-location**: All code related to a feature lives together
2. **Low Coupling**: Features communicate through well-defined public APIs
3. **High Cohesion**: Related functionality stays together
4. **Self-Contained**: Features can be understood and modified in isolation

**Anti-Patterns Violated:**
- Technical layer folders (`services/`, `components/`, `hooks/`)
- Centralized models/exceptions/config
- Deep imports across feature boundaries

---

## Current State Analysis

### Directory Structure

```
/home/arch/Coding/algobet/
├── algobet/                    # Python backend (FastAPI)
│   ├── api/                    # API layer with technical separation
│   ├── cli/                    # CLI with mixed concerns
│   ├── services/               # CRITICAL: Technical layer folder
│   ├── predictions/            # Feature-based (but internal issues)
│   ├── importers/
│   ├── scheduler/
│   ├── models.py               # All models centralized
│   ├── database.py             # Infrastructure at root
│   ├── config.py               # Configuration at root
│   ├── exceptions.py           # Exceptions centralized
│   └── scraper.py              # Logic at root level
│
├── frontend/                   # Next.js frontend
│   ├── app/                    # Next.js App Router (pages by feature)
│   ├── components/             # CRITICAL: Technical layer folder
│   ├── hooks/                  # CRITICAL: Technical layer folder
│   ├── stores/                 # CRITICAL: Technical layer folder
│   └── lib/                    # Utilities (some issues)
│
└── tests/                      # Tests by type (not feature)
```

---

## Violation Categories

### Category 1: CRITICAL - Technical Layer Folders at Root Level

**Severity**: 🔴 **CRITICAL** - Prevents feature isolation and modularity

| Violation | Location | Impact |
|-----------|----------|--------|
| **Services organized by layer** | `algobet/services/` | 15+ service files mixing features; high coupling |
| **All domain models centralized** | `algobet/models.py` | Single file with all SQLAlchemy models; hard to maintain |
| **Frontend hooks separated** | `frontend/hooks/` | Hooks divorced from components; reusability confusion |
| **Frontend stores centralized** | `frontend/stores/` | State management separated from features |
| **Frontend components by layer** | `frontend/components/` | Mix of feature and generic; pages separated from components |

**Impact**:
- Changes require touching multiple directories
- Difficult to understand feature scope
- High coupling between features
- Risk of circular dependencies

---

### Category 2: HIGH - Root-Level Infrastructure Files

**Severity**: 🟠 **HIGH** - Creates centralized coupling points

| Violation | Location | Impact |
|-----------|----------|--------|
| **Configuration centralized** | `algobet/config.py` | All config in one place; env-specific overrides hard |
| **Exceptions centralized** | `algobet/exceptions.py` | Generic exceptions divorced from context |
| **Database at root** | `algobet/database.py` | Infrastructure mixed with domain |
| **Scraping logic at root** | `algobet/scraper.py` | Should be in scraping feature |
| **CLI technical files** | `algobet/cli/*.py` | `error_handler.py`, `logger.py`, `presenters.py` at root |

---

### Category 3: MEDIUM - API Layer Separation

**Severity**: 🟡 **MEDIUM** - Adds navigation overhead

| Violation | Location | Impact |
|-----------|----------|--------|
| **Routers and schemas separate** | `api/routers/` + `api/schemas/` | Schemas should co-locate with routers |
| **Dependencies centralized** | `api/dependencies.py` | Mixed concerns; per-feature dependencies preferred |
| **Frontend API clients separate** | `frontend/lib/api/` | Should be `app/{feature}/api.ts` |
| **Frontend queries separate** | `frontend/lib/queries/` | TanStack Query hooks should co-locate |

---

### Category 4: MEDIUM - Internal Feature Technical Layering

**Severity**: 🟡 **MEDIUM** - Minor inconsistency

| Violation | Location | Impact |
|-----------|----------|--------|
| **Predictions subfolders by layer** | `predictions/{data,features,models,training}/` | Technical layers within feature |

---

### Category 5: HIGH - Cross-Feature Coupling

**Severity**: 🟠 **HIGH** - Violates encapsulation

**Examples found:**

```python
# In prediction_service.py - reaches into other feature internals
from algobet.predictions.data.queries import MatchRepository
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.models.registry import ModelRegistry

# In scraping_service.py - imports centralized models
from algobet.models import Match, Team, Tournament
from algobet.scraper import OddsPortalScraper
```

**Impact**: Changing internal implementation of one feature breaks others

---

### Category 6: MEDIUM - Test Organization

**Severity**: 🟡 **MEDIUM** - Poor test discoverability

| Violation | Location | Impact |
|-----------|----------|--------|
| **Tests by type** | `tests/unit/`, `tests/integration/` | Hard to find tests for specific feature |
| **Test files scattered** | Root of `tests/` | No clear organization |

---

## Remediation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal**: Establish new directory structure

1. Create `algobet/infrastructure/` for cross-cutting concerns
2. Create `algobet/matches/`, `algobet/scraping/`, `algobet/scheduling/` feature folders
3. Create `frontend/lib/infrastructure/` for shared utilities

### Phase 2: Backend Migration (Week 3-4)
**Goal**: Migrate backend to feature-root structure

1. **Decompose models.py** → `matches/models.py`, `predictions/models.py`, etc.
2. **Move services** → `matches/service.py`, `scraping/service.py`, etc.
3. **Migrate infrastructure** → Move `config.py`, `database.py`, `exceptions.py`
4. **Update imports** throughout codebase

### Phase 3: Frontend Migration (Week 5-6)
**Goal**: Migrate frontend to feature-root structure

1. **Co-locate components** → Move `components/matches/` to `app/matches/components/`
2. **Co-locate hooks** → Move `hooks/use-matches.ts` to `app/matches/hooks/`
3. **Co-locate API clients** → Move `lib/api/matches.ts` to `app/matches/api.ts`
4. **Co-locate stores** → Move feature-specific stores

### Phase 4: Testing & Cleanup (Week 7-8)
**Goal**: Complete migration and establish patterns

1. **Reorganize tests** → By feature rather than type
2. **Remove deprecated folders**
3. **Update documentation**
4. **Establish linting rules** to prevent regressions

---

## Specific Remediation Guides

### Guide 1: Decomposing `models.py`

**Current State:**
```
algobet/
└── models.py          # All models (200+ lines)
```

**Target State:**
```
algobet/
├── infrastructure/
│   └── models.py      # Base classes only
├── matches/
│   └── models.py      # Match, MatchStats, etc.
├── predictions/
│   └── models.py      # Prediction, ModelPerformance, etc.
├── scraping/
│   └── models.py      # ScrapingJob, ScrapedData, etc.
├── scheduling/
│   └── models.py      # Schedule, Task, etc.
└── teams/
    └── models.py      # Team, Tournament, etc.
```

**Step-by-Step:**

1. **Create infrastructure models file:**
```python
# algobet/infrastructure/models.py
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Any truly shared mixins
class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

2. **Extract Match models:**
```python
# algobet/matches/models.py
from algobet.infrastructure.models import Base, TimestampMixin

class Match(Base, TimestampMixin):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    # ... other match fields
```

3. **Extract Prediction models:**
```python
# algobet/predictions/models.py
from algobet.infrastructure.models import Base

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    # ... other prediction fields
```

4. **Repeat for other entities**

5. **Update imports across codebase:**
```python
# Before:
from algobet.models import Match, Prediction, Team

# After:
from algobet.matches.models import Match
from algobet.predictions.models import Prediction
from algobet.teams.models import Team
```

**Estimated Effort**: 2-3 days

---

### Guide 2: Migrating Services from Layer to Feature

**Current State:**
```
algobet/services/
├── base.py
├── dto.py
├── database_service.py
├── prediction_service.py
├── scraping_service.py
├── scheduler_service.py
├── query_service.py
├── model_management_service.py
├── analysis_service.py
├── async_database_service.py
└── async_scraping_service.py
```

**Target State:**
```
algobet/
├── infrastructure/
│   └── services/
│       ├── base.py              # Generic base (optional)
│       └── database.py          # Generic DB operations
├── matches/
│   ├── service.py
│   └── dto.py
├── predictions/
│   ├── service.py
│   └── dto.py
├── scraping/
│   ├── service.py
│   ├── async_service.py
│   └── dto.py
└── scheduling/
    ├── service.py
    └── dto.py
```

**Step-by-Step:**

1. **Create infrastructure base (optional):**
```python
# algobet/infrastructure/services/base.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar('T')

class BaseService(ABC, Generic[T]):
    """Generic base service - can be kept or removed"""

    @abstractmethod
    async def get_by_id(self, id: int) -> T | None:
        pass
```

2. **Migrate Match Service:**
```python
# algobet/matches/service.py
from typing import Optional
from algobet.matches.models import Match
from algobet.infrastructure.database import get_session

class MatchService:
    """Service for match-related operations"""

    async def get_match(self, match_id: int) -> Optional[Match]:
        async with get_session() as session:
            return await session.get(Match, match_id)

    async def list_matches(self, filters: MatchFilters) -> list[Match]:
        # Implementation
        pass

# Optional: Keep backward compatibility alias
from algobet.matches.service import MatchService as MatchServiceImpl
```

3. **Migrate Prediction Service:**
```python
# algobet/predictions/service.py
from algobet.predictions.models import Prediction
from algobet.matches.service import MatchService  # Cross-feature import OK if public API

class PredictionService:
    """Service for prediction-related operations"""

    def __init__(self, match_service: MatchService):
        self.match_service = match_service

    async def create_prediction(self, match_id: int) -> Prediction:
        match = await self.match_service.get_match(match_id)
        # ... prediction logic
```

4. **Migrate DTOs with services:**
```python
# algobet/matches/dto.py
from pydantic import BaseModel
from datetime import datetime

class MatchCreate(BaseModel):
    home_team_id: int
    away_team_id: int
    scheduled_at: datetime

class MatchResponse(BaseModel):
    id: int
    home_team: str
    away_team: str
    score: str | None
```

5. **Update all imports:**
```python
# Before:
from algobet.services.prediction_service import PredictionService
from algobet.services.dto import PredictionCreate

# After:
from algobet.predictions.service import PredictionService
from algobet.predictions.dto import PredictionCreate
```

**Estimated Effort**: 3-4 days

---

### Guide 3: Migrating Frontend Components

**Current State:**
```
frontend/
├── app/
│   ├── matches/
│   │   └── page.tsx
│   ├── scraping/
│   │   └── page.tsx
│   └── backtest/
│       └── page.tsx
└── components/
    ├── matches/
    │   ├── MatchCard.tsx
    │   └── MatchList.tsx
    ├── scraping/
    │   ├── ScrapingForm.tsx
    │   └── ScrapingResults.tsx
    ├── backtest/
    │   └── BacktestPanel.tsx
    ├── ui/              # Shared primitives (keep)
    ├── charts/          # Generic charts (keep)
    ├── layout/          # Layout components (keep)
    └── skeletons/       # Loading states (move to ui/)
```

**Target State:**
```
frontend/
├── app/
│   ├── matches/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── components/
│   │       ├── MatchCard.tsx
│   │       └── MatchList.tsx
│   ├── scraping/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── components/
│   │       ├── ScrapingForm.tsx
│   │       └── ScrapingResults.tsx
│   └── backtest/
│       ├── page.tsx
│       └── components/
│           └── BacktestPanel.tsx
└── components/
    └── ui/              # Shared UI primitives only
        ├── Button.tsx
        ├── Input.tsx
        ├── charts/
        └── skeletons/
```

**Step-by-Step:**

1. **Create component directories in app:**
```bash
mkdir -p frontend/app/matches/components
mkdir -p frontend/app/scraping/components
mkdir -p frontend/app/backtest/components
```

2. **Move feature components:**
```bash
# Move matches components
mv frontend/components/matches/* frontend/app/matches/components/
rmdir frontend/components/matches

# Move scraping components
mv frontend/components/scraping/* frontend/app/scraping/components/
rmdir frontend/components/scraping

# Move backtest components
mv frontend/components/backtest/* frontend/app/backtest/components/
rmdir frontend/components/backtest
```

3. **Organize generic components:**
```bash
# Move skeletons to ui
mv frontend/components/skeletons/* frontend/components/ui/skeletons/
rmdir frontend/components/skeletons

# Move charts to ui
mv frontend/components/charts/* frontend/components/ui/charts/
rmdir frontend/components/charts

# Keep layout components
# frontend/components/layout/ stays
```

4. **Update imports in moved components:**
```typescript
// Before (in components/matches/MatchCard.tsx):
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'

// After (in app/matches/components/MatchCard.tsx):
import { Button } from '@/components/ui/Button'
import { Card } from '@/components/ui/Card'
// Same imports, just file location changed
```

5. **Update page imports:**
```typescript
// Before (in app/matches/page.tsx):
import { MatchList } from '@/components/matches/MatchList'

// After (in app/matches/page.tsx):
import { MatchList } from './components/MatchList'
```

**Estimated Effort**: 1-2 days

---

### Guide 4: Migrating Frontend Hooks

**Current State:**
```
frontend/hooks/
├── use-matches.ts
├── use-predictions.ts
├── use-scraping.ts
├── use-auth.ts
└── use-local-storage.ts
```

**Target State:**
```
frontend/
├── app/
│   ├── matches/
│   │   └── hooks/
│   │       └── use-matches.ts
│   ├── scraping/
│   │   └── hooks/
│   │       └── use-scraping.ts
│   └── (shared)/
│       └── hooks/
│           ├── use-auth.ts
│           └── use-local-storage.ts
└── lib/
    └── hooks/           # Keep truly generic hooks
        └── use-debounce.ts
```

**Step-by-Step:**

1. **Identify hook scope:**
```typescript
// Feature-specific hooks (move to app/{feature}/hooks/):
// - use-matches.ts (only used in matches feature)
// - use-scraping.ts (only used in scraping feature)

// Generic hooks (keep in lib/hooks/):
// - use-local-storage.ts (used across app)
// - use-debounce.ts (utility)
```

2. **Create feature hook directories:**
```bash
mkdir -p frontend/app/matches/hooks
mkdir -p frontend/app/scraping/hooks
mkdir -p frontend/lib/hooks
```

3. **Move feature-specific hooks:**
```bash
mv frontend/hooks/use-matches.ts frontend/app/matches/hooks/
mv frontend/hooks/use-scraping.ts frontend/app/scraping/hooks/
```

4. **Move generic hooks:**
```bash
mv frontend/hooks/use-local-storage.ts frontend/lib/hooks/
mv frontend/hooks/use-auth.ts frontend/lib/hooks/
```

5. **Update imports:**
```typescript
// Before:
import { useMatches } from '@/hooks/use-matches'

// After:
import { useMatches } from './hooks/use-matches'
// or if from another feature:
import { useMatches } from '@/app/matches/hooks/use-matches'
```

**Estimated Effort**: 1 day

---

### Guide 5: Migrating Frontend API Clients

**Current State:**
```
frontend/lib/api/
├── client.ts
├── index.ts
├── matches.ts
├── predictions.ts
├── scraping.ts
└── teams.ts
```

**Target State:**
```
frontend/
├── app/
│   ├── matches/
│   │   └── api.ts      # Match API functions
│   ├── predictions/
│   │   └── api.ts      # Prediction API functions
│   └── scraping/
│       └── api.ts      # Scraping API functions
└── lib/
    └── infrastructure/
        └── api/
            └── client.ts  # Generic HTTP client only
```

**Step-by-Step:**

1. **Create infrastructure API client:**
```typescript
// frontend/lib/infrastructure/api/client.ts
import axios from 'axios'

export const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: { 'Content-Type': 'application/json' }
})

export type ApiResponse<T> = {
  data: T
  status: number
}
```

2. **Move match API to feature:**
```typescript
// frontend/app/matches/api.ts
import { apiClient } from '@/lib/infrastructure/api/client'
import type { Match, MatchFilters } from './types'

export async function getMatches(filters?: MatchFilters): Promise<Match[]> {
  const response = await apiClient.get('/matches', { params: filters })
  return response.data
}

export async function getMatch(id: string): Promise<Match> {
  const response = await apiClient.get(`/matches/${id}`)
  return response.data
}
```

3. **Repeat for other features**

4. **Update imports:**
```typescript
// Before:
import { getMatches } from '@/lib/api/matches'

// After:
import { getMatches } from '@/app/matches/api'
```

**Estimated Effort**: 1-2 days

---

### Guide 6: Fixing Cross-Feature Imports

**Current State (example violation):**
```python
# algobet/services/prediction_service.py
from algobet.models import Match, Team                    # Centralized models
from algobet.predictions.data.queries import MatchRepository  # Deep import
from algobet.predictions.features.form_features import FormCalculator
from algobet.predictions.models.registry import ModelRegistry
```

**Target State:**
```python
# algobet/predictions/service.py
from algobet.matches.models import Match     # Import from feature public API
from algobet.teams.models import Team
from .models import Prediction               # Local feature import
from .data.queries import MatchRepository     # Local feature import
```

**Step-by-Step:**

1. **Identify all cross-feature imports:**
```bash
# Find imports from other features
grep -r "from algobet\." --include="*.py" | grep -v "__pycache__"
```

2. **Create public API for each feature:**
```python
# algobet/matches/__init__.py
"""Matches feature public API"""
from .models import Match, MatchStatus
from .service import MatchService
from .dto import MatchCreate, MatchResponse

__all__ = ['Match', 'MatchStatus', 'MatchService', 'MatchCreate', 'MatchResponse']
```

3. **Update imports to use public APIs:**
```python
# Before:
from algobet.models import Match
from algobet.predictions.data.queries import MatchRepository

# After:
from algobet.matches import Match
from algobet.predictions.data.queries import MatchRepository  # Now local
```

4. **Move deep imports to local:**
```python
# Before (in algobet/services/prediction_service.py):
from algobet.predictions.models.registry import ModelRegistry

# After (in algobet/predictions/service.py):
from .models.registry import ModelRegistry  # Relative import
```

**Dependency Rule:**
- Features can import from `infrastructure.*`
- Features can import from other features' public APIs (`from algobet.matches import ...`)
- Features should NOT import internal modules from other features (`from algobet.matches.internal import ...`)

**Estimated Effort**: 2-3 days

---

### Guide 7: Migrating Tests to Feature Structure

**Current State:**
```
tests/
├── conftest.py
├── unit/
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── test_api.py
│   └── test_scraping.py
└── test_scraping_integration.py
```

**Target State:**
```
algobet/                      # Backend features with tests
├── matches/
│   ├── models.py
│   ├── service.py
│   └── tests/               # Co-located tests
│       ├── test_models.py
│       ├── test_service.py
│       └── fixtures/
├── predictions/
│   └── tests/
└── conftest.py              # Shared fixtures

tests/                       # Integration tests only
├── integration/
│   ├── test_api_flows.py
│   └── test_scraping_e2e.py
└── conftest.py
```

**Step-by-Step:**

1. **Create test directories in features:**
```bash
mkdir -p algobet/matches/tests
mkdir -p algobet/predictions/tests
mkdir -p algobet/scraping/tests
```

2. **Move unit tests to features:**
```bash
# Move model tests
mv tests/unit/test_models.py algobet/matches/tests/test_models.py

# Move service tests
mv tests/unit/test_prediction_service.py algobet/predictions/tests/test_service.py
```

3. **Update test imports:**
```python
# Before:
from algobet.models import Match

# After:
from algobet.matches.models import Match
```

4. **Keep integration tests at root:**
```bash
mv tests/integration/* tests/
rmdir tests/integration
```

5. **Update pytest configuration (pyproject.toml or setup.cfg):**
```toml
[tool.pytest.ini_options]
testpaths = [
    "algobet",           # Feature tests
    "tests"              # Integration tests
]
python_files = ["test_*.py", "*_test.py"]
```

**Estimated Effort**: 1-2 days

---

### Guide 8: Infrastructure Migration

**Current State:**
```
algobet/
├── config.py              # Root-level config
├── database.py            # Root-level database
├── exceptions.py          # Root-level exceptions
└── logging_config.py      # Root-level logging
```

**Target State:**
```
algobet/
└── infrastructure/
    ├── config.py          # Configuration management
    ├── database.py        # Database connection/session
    ├── exceptions.py      # Shared exceptions (if truly shared)
    └── logging.py         # Logging configuration
```

**Step-by-Step:**

1. **Create infrastructure package:**
```bash
mkdir -p algobet/infrastructure
```

2. **Move files:**
```bash
mv algobet/config.py algobet/infrastructure/
mv algobet/database.py algobet/infrastructure/
mv algobet/exceptions.py algobet/infrastructure/
mv algobet/logging_config.py algobet/infrastructure/logging.py
```

3. **Create __init__.py:**
```python
# algobet/infrastructure/__init__.py
"""Infrastructure layer - cross-cutting concerns"""
from .config import settings
from .database import get_session, engine
from .exceptions import DatabaseError, ValidationError

__all__ = ['settings', 'get_session', 'engine', 'DatabaseError', 'ValidationError']
```

4. **Update imports across codebase:**
```python
# Before:
from algobet.config import settings
from algobet.database import get_session

# After:
from algobet.infrastructure import settings, get_session
# Or:
from algobet.infrastructure.config import settings
from algobet.infrastructure.database import get_session
```

**Estimated Effort**: 1 day

---

## Migration Checklist

### Pre-Migration
- [ ] Create feature architecture documentation
- [ ] Set up linting rules to prevent new violations
- [ ] Create backup of current codebase
- [ ] Establish CI/CD for testing

### Phase 1: Foundation
- [ ] Create `algobet/infrastructure/` directory
- [ ] Create `algobet/{matches,predictions,scraping,scheduling}/` directories
- [ ] Create `frontend/lib/infrastructure/` directory
- [ ] Update path aliases in tsconfig.json/paths

### Phase 2: Backend
- [ ] Decompose `models.py` into feature models
- [ ] Migrate services to feature folders
- [ ] Move infrastructure files (config, database, exceptions)
- [ ] Fix cross-feature imports
- [ ] Update all import statements
- [ ] Run tests to verify

### Phase 3: Frontend
- [ ] Move components to `app/{feature}/components/`
- [ ] Move hooks to `app/{feature}/hooks/`
- [ ] Move API clients to `app/{feature}/api.ts`
- [ ] Move stores to feature folders (if feature-specific)
- [ ] Organize `components/ui/` as shared primitives only
- [ ] Update all import statements
- [ ] Run tests to verify

### Phase 4: Testing & Cleanup
- [ ] Move unit tests to feature folders
- [ ] Reorganize integration tests
- [ ] Update pytest configuration
- [ ] Run full test suite
- [ ] Remove deprecated folders
- [ ] Update documentation

### Post-Migration
- [ ] Conduct code review
- [ ] Update developer onboarding docs
- [ ] Document new architecture patterns
- [ ] Monitor for issues

---

## Estimated Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Phase 1: Foundation | Week 1 | 3-4 days |
| Phase 2: Backend | Week 2-3 | 8-10 days |
| Phase 3: Frontend | Week 4-5 | 6-8 days |
| Phase 4: Testing & Cleanup | Week 6 | 4-5 days |
| **Total** | **6 weeks** | **~20-25 developer-days** |

---

## Prevention Measures

To prevent future violations:

1. **Linting Rules:**
   - Import restrictions (prevent `from algobet.feature.internal import`)
   - Folder structure enforcement

2. **Code Review Checklist:**
   - Are new features self-contained?
   - Are imports from other features using public APIs?
   - Are components co-located with their pages?

3. **Documentation:**
   - Architecture Decision Records (ADRs)
   - Developer guide for feature development

---

## Benefits of Remediation

After completing this migration:

1. **Modularity**: Features can be understood and modified in isolation
2. **Scalability**: New features follow established patterns
3. **Maintainability**: Clear ownership and boundaries
4. **Testability**: Tests co-located with code
5. **Onboarding**: New developers understand scope quickly
6. **Deployment**: Features can potentially be deployed independently

---

## References

- [Feature-Based Project Structure](https:// featurebasedarchitecture.com/)
- [Vertical Slice Architecture](https://jimmybogard.com/vertical-slice-architecture/)
- [React Folder Structure](https:// react-folder-structure.com/)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

*Document generated: March 25, 2026*
*Version: 1.0*
*Maintainer: Architecture Team*
