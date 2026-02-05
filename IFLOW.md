# AlgoBet Instruction Flow

This document describes the data flow, workflow patterns, and instruction flow through the AlgoBet system.

## Overview

AlgoBet is a football match database and OddsPortal scraper that follows a specific instruction flow from CLI commands through to database operations and ML predictions.

## Command Flow

### 1. CLI Entry Point → Command Handler

```
User Input
    ↓
cli.py (@click.group)
    ↓
Specific Command (@cli.command)
    ↓
Business Logic / Scraper / Export
    ↓
session_scope() → Database
```

### 2. API Flow → HTTP Request Handler

```
HTTP Request
    ↓
FastAPI main.py (app initialization)
    ↓
Router (matches, predictions, teams, etc.)
    ↓
Dependency Injection (get_db())
    ↓
Business Logic / Database Queries
    ↓
Pydantic Response Models
    ↓
JSON Response
```

### 2. CLI Commands (cli.py)

| Command | Purpose | Flow |
|---------|---------|------|
| `scrape` | Scrape match data from OddsPortal | `cli.scrape()` → `OddsPortalScraper.scrape_tournament()` |
| `export` | Export match data | `cli.export()` → `MatchRepository` queries → File output |
| `init` | Initialize database | `cli.init()` → `init_db()` → Create tables |
| `seasons` | Manage seasons | `cli.seasons()` → Season management logic |

### 3. Prediction CLI Commands (predictions_cli.py)

| Command | Purpose | Flow |
|---------|---------|------|
| `train` | Train ML prediction model | `predictions.train()` → Feature engineering → Model training → ModelRegistry |
| `predict` | Generate match predictions | `predictions.predict()` → Load active model → Feature extraction → Predictions |
| `evaluate` | Evaluate model performance | `predictions.evaluate()` → Load test data → Calculate metrics → Report |
| `list-models` | List available models | `predictions.list_models()` → Query model_versions → Display table |
| `activate-model` | Activate model version | `predictions.activate()` → Update is_active flag → Model ready for inference |
| `export-predictions` | Export predictions to file | `predictions.export()` → Query predictions → CSV/JSON output |

### 3. API Endpoints (FastAPI)

| Endpoint | Purpose | Flow |
|----------|---------|------|
| `/api/v1/tournaments` | Tournament management | FastAPI → TournamentRouter → Database |
| `/api/v1/seasons` | Season management | FastAPI → SeasonRouter → Database |
| `/api/v1/teams` | Team information | FastAPI → TeamRouter → Database |
| `/api/v1/matches` | Match data and filtering | FastAPI → MatchRouter → MatchRepository |
| `/api/v1/predictions` | Prediction management | FastAPI → PredictionRouter → ModelRegistry |
| `/api/v1/models` | ML model management | FastAPI → ModelRouter → ModelRegistry |
| `/api/v1/value-bets` | Value bet identification | FastAPI → ValueBetsRouter → Business Logic |

## Data Flow Patterns

### Pattern 0: API Request Flow

```
HTTP Request
    ↓
FastAPI Router (Path Operation Function)
    ↓
Dependency Injection (get_db() → session_scope)
    ↓
Business Logic / Repository Pattern
    ↓
Database Operations (SQLAlchemy)
    ↓
Pydantic Model Serialization
    ↓
JSON Response
```

**Key Components:**
- [`get_db()`](algobet/api/dependencies.py) - FastAPI dependency for database sessions
- [`session_scope()`](algobet/database.py) - Context manager for database transactions
- Pydantic schemas in [`algobet/api/schemas/`](algobet/api/schemas/) - Request/response validation
- FastAPI routers in [`algobet/api/routers/`](algobet/api/routers/) - API endpoint definitions

### Pattern 1: Scraping Flow

```
OddsPortalScraper (scraper.py)
    ↓
headless browser navigation
    ↓
HTML parsing (BeautifulSoup)
    ↓
Data extraction (tournaments, teams, matches)
    ↓
session_scope() context
    ↓
get_or_create_pattern (Tournament/Team/Match)
    ↓
Database INSERT/UPDATE
```

**Key Functions:**
- [`OddsPortalScraper.__aenter__()`](algobet/scraper.py) - Initialize browser
- [`OddsPortalScraper.scrape_tournament()`](algobet/scraper.py) - Scrape tournament data
- [`get_or_create_tournament()`](algobet/models.py) - Get or create tournament
- [`get_or_create_team()`](algobet/models.py) - Get or create team
- [`save_match()`](algobet/models.py) - Save match with deduplication

### Pattern 2: Query Flow

```
MatchRepository (predictions/data/queries.py)
    ↓
session_scope() context
    ↓
SQLAlchemy queries
    ↓
Data transformation
    ↓
Return match data
```

**Key Methods:**
- [`MatchRepository.get_historical_matches()`](algobet/predictions/data/queries.py) - Get matches by date range
- [`MatchRepository.get_team_matches()`](algobet/predictions/data/queries.py) - Get matches for a team
- [`MatchRepository.get_h2h_matches()`](algobet/predictions/data/queries.py) - Get head-to-head matches

### Pattern 3: Feature Engineering Flow

```
FormCalculator (predictions/features/form_features.py)
    ↓
MatchRepository queries
    ↓
Calculate form metrics
    ↓
Return features dict
```

**Key Methods:**
- [`FormCalculator.calculate_recent_form()`](algobet/predictions/features/form_features.py) - Calculate recent form
- [`FormCalculator.calculate_home_away_form()`](algobet/predictions/features/form_features.py) - Calculate venue-specific form
- [`FormCalculator.calculate_goal_stats()`](algobet/predictions/features/form_features.py) - Calculate goal statistics

### Pattern 4: Model Registry Flow

```
ModelRegistry (predictions/models/registry.py)
    ↓
Save model → file storage
    ↓
Database record (model_versions table)
    ↓
Activate model → update is_active flag
    ↓
Load model → inference
```

**Key Methods:**
- [`ModelRegistry.save_model()`](algobet/predictions/models/registry.py) - Save trained model
- [`ModelRegistry.activate_model()`](algobet/predictions/models/registry.py) - Activate model version
- [`ModelRegistry.load_model()`](algobet/predictions/models/registry.py) - Load model for inference

## Database Transaction Flow

### session_scope() Context Manager (CLI/Background Tasks)

```
session_scope() enter
    ↓
Create new session
    ↓
Execute business logic
    ↓
session_scope() exit
    ↓
Commit on success, Rollback on exception
```

**Usage Pattern:**
```python
with session_scope() as session:
    # All database operations here
    session.add(entity)
    session.flush()  # Get IDs without committing
# Automatic commit/rollback on exit
```

### FastAPI Dependency Injection (API Requests)

```
HTTP Request
    ↓
get_db() dependency called
    ↓
session_scope() context manager
    ↓
Database session yielded to endpoint
    ↓
Endpoint executes business logic
    ↓
Response returned to client
    ↓
Automatic commit/rollback on exit
```

**Usage Pattern:**
```python
@router.get("/matches")
def get_matches(db: Session = Depends(get_db)):
    # Database operations with automatic session management
    return db.query(Match).all()
```

```
session_scope() enter
    ↓
Create new session
    ↓
Execute business logic
    ↓
session_scope() exit
    ↓
Commit on success, Rollback on exception
```

**Usage Pattern:**
```python
with session_scope() as session:
    # All database operations here
    session.add(entity)
    session.flush()  # Get IDs without committing
# Automatic commit/rollback on exit
```

## Scraper Instruction Flow

### OddsPortalScraper Lifecycle

```
with OddsPortalScraper(headless=True) as scraper:
    # __aenter__ → launch browser
    ↓
    scraper.scrape_tournament(url)
    ↓
    navigate_to(url)
    ↓
    parse_results_page()
    ↓
    extract_match_data()
    ↓
    save_to_database()
    ↓
# __aexit__ → close browser
```

### Page Parsing Flow

```
Page HTML
    ↓
BeautifulSoup.find_all('tr', class_='table-odds')
    ↓
Extract team names, scores, dates
    ↓
Parse odds if available
    ↓
Build Match objects
    ↓
Deduplicate and save
```

## Model Training Flow

```
Training Data (historical matches)
    ↓
Feature Engineering (FormCalculator)
    ↓
Feature Matrix (pandas/numpy)
    ↓
Train model (sklearn/xgboost)
    ↓
Evaluate (metrics)
    ↓
Save (ModelRegistry)
    ↓
Activate (production)
```

## Prediction Flow

```
Input: upcoming match
    ↓
Feature extraction (FormCalculator)
    ↓
Build feature vector
    ↓
Load active model (ModelRegistry)
    ↓
Predict
    ↓
Return probability/outcome
```

## File Structure Reference

```
algobet/
├── cli.py              # CLI entry point, command routing
├── database.py         # session_scope(), engine setup
├── models.py           # ORM models, get_or_create helpers
├── scraper.py          # OddsPortalScraper, page parsing
├── api/                # FastAPI application layer
│   ├── main.py         # FastAPI app initialization
│   ├── dependencies.py # FastAPI dependency injection
│   ├── routers/        # API endpoint routers
│   │   ├── matches.py  # Match endpoints
│   │   ├── predictions.py # Prediction endpoints
│   │   ├── teams.py    # Team endpoints
│   │   ├── tournaments.py # Tournament endpoints
│   │   ├── seasons.py  # Season endpoints
│   │   ├── models.py   # ML model endpoints
│   │   └── value_bets.py # Value bet endpoints
│   └── schemas/        # Pydantic request/response models
│       ├── match.py    # Match schemas
│       ├── prediction.py # Prediction schemas
│       ├── team.py     # Team schemas
│       ├── tournament.py # Tournament schemas
│       ├── model.py    # Model schemas
│       └── common.py   # Common schema components
└── predictions/
    ├── data/
    │   └── queries.py  # MatchRepository queries
    ├── features/
    │   └── form_features.py  # FormCalculator
    └── models/
        └── registry.py  # ModelRegistry
```

## Error Handling Flow

```
Exception occurs
    ↓
session_scope() catches
    ↓
Rollback transaction
    ↓
Log error
    ↓
Re-raise / Handle gracefully
```

## Validation Flow

```
User input / scraped data
    ↓
Type validation (Pydantic/type hints)
    ↓
Database constraints (unique, foreign key)
    ↓
Business logic validation
    ↓
Raise ValidationError if invalid
```

## Environment Configuration Flow

### CLI/Scraper Configuration

```
.env file
    ↓
python-dotenv loads variables
    ↓
database.py reads POSTGRES_*
    ↓
Create SQLAlchemy engine
    ↓
Connection pool ready
```

### FastAPI Application Configuration

```
Environment Variables (.env)
    ↓
FastAPI app initialization (main.py)
    ↓
CORS configuration (origins, methods, headers)
    ↓
Router registration with prefixes
    ↓
Dependency injection setup
    ↓
Application ready for requests
```

**Key Configuration:**
- CORS origins: `http://localhost:3000`, `http://127.0.0.1:3000`
- API prefix: `/api/v1`
- Documentation: `/docs` (Swagger), `/redoc` (ReDoc)
- Health check: `/health`

## Development Workflows

### Code Quality and Linting

```
Pre-commit hooks (.pre-commit-config.yaml)
    ↓
Ruff linting and formatting
    ↓
MyPy type checking
    ↓
Basic file checks (trailing whitespace, YAML, large files)
    ↓
Automatic fixes applied
    ↓
Commit proceeds if all checks pass
```

**Development Commands:**
- `pre-commit install` - Install git hooks
- `pre-commit run --all-files` - Run all checks manually
- `ruff check .` - Run linting only
- `ruff format .` - Run formatting only
- `mypy algobet/` - Run type checking only

## Common Workflows

### Workflow 0: API Server Startup

1. Run `uvicorn algobet.api.main:app --host 0.0.0.0 --port 8000 --reload`
2. FastAPI initializes application with lifespan context
3. Routers are registered with prefixes and tags
4. CORS middleware is configured
5. Application ready to handle HTTP requests at `http://localhost:8000`

### Workflow 1: API Match Query

1. Client sends GET request to `/api/v1/matches`
2. FastAPI matches route to `matches_router.list_matches()`
3. Dependency injection provides database session via `get_db()`
4. Query parameters parsed and validated by Pydantic
5. `MatchRepository` queries database with filters
6. Results serialized to `MatchResponse` schema
7. JSON response returned to client

### Workflow 2: API Prediction Request

1. Client sends POST request to `/api/v1/predictions`
2. FastAPI validates request body against Pydantic schema
3. `predictions_router` processes prediction request
4. Active model loaded via `ModelRegistry`
5. Features generated using `FormCalculator`
6. Model prediction executed
7. Results saved to database and returned as JSON

### Workflow 3: Scrape New Tournament

1. User runs `python -m algobet scrape --url <url>`
2. CLI validates URL format
3. Scraper navigates to OddsPortal
4. Parses tournament/season/team data
5. Saves to database with `session_scope()`
6. Returns summary of scraped data

### Workflow 2: Export Historical Data

1. User runs `python -m algobet export --tournament <name> --start-date <date>`
2. CLI parses arguments
3. `MatchRepository.get_historical_matches()` queries database
4. Data converted to CSV/JSON
5. File written to specified path

### Workflow 3: Train New Model

1. User runs prediction training command
2. `MatchRepository` fetches historical matches
3. `FormCalculator` generates features
4. Model trained with sklearn/xgboost
5. `ModelRegistry.save_model()` persists model
6. `ModelRegistry.activate_model()` sets as active

### Workflow 5: Docker Deployment

1. **Database Setup**: `docker-compose up -d db`
   - PostgreSQL container starts with health checks
   - Database initialized with algobet user and football database
   - Port 5432 exposed for external connections

2. **API Deployment**: `docker-compose up -d api`
   - FastAPI container built from Dockerfile.api
   - UV package manager installs dependencies
   - Application code mounted as volume for development
   - Port 8000 exposed for HTTP requests
   - Automatically connects to database service

3. **Development Mode**: `docker-compose up`
   - Both services start with dependency ordering
   - Hot reload enabled for code changes
   - Shared volumes for data persistence
   - Environment variables configured for development

### Workflow 7: Train Prediction Model

1. User runs `python -m algobet predictions train --tournament <name> --start-date <date>`
2. CLI validates arguments and date ranges
3. `MatchRepository` fetches historical matches for training
4. `FormCalculator` generates feature matrix from match data
5. Model trained using scikit-learn/xgboost algorithms
6. `ModelRegistry.save_model()` persists trained model to disk
7. Model metadata saved to `model_versions` table
8. Training metrics and validation results displayed

### Workflow 8: Generate Predictions

1. User runs `python -m algobet predictions predict --match-id <id>`
2. CLI loads active model via `ModelRegistry.load_model()`
3. Recent match data fetched for feature calculation
4. `FormCalculator` generates prediction features
5. Model prediction executed with probability outputs
6. Results optionally saved to database
7. Predictions displayed in formatted table

### Workflow 9: Export Predictions

1. User runs `python -m algobet predictions export-predictions --format csv`
2. CLI queries database for prediction records
3. Results formatted as CSV or JSON
4. Output written to file or stdout
5. Optional filtering by date range, tournament, or model

### Workflow 11: Run Tests

1. **Unit Tests**: `pytest tests/` or `python -m pytest`
   - Test configuration loaded from `pytest.ini`
   - Test files follow `test_*.py` pattern
   - Database fixtures in `conftest.py` setup test environment
   - API, dependencies, and schemas tested separately

2. **Test Database Setup**:
   - Separate test database configuration
   - Fixtures create clean database state
   - Automatic cleanup after test completion

3. **Test Categories**:
   - `test_api.py` - FastAPI endpoint testing
   - `test_dependencies.py` - Dependency injection testing
   - `test_schemas.py` - Pydantic schema validation testing

### Workflow 13: Make Predictions

1. Load active model via `ModelRegistry`
2. Get recent match data for teams
3. Generate features with `FormCalculator`
4. Run model.predict()
5. Return probability scores
