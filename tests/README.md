# AlgoBet API Tests

Critical tests for the FastAPI backend API.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=algobet.api --cov-report=html

# Run specific test file
pytest tests/test_schemas.py

# Run specific test class
pytest tests/test_api.py::TestTournamentsEndpoints

# Run specific test
pytest tests/test_api.py::TestTournamentsEndpoints::test_list_tournaments_empty
```

## Test Coverage

### `test_schemas.py`
- Pydantic schema validation
- Tournament, Season, Team schemas
- Match schema with status and result validation
- Prediction schema with probability and confidence validation
- Model version schema with accuracy validation
- Form breakdown schema

### `test_api.py`
- Root and health endpoints
- Tournaments API (list, get, seasons)
- Teams API (list, search, get)
- Matches API (list, filter, get, preview, h2h)
- Predictions API (list, generate, upcoming, history, value-bets)
- Models API (list, active, get, activate, delete, metrics)
- CORS configuration

### `test_dependencies.py`
- Database session dependency injection
- Session query functionality

## Test Database

Tests use an in-memory SQLite database for fast, isolated test execution.

## Dependencies

- `pytest` - Testing framework
- `httpx` - HTTP client for TestClient
- `fastapi.testclient.TestClient` - API testing utility
