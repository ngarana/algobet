# AlgoBet API Documentation

## Overview

The AlgoBet API is a RESTful API built with FastAPI that provides access to football match data, predictions, scraping operations, and schedule management.

**Base URL**: `http://localhost:8000/api/v1`

**WebSocket URL**: `ws://localhost:8000/ws`

## Authentication

Currently, the API does not require authentication. Future versions may implement API key or JWT-based authentication.

## Content Types

All requests and responses use JSON format:

```http
Content-Type: application/json
```

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "data": { ... }
}
```

### Error Response
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Endpoints

### Matches

#### List Matches
```http
GET /matches
```

Query Parameters:
- `status` (string): Filter by status - 'SCHEDULED', 'FINISHED', 'LIVE'
- `tournament_id` (int): Filter by tournament
- `season_id` (int): Filter by season
- `team_id` (int): Filter by team (home or away)
- `from_date` (string): ISO 8601 datetime
- `to_date` (string): ISO 8601 datetime
- `days_ahead` (int): Number of days ahead for upcoming matches
- `has_odds` (bool): Filter matches with available odds
- `limit` (int): Results per page (default: 50, max: 100)
- `offset` (int): Pagination offset

Response:
```json
[
  {
    "id": 1,
    "tournament_id": 1,
    "season_id": 1,
    "home_team_id": 1,
    "away_team_id": 2,
    "match_date": "2026-02-15T15:00:00",
    "home_score": null,
    "away_score": null,
    "status": "SCHEDULED",
    "odds_home": 2.10,
    "odds_draw": 3.40,
    "odds_away": 3.20,
    "num_bookmakers": 12
  }
]
```

#### Get Match Details
```http
GET /matches/{id}
```

Response:
```json
{
  "id": 1,
  "tournament_id": 1,
  "season_id": 1,
  "home_team_id": 1,
  "away_team_id": 2,
  "match_date": "2026-02-15T15:00:00",
  "home_score": null,
  "away_score": null,
  "status": "SCHEDULED",
  "odds_home": 2.10,
  "odds_draw": 3.40,
  "odds_away": 3.20,
  "num_bookmakers": 12,
  "tournament": { "id": 1, "name": "Premier League", "country": "England" },
  "home_team": { "id": 1, "name": "Manchester United" },
  "away_team": { "id": 2, "name": "Arsenal" }
}
```

#### Get Head-to-Head
```http
GET /matches/{id}/h2h
```

Returns historical matches between the two teams.

---

### Predictions

#### List Predictions
```http
GET /predictions
```

Query Parameters:
- `match_id` (int): Filter by specific match
- `model_version_id` (int): Filter by model version
- `has_result` (bool): Filter by matches with actual results
- `from_date` (string): ISO 8601 datetime
- `to_date` (string): ISO 8601 datetime
- `min_confidence` (float): Minimum confidence threshold (0-1)
- `limit` (int): Results per page
- `offset` (int): Pagination offset

Response:
```json
[
  {
    "id": 1,
    "match_id": 1,
    "model_version_id": 1,
    "prob_home": 0.45,
    "prob_draw": 0.28,
    "prob_away": 0.27,
    "predicted_outcome": "H",
    "confidence": 0.72,
    "predicted_at": "2026-02-14T10:00:00",
    "match": { ... }
  }
]
```

#### Get Upcoming Predictions
```http
GET /predictions/upcoming?days=7
```

Returns predictions for scheduled matches in the next N days.

#### Generate Predictions
```http
POST /predictions/generate
```

Request Body:
```json
{
  "match_ids": [1, 2, 3],
  "model_version": "v1.0.0",
  "tournament_id": null,
  "days_ahead": 7
}
```

Response:
```json
{
  "generated": 3,
  "predictions": [ ... ]
}
```

---

### Scraping

#### Scrape Upcoming Matches
```http
POST /scraping/upcoming
```

Request Body:
```json
{
  "url": "https://www.oddsportal.com/matches/football/"
}
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "job_type": "upcoming",
  "url": "https://www.oddsportal.com/matches/football/",
  "status": "pending",
  "created_at": "2026-02-15T10:00:00"
}
```

#### Scrape Historical Results
```http
POST /scraping/results
```

Request Body:
```json
{
  "url": "https://www.oddsportal.com/football/england/premier-league/results/",
  "max_pages": 5
}
```

#### List Scraping Jobs
```http
GET /scraping/jobs
```

Query Parameters:
- `status` (string): Filter by status - 'pending', 'running', 'completed', 'failed', 'cancelled'

Response:
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "job_type": "results",
    "url": "https://www.oddsportal.com/...",
    "status": "completed",
    "created_at": "2026-02-15T10:00:00",
    "progress": {
      "current_page": 5,
      "total_pages": 5,
      "matches_scraped": 120,
      "matches_saved": 115,
      "message": "Completed! Scraped 120 matches from 5 pages, saved 115."
    }
  }
]
```

#### Get Job Status
```http
GET /scraping/jobs/{job_id}
```

---

### Schedules

#### List Scheduled Tasks
```http
GET /schedules
```

Response:
```json
[
  {
    "id": 1,
    "name": "daily-upcoming-scrape",
    "task_type": "scrape_upcoming",
    "cron_expression": "0 6 * * *",
    "is_active": true,
    "last_run_at": "2026-02-15T06:00:00",
    "next_run_at": "2026-02-16T06:00:00"
  }
]
```

#### Create Schedule
```http
POST /schedules
```

Request Body:
```json
{
  "name": "evening-scrape",
  "task_type": "scrape_upcoming",
  "cron_expression": "0 18 * * *",
  "config": {
    "url": "https://www.oddsportal.com/matches/football/"
  },
  "is_active": true
}
```

#### Get Schedule
```http
GET /schedules/{id}
```

#### Update Schedule
```http
PATCH /schedules/{id}
```

Request Body:
```json
{
  "cron_expression": "0 12 * * *",
  "is_active": false
}
```

#### Delete Schedule
```http
DELETE /schedules/{id}
```

#### Run Task Now
```http
POST /schedules/{id}/run
```

Triggers immediate execution of the scheduled task.

#### Get Execution History
```http
GET /schedules/{id}/history
```

Response:
```json
[
  {
    "id": 1,
    "scheduled_task_id": 1,
    "status": "completed",
    "started_at": "2026-02-15T06:00:00",
    "completed_at": "2026-02-15T06:05:30",
    "matches_scraped": 45,
    "matches_saved": 45,
    "triggered_by": "scheduler"
  }
]
```

---

### Value Bets

#### Find Value Bets
```http
GET /value-bets
```

Query Parameters:
- `min_ev` (float): Minimum expected value threshold (default: 0.05)
- `max_odds` (float): Maximum odds to consider (default: 10.0)
- `days` (int): Days ahead to search (default: 7)
- `model_version` (string): Specific model version
- `min_confidence` (float): Minimum confidence threshold
- `max_matches` (int): Maximum matches to return (default: 20)

Response:
```json
[
  {
    "match_id": 1,
    "outcome": "H",
    "predicted_probability": 0.65,
    "market_odds": 2.10,
    "expected_value": 0.15,
    "kelly_fraction": 0.08,
    "match": { ... }
  }
]
```

---

### Models

#### List Models
```http
GET /models
```

Query Parameters:
- `algorithm` (string): Filter by algorithm type
- `active_only` (bool): Return only active model

#### Get Active Model
```http
GET /models/active
```

#### Get Model Details
```http
GET /models/{id}
```

#### Activate Model
```http
POST /models/{id}/activate
```

#### Delete Model
```http
DELETE /models/{id}
```

---

### Tournaments

#### List Tournaments
```http
GET /tournaments
```

#### Get Tournament
```http
GET /tournaments/{id}
```

#### Get Tournament Seasons
```http
GET /tournaments/{id}/seasons
```

---

### Teams

#### List Teams
```http
GET /teams
```

Query Parameters:
- `search` (string): Search by name
- `tournament_id` (int): Filter by tournament
- `limit` (int)
- `offset` (int)

#### Get Team
```http
GET /teams/{id}
```

#### Get Team Form
```http
GET /teams/{id}/form?n_matches=5
```

#### Get Team Matches
```http
GET /teams/{id}/matches?venue=home&limit=10
```

---

## WebSocket

### Scraping Progress

Connect to receive real-time updates during scraping operations.

```
ws://localhost:8000/ws/scraping/{job_id}
```

Messages:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "current_page": 3,
  "total_pages": 10,
  "matches_scraped": 45,
  "matches_saved": 42,
  "message": "Scraping page 3/10...",
  "error": null
}
```

Status values:
- `pending`: Job queued
- `running`: Actively scraping
- `completed`: Successfully finished
- `failed`: Error occurred
- `cancelled`: Manually cancelled

### Example WebSocket Client (JavaScript)

```javascript
const jobId = '550e8400-e29b-41d4-a716-446655440000';
const ws = new WebSocket(`ws://localhost:8000/ws/scraping/${jobId}`);

ws.onopen = () => {
  console.log('Connected to progress updates');
};

ws.onmessage = (event) => {
  const progress = JSON.parse(event.data);

  if (progress.status === 'running') {
    const percent = (progress.current_page / progress.total_pages) * 100;
    console.log(`Progress: ${percent.toFixed(1)}%`);
    console.log(`${progress.matches_scraped} matches scraped`);
  } else if (progress.status === 'completed') {
    console.log('Scraping completed!');
    ws.close();
  } else if (progress.status === 'failed') {
    console.error('Scraping failed:', progress.error);
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};
```

---

## Error Codes

| Status Code | Meaning | Description |
|-------------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded, no content returned |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error occurred |

## Rate Limiting

Currently no rate limiting is implemented. Future versions may include:
- 100 requests per minute per IP
- WebSocket connection limits

## Pagination

List endpoints support pagination via `limit` and `offset` parameters:

```http
GET /matches?limit=50&offset=100
```

Response includes total count in headers:
```http
X-Total-Count: 1500
```

## Date Formats

All dates use ISO 8601 format:
- Date: `2026-02-15`
- DateTime: `2026-02-15T15:00:00` or `2026-02-15T15:00:00Z` (UTC)

## Cron Expressions

Scheduled tasks use standard cron expressions:

| Expression | Description |
|------------|-------------|
| `0 6 * * *` | Daily at 6:00 AM |
| `0 */6 * * *` | Every 6 hours |
| `0 0 * * 1` | Weekly on Monday at midnight |
| `0 18 * * *` | Daily at 6:00 PM |

Format: `minute hour day month day_of_week`

## Testing

### Using curl

```bash
# Get all matches
curl http://localhost:8000/api/v1/matches

# Start scraping
curl -X POST http://localhost:8000/api/v1/scraping/upcoming \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.oddsportal.com/matches/football/"}'

# Get predictions
curl "http://localhost:8000/api/v1/predictions/upcoming?days=7"

# Create schedule
curl -X POST http://localhost:8000/api/v1/schedules \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-task",
    "task_type": "scrape_upcoming",
    "cron_expression": "0 12 * * *"
  }'
```

### Using HTTPie

```bash
# Install HTTPie
pip install httpie

# Get matches
http :8000/api/v1/matches

# Scrape upcoming
http POST :8000/api/v1/scraping/upcoming \
  url="https://www.oddsportal.com/matches/football/"

# Get value bets
http :8000/api/v1/value-bets min_ev==0.1 days==3
```

## OpenAPI Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## SDKs and Client Libraries

### JavaScript/TypeScript

```typescript
// lib/api/client.ts example
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }

  return response.json();
}

// Usage
const matches = await fetchAPI('/api/v1/matches?status=SCHEDULED');
```

### Python

```python
import httpx

async def get_matches(status: str = None):
    params = {"status": status} if status else {}
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/api/v1/matches",
            params=params
        )
        response.raise_for_status()
        return response.json()
```

## Changelog

### v1.0.0 (2026-02-14)
- Initial API release
- All core endpoints implemented
- WebSocket support for scraping progress
- Schedule management
- Complete test coverage (155 tests)

## Support

For API issues or questions:
- Check the [main README](../README.md)
- Review test files for usage examples
- Examine frontend code in `/frontend/lib/api/`
