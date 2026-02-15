# AlgoBet System Overview

## Introduction

AlgoBet is a comprehensive football match prediction and betting analytics platform that combines web scraping, machine learning, and modern web technologies to provide accurate match predictions and identify value betting opportunities.

## System Components

### 1. Data Collection Layer

**OddsPortal Scraper**
- Playwright-based browser automation
- Handles JavaScript-rendered content
- Extracts match data, results, and betting odds
- Supports both upcoming matches and historical results
- Real-time progress tracking via WebSocket

**Key Features:**
- Headless browser operation
- Automatic retry on failures
- Deduplication of matches
- Tournament and team normalization

### 2. Data Storage Layer

**PostgreSQL Database**
- Relational data model for structured data
- JSONB support for flexible feature storage
- ACID compliance for data integrity
- Full support for complex queries

**Model Registry (File System)**
- Versioned ML model artifacts
- Feature transformers and scalers
- Metadata and configuration files
- Symlink for production model

### 3. Machine Learning Layer

**Prediction Engine**
- XGBoost and LightGBM classifiers
- Ensemble stacking with logistic regression
- Probability calibration (Isotonic/Sigmoid)
- Feature engineering pipeline

**Key Capabilities:**
- Team form calculation (rolling windows)
- Head-to-head statistics
- Tournament context features
- Odds-based value detection

**Performance Targets:**
- Top-1 Accuracy: 50-55%
- Log Loss: < 0.95
- Brier Score: < 0.20
- ROI (Value Bets): 5-10%

### 4. Business Logic Layer (Services)

**Service Architecture**
All business logic is encapsulated in services:

- **PredictionService**: Model inference and batch predictions
- **ScrapingService**: Orchestrates scraping with progress tracking
- **SchedulerService**: APScheduler task management
- **BaseService**: Common patterns for session management

**Benefits:**
- Reusable across CLI, API, and scheduled tasks
- Easy to test in isolation
- Consistent error handling
- Transaction management

### 5. API Layer

**FastAPI Application**
- RESTful endpoints for all operations
- WebSocket support for real-time updates
- Automatic OpenAPI documentation
- Background task execution
- Dependency injection for database sessions

**Router Structure:**
```
/api/v1/matches       - Match CRUD and queries
/api/v1/predictions   - Prediction generation and retrieval
/api/v1/scraping      - Scraping job management
/api/v1/schedules     - Scheduled task management
/api/v1/value-bets    - Value betting opportunities
/api/v1/models        - ML model registry
/api/v1/tournaments   - Tournament information
/api/v1/teams         - Team statistics
```

### 6. Frontend Layer

**Next.js 15 Application**
- Server Components for data fetching
- Client Components for interactivity
- App Router for file-based routing
- TypeScript for type safety

**Key Features:**
- Real-time scraping progress dashboard
- Interactive match analysis
- Prediction confidence visualization
- Schedule management UI
- Responsive design with Tailwind CSS

**State Management:**
- TanStack Query for server state
- Zustand for client state
- URL state for shareable filters

### 7. Automation Layer

**APScheduler Integration**
- Cron-based task scheduling
- Multiple execution options (Docker, systemd, in-process)
- Execution history tracking
- Failure recovery and alerting

**Default Tasks:**
- Daily upcoming match scraping (6 AM, 6 PM)
- Daily prediction generation (7 AM)
- Weekly results scraping (Monday 3 AM)

## Data Flow

### Training Pipeline
```
Raw Data → Scraper → Database → Queries → Features → Model Training → Registry
```

1. Scraper collects historical match data
2. Data stored in PostgreSQL
3. Query repository extracts relevant matches
4. Feature pipeline calculates team form, H2H, etc.
5. Models trained on temporal splits (preventing data leakage)
6. Trained models registered with metadata

### Prediction Pipeline
```
Upcoming Matches → Feature Engineering → Model Loading → Prediction → Storage
```

1. Upcoming matches retrieved from database
2. Features calculated for each match
3. Production model loaded from registry
4. Probabilities generated for H/D/A outcomes
5. Predictions stored with confidence scores

### Scraping Pipeline
```
API Request → Job Creation → Background Task → Progress Updates → Results Storage
```

1. Frontend/API initiates scraping request
2. ScrapingService creates job record
3. Background task runs scraper
4. WebSocket broadcasts progress
5. Results saved to database

## Design Principles

### 1. Service-Oriented Architecture
- Business logic in services, not controllers
- Reusable across different interfaces
- Easy to test and maintain

### 2. API-First Design
- Frontend consumes REST API exclusively
- WebSocket for real-time features
- Enables multiple client types

### 3. Temporal Integrity
- Time-based data splits for ML
- No future data leakage
- Proper handling of match sequencing

### 4. Automation
- Scheduled tasks reduce manual work
- Automatic retry on failures
- Comprehensive logging

### 5. Observability
- Progress tracking for long operations
- Execution history for all tasks
- Performance metrics for models

## Scalability Considerations

### Current Limitations
- In-memory job storage (should use Redis for multi-instance)
- Single scheduler instance
- File-based model registry

### Future Improvements
- Redis for distributed job tracking
- Celery for task queue
- S3/MinIO for model artifacts
- Kubernetes deployment
- Horizontal scaling of API servers

## Security

### Current Implementation
- No authentication (development mode)
- CORS configured for frontend
- SQL injection prevention via SQLAlchemy
- Input validation via Pydantic

### Recommended for Production
- JWT or API key authentication
- Rate limiting
- HTTPS enforcement
- Database connection pooling
- Secrets management (Vault/AWS Secrets)

## Monitoring

### Metrics to Track
- API response times
- Scraping success rates
- Prediction accuracy over time
- Model drift detection
- Database performance
- Scheduler task execution

### Logging
- Structured logging (JSON)
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Correlation IDs for request tracing
- Separate logs for scraper (Playwright output)

## Development Workflow

### Local Development
```bash
# Terminal 1: Backend
uvicorn algobet.api.main:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Scheduler (optional)
algobet-scheduler
```

### Testing
```bash
# Backend tests
pytest

# Frontend tests
cd frontend && npm test

# E2E tests
# (To be implemented)
```

### Code Quality
```bash
# Linting
ruff check .

# Type checking
mypy algobet

# Formatting
ruff format .
```

## Deployment Options

### Docker Compose (Recommended for Single Server)
```bash
docker-compose up -d
```

Includes:
- PostgreSQL database
- FastAPI application
- Frontend (served via Next.js or nginx)
- Scheduler service

### Kubernetes
Future support for:
- Helm charts
- StatefulSets for database
- HorizontalPodAutoscaler for API
- CronJobs for scheduled tasks

### Cloud Deployment
- AWS: ECS/Fargate for API, RDS for database
- GCP: Cloud Run for API, Cloud SQL for database
- Azure: Container Instances, Azure Database

## Troubleshooting

### Common Issues

**Scraper Timeouts**
- Check network connectivity to OddsPortal
- Increase timeout values
- Use non-headless mode for debugging

**Database Connection Errors**
- Verify PostgreSQL is running
- Check DATABASE_URL environment variable
- Ensure database migrations applied

**Model Loading Failures**
- Verify MODELS_PATH exists
- Check model file permissions
- Review model metadata

**WebSocket Connection Issues**
- Check firewall settings
- Verify WebSocket URL configuration
- Review browser console for errors

### Debug Commands
```bash
# Check database status
algobet db-stats

# Test scraper manually
python -c "from algobet.scraper import OddsPortalScraper; ..."

# Verify API health
curl http://localhost:8000/health

# Check scheduler status
algobet-runner --list
```

## Future Roadmap

### Phase 1: Core Features (✅ Complete)
- [x] Database and scraper
- [x] ML prediction engine
- [x] REST API
- [x] React frontend
- [x] Automated scheduling

### Phase 2: Advanced Analytics (In Progress)
- [ ] Live match odds tracking
- [ ] Advanced statistical models
- [ ] Portfolio optimization for bets
- [ ] Backtesting framework

### Phase 3: Scale & Operations
- [ ] User authentication
- [ ] Multi-tenant support
- [ ] Advanced monitoring
- [ ] Mobile app

### Phase 4: AI Enhancements
- [ ] Deep learning models
- [ ] Natural language processing for news
- [ ] Computer vision for match analysis
- [ ] Reinforcement learning for betting strategies

## Resources

### Documentation
- [API Documentation](api/README.md)
- [Frontend Development Plan](frontend_development_plan.md)
- [Prediction Engine](prediction_engine_architecture.md)
- [ML Model Design](ml_model_design.md)
- [Refactoring Plan](refactor.md)

### External References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## Glossary

| Term | Definition |
|------|------------|
| **1X2** | Three-way betting market: 1=Home, X=Draw, 2=Away |
| **Cron** | Time-based job scheduler (Unix) |
| **EV** | Expected Value - measure of bet profitability |
| **H2H** | Head-to-Head - historical matchups |
| **Kelly** | Kelly Criterion - optimal bet sizing formula |
| **ORM** | Object-Relational Mapping |
| **ROI** | Return on Investment |
| **WebSocket** | Protocol for full-duplex communication |
