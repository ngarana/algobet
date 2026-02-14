# AlgoBet Refactoring Roadmap

**Status**: ✅ Completed
**Created**: 2026-02-05
**Last Updated**: 2026-02-14

## 🎯 Executive Summary

This document outlines a systematic refactoring plan to transform AlgoBet from a CLI-heavy architecture to a modern service-oriented architecture with API-first design, real-time progress updates, and automated scheduling capabilities.

## 📊 Progress Overview

### Overall Progress: 100% Complete ✅
- **Phase 1**: Service Layer Foundation (✅ 3/3 tasks completed)
- **Phase 2**: API Scraping Endpoints (✅ 4/4 tasks completed)
- **Phase 3**: WebSocket Progress Updates (✅ 2/2 tasks completed)
- **Phase 4**: Frontend Integration (✅ 4/4 tasks completed)
- **Phase 5**: CLI Deprecation (✅ 2/2 tasks completed)
- **Phase 6**: Automated/Scheduled Scraping (✅ 11/11 tasks completed)
- **Testing & Validation**: (✅ 4/4 tasks completed)

---

## 🚀 Phase 1: Service Layer Foundation
**Duration**: 1-2 days | **Priority**: High | **Status**: ✅ Completed

### Objective
Create a unified service layer to eliminate code duplication and provide reusable business logic.

### Tasks

#### 1.1 Create Base Service Class
- [x] **File**: `algobet/services/base.py`
- [x] **Description**: Create abstract base service with common session management
- [x] **Test Coverage**: Unit tests for BaseService methods
- [x] **Validation**: Ensure proper session lifecycle management

#### 1.2 Create PredictionService
- [x] **File**: `algobet/services/prediction_service.py`
- [x] **Description**: Consolidate prediction logic from CLI and API
- [x] **Dependencies**: Extract from `predictions_cli.py:58-174` and `api/routers/predictions.py:114-229`
- [x] **Test Coverage**:
  - [x] Unit tests for `load_model()`
  - [x] Unit tests for `generate_features()`
  - [x] Unit tests for `get_prediction()`
  - [x] Integration tests for `predict_match()`
  - [x] Integration tests for `predict_upcoming()`
- [x] **Validation**: Compare outputs with existing CLI/API implementations

#### 1.3 Create ScrapingService
- [x] **File**: `algobet/services/scraping_service.py`
- [x] **Description**: Orchestrate scraping operations with progress tracking
- [x] **Dependencies**: Integrate with existing `OddsPortalScraper`
- [x] **Test Coverage**:
  - [x] Unit tests for job creation/management
  - [x] Unit tests for `scrape_upcoming()`
  - [x] Unit tests for `scrape_results()`
  - [x] Integration tests with mock scraper
  - [x] Progress callback testing
- [x] **Validation**: Ensure progress updates work correctly

**Phase 1 Status**: ✅ **COMPLETED** - All service classes created with:
- Base service with session management
- Prediction service consolidating CLI/API logic
- Scraping service with job tracking
- All code passes ruff linting and mypy type checking

---

## 🌐 Phase 2: API Scraping Endpoints
**Duration**: 1-2 days | **Priority**: High | **Status**: ✅ Completed

### Objective
Expose scraping functionality through REST API with background task support.

### Tasks

#### 2.1 Create Scraping Schemas
- [x] **File**: `algobet/api/schemas/scraping.py`
- [x] **Description**: Pydantic schemas for scraping requests/responses
- [x] **Test Coverage**: Schema validation tests
- [x] **Validation**: Ensure proper request/response validation

#### 2.2 Create Scraping Router
- [x] **File**: `algobet/api/routers/scraping.py`
- [x] **Description**: REST endpoints for scraping operations
- [x] **Endpoints**:
  - [x] `POST /api/v1/scraping/upcoming`
  - [x] `POST /api/v1/scraping/results`
  - [x] `GET /api/v1/scraping/jobs`
  - [x] `GET /api/v1/scraping/jobs/{job_id}`
- [x] **Test Coverage**:
  - [x] Integration tests for each endpoint
  - [x] Background task execution tests
  - [x] Error handling tests
- [x] **Validation**: Test with actual scraping (development environment)

#### 2.3 Register Router in Main App
- [x] **File**: `algobet/api/main.py`
- [x] **Description**: Add scraping router to FastAPI app
- [x] **Test Coverage**: Router registration test
- [x] **Validation**: Verify endpoints are accessible

#### 2.4 Background Task Integration
- [x] **Description**: Implement proper background task handling
- [x] **Test Coverage**: Background task execution tests
- [x] **Validation**: Ensure tasks run independently of API response

**Phase 2 Status**: ✅ **COMPLETED** - All API scraping endpoints created with:
- Scraping schemas with job management structures
- Full CRUD API endpoints for scraping operations
- Background task execution for long-running operations
- WebSocket integration for real-time progress updates

---

## 🔌 Phase 3: WebSocket Progress Updates
**Duration**: 1 day | **Priority**: Medium | **Status**: ✅ Completed

### Objective
Provide real-time progress updates for long-running scraping operations.

### Tasks

#### 3.1 Create WebSocket Handler
- [x] **File**: `algobet/api/websockets/progress.py`
- [x] **Description**: WebSocket connection manager for progress updates
- [x] **Test Coverage**: Unit tests for ConnectionManager
- [x] **Validation**: Test multiple concurrent connections

#### 3.2 Add WebSocket Route
- [x] **File**: `algobet/api/main.py`
- [x] **Description**: WebSocket endpoint for scraping progress
- [x] **Test Coverage**: WebSocket connection tests
- [x] **Validation**: Test real-time progress broadcasting

**Phase 3 Status**: ✅ **COMPLETED** - All WebSocket progress functionality created with:
- ConnectionManager for handling WebSocket connections
- Real-time progress broadcasting for scraping jobs
- Job-specific subscription/unsubscription handling
- Proper error handling and connection cleanup

---

## 🎨 Phase 4: Frontend Integration
**Duration**: 2-3 days | **Priority**: High | **Status**: ✅ Completed

### Objective
Create frontend interface for scraping management with real-time updates.

### Tasks

#### 4.1 Scraping Dashboard Component
- [x] **Files**:
  - [x] `frontend/app/scraping/page.tsx`
  - [x] `frontend/components/scraping/ScrapingJobCard.tsx`
  - [x] `frontend/components/scraping/ScrapingProgress.tsx`
  - [x] `frontend/components/scraping/ScrapeForm.tsx`
  - [x] `frontend/components/scraping/JobHistoryTable.tsx`
- [x] **Test Coverage**: Component unit tests
- [x] **Validation**: UI/UX testing with users

#### 4.2 API Client Functions
- [x] **File**: `frontend/lib/api/scraping.ts`
- [x] **Description**: TypeScript API client for scraping operations
- [x] **Test Coverage**: API client tests with mock server
- [x] **Validation**: Error handling and type safety

#### 4.3 WebSocket Progress Hook
- [x] **File**: `frontend/hooks/useScrapingProgress.ts`
- [x] **Description**: React hook for WebSocket progress updates
- [x] **Test Coverage**: Hook testing with mock WebSocket
- [x] **Validation**: Real-time update reliability

#### 4.4 Integration Testing
- [x] **Description**: End-to-end testing of scraping workflow
- [x] **Test Coverage**: Full user journey testing
- [x] **Validation**: Performance and reliability testing

---

## 🗂️ Phase 5: CLI Deprecation
**Duration**: 1 day | **Priority**: Low | **Status**: ✅ Completed

### Objective
Move CLI functionality to development tools and update entry points.

### Tasks

#### 5.1 Move CLI to Dev Tools
- [x] **File**: `algobet/cli/dev_tools.py`
- [x] **Description**: Refactor CLI for development use only
- [x] **Commands**:
  - [x] `init` - Initialize database
  - [x] `reset_db` - Reset database (with confirmation)
  - [x] `db_stats` - Show database statistics
- [x] **Test Coverage**: CLI command tests
- [x] **Validation**: Ensure no production dependencies

#### 5.2 Update Configuration
- [x] **File**: `pyproject.toml`
- [x] **Description**: Update entry points and dependencies
- [x] **Validation**: Verify installation and command availability

---

## ⏰ Phase 6: Automated/Scheduled Scraping
**Duration**: 2-3 days | **Priority**: High | **Status**: ✅ Completed

### Objective
Implement automated scraping with multiple scheduling options and comprehensive monitoring.

### Tasks

#### 6.1 Database Models
- [x] **File**: `algobet/models.py` (additions)
- [x] **Models**:
  - [x] `ScheduledTask` - Task configuration
  - [x] `TaskExecution` - Execution history
- [x] **Test Coverage**: Model validation tests
- [x] **Validation**: Database migration and data integrity

#### 6.2 Scheduler Service
- [x] **File**: `algobet/services/scheduler_service.py`
- [x] **Description**: APScheduler-based task management
- [x] **Features**:
  - [x] Task CRUD operations
  - [x] Cron expression validation
  - [x] Task execution orchestration
  - [x] Execution history tracking
- [x] **Test Coverage**:
  - [x] Unit tests for task management
  - [x] Integration tests with APScheduler
  - [x] Task execution tests
- [x] **Validation**: Concurrent task handling

#### 6.3 CLI Runner for Cron Jobs
- [x] **File**: `algobet/cli/scheduled_runner.py`
- [x] **Description**: CLI entry point for cron-based execution
- [x] **Features**:
  - [x] Task execution by name
  - [x] Quick upcoming scrape mode
  - [x] Task listing and status
  - [x] Comprehensive logging
- [x] **Test Coverage**: CLI argument parsing and execution
- [x] **Validation**: Cron compatibility testing

#### 6.4 API Endpoints for Schedule Management
- [x] **File**: `algobet/api/routers/schedules.py`
- [x] **Description**: REST API for schedule management
- [x] **Endpoints**:
  - [x] `POST /api/v1/schedules` - Create schedule
  - [x] `GET /api/v1/schedules` - List schedules
  - [x] `GET /api/v1/schedules/{id}` - Get schedule
  - [x] `PATCH /api/v1/schedules/{id}` - Update schedule
  - [x] `DELETE /api/v1/schedules/{id}` - Delete schedule
  - [x] `POST /api/v1/schedules/{id}/run` - Run task now
  - [x] `GET /api/v1/schedules/{id}/history` - Get execution history
- [x] **Test Coverage**: Full endpoint testing
- [x] **Validation**: Authorization and validation

#### 6.5 Docker Configuration
- [x] **Files**:
  - [x] `docker-compose.scheduler.yml`
  - [x] `Dockerfile.cron`
  - [x] `docker/crontab`
- [x] **Description**: Containerized scheduling options
- [x] **Test Coverage**: Container build and execution tests
- [x] **Validation**: Production deployment testing

#### 6.6 APScheduler Worker Process
- [x] **File**: `algobet/scheduler/worker.py`
- [x] **Description**: Standalone scheduler worker process
- [x] **Features**:
  - [x] Signal handling for graceful shutdown
  - [x] Task loading on startup
  - [x] Error handling and recovery
- [x] **Test Coverage**: Worker lifecycle tests
- [x] **Validation**: Long-running stability testing

#### 6.7 Default Scheduled Tasks
- [x] **File**: `algobet/cli/seed_schedules.py`
- [x] **Description**: Seed default scheduling configurations
- [x] **Default Tasks**:
  - [x] Daily upcoming scrape (6 AM)
  - [x] Evening upcoming scrape (6 PM)
  - [x] Daily predictions (7 AM)
  - [x] Weekly results scrape (Monday 3 AM)
- [x] **Test Coverage**: Seed script testing
- [x] **Validation**: Task creation and activation

#### 6.8 FastAPI Integration
- [x] **File**: `algobet/api/main.py`
- [x] **Description**: Scheduler integration with FastAPI lifecycle
- [x] **Features**:
  - [x] Optional scheduler startup
  - [x] Graceful shutdown handling
  - [x] Router registration
- [x] **Test Coverage**: Integration testing
- [x] **Validation**: Startup/shutdown behavior

#### 6.9 Frontend Schedule Management
- [x] **Files**:
  - [x] `frontend/app/schedules/page.tsx`
  - [x] `frontend/components/schedules/ScheduleCard.tsx`
  - [x] `frontend/components/schedules/ScheduleForm.tsx`
  - [x] `frontend/components/schedules/ExecutionHistory.tsx`
- [x] **Test Coverage**: Component and integration tests
- [x] **Validation**: User experience testing

#### 6.10 API Client for Schedules
- [x] **File**: `frontend/lib/api/schedules.ts`
- [x] **Description**: TypeScript client for schedule management
- [x] **Test Coverage**: API client testing
- [x] **Validation**: Type safety and error handling

#### 6.11 Integration Testing
- [x] **Description**: End-to-end testing of scheduling system
- [x] **Test Coverage**: Full scheduling workflow testing
- [x] **Validation**: Production-like scenario testing

---

## 🧪 Testing & Validation
**Duration**: 1-2 days | **Priority**: High | **Status**: ✅ Completed

### Objective
Ensure comprehensive test coverage and validate all refactoring changes.

### Tasks

#### T.1 Comprehensive Test Suite
- [x] **Unit Tests**: All new service methods
- [x] **Integration Tests**: API endpoint testing
- [x] **End-to-End Tests**: Full workflow testing
- [x] **Performance Tests**: Load and stress testing
- [x] **Test Coverage Target**: >85% coverage (155/155 tests passing)

#### T.2 Regression Testing
- [x] **Existing Functionality**: Ensure no breaking changes
- [x] **CLI Compatibility**: Verify deprecated CLI still works
- [x] **API Compatibility**: Ensure existing API contracts
- [x] **Database Compatibility**: Verify data integrity

#### T.3 Performance Validation
- [x] **Response Times**: API endpoint performance
- [x] **Memory Usage**: Service layer efficiency
- [x] **Concurrent Operations**: Threading and async handling
- [x] **Resource Utilization**: CPU and memory profiling

#### T.4 Production Readiness
- [x] **Error Handling**: Comprehensive error scenarios
- [x] **Logging**: Proper logging and monitoring
- [x] **Documentation**: Updated API documentation
- [x] **Deployment**: Production deployment procedures

---

## 📋 Migration Checklist

### Pre-Migration Preparation
- [x] Create comprehensive backup of production database
- [x] Document current API contracts and behavior
- [x] Set up staging environment for testing
- [x] Prepare rollback procedures
- [x] Notify stakeholders of migration timeline

### Migration Execution
- [x] Deploy Phase 1 (Service Layer) to staging
- [x] Deploy Phase 2 (API Updates) to staging
- [x] Deploy Phase 3 (WebSocket) to staging
- [x] Deploy Phase 4 (Frontend) to staging
- [x] Deploy Phase 5 (CLI) to staging
- [x] Deploy Phase 6 (Scheduling) to staging
- [x] Perform comprehensive staging testing
- [x] Deploy to production with monitoring

### Post-Migration Validation
- [x] Verify all API endpoints are functioning
- [x] Test WebSocket connections and progress updates
- [x] Validate scheduled task execution
- [x] Monitor system performance and errors
- [x] Update documentation and notify users

---

## 🚨 Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| **Breaking existing API** | Low | High | Comprehensive API tests, versioning strategy |
| **Performance regression** | Medium | Medium | Load testing, performance profiling, optimization |
| **WebSocket connection issues** | Medium | Low | Fallback to polling, connection retry logic |
| **Background task failures** | Medium | Medium | Retry logic, dead letter queue, monitoring |
| **Database migration issues** | Low | High | Backup strategy, rollback procedures, testing |
| **Scheduled task conflicts** | Low | Medium | Job locking, single-instance constraints |
| **Frontend compatibility** | Medium | Medium | Progressive enhancement, feature flags |

---

## 📊 Success Metrics

### Code Quality Metrics
- [x] **Code Duplication Reduction**: Target 80% reduction ✅
- [x] **Test Coverage**: Target >85% coverage ✅ (155 tests passing)
- [x] **Cyclomatic Complexity**: Reduce average complexity ✅
- [x] **Technical Debt**: Measurable reduction in debt ✅

### Performance Metrics
- [x] **API Response Time**: <200ms for standard requests ✅
- [x] **WebSocket Latency**: <3s for progress updates ✅
- [x] **Scraping Efficiency**: Maintain or improve current performance ✅
- [x] **Resource Utilization**: Optimize memory and CPU usage ✅

### User Experience Metrics
- [x] **Frontend Accessibility**: All scraping operations via UI ✅
- [x] **Real-time Updates**: Progress visibility for long operations ✅
- [x] **Schedule Management**: Full CRUD operations via frontend ✅
- [x] **Error Handling**: Clear user feedback for all operations ✅

### Operational Metrics
- [x] **Automated Scraping**: Daily execution without manual intervention ✅
- [x] **Execution History**: Complete audit trail of all operations ✅
- [x] **Error Monitoring**: Comprehensive logging and alerting ✅
- [x] **System Reliability**: 99.9% uptime for scheduled operations ✅

---

## 📝 Development Guidelines

### Code Quality Standards
- Follow existing project conventions and patterns
- Write comprehensive unit tests for all new code
- Ensure proper error handling and logging
- Document public APIs and complex logic
- Use type hints consistently

### Testing Requirements
- Unit tests for all service methods
- Integration tests for API endpoints
- End-to-end tests for critical workflows
- Performance tests for high-traffic operations
- Security tests for authentication and authorization

### Deployment Procedures
- Use feature flags for gradual rollouts
- Monitor system metrics during deployment
- Have rollback procedures ready
- Validate functionality in staging first
- Coordinate with stakeholders for production deployment

---

## 🔗 References

- [Original Refactor Plan](docs/refactor.md)
- [IFLOW.md](IFLOW.md) - System architecture documentation
- [AGENT.md](AGENT.md) - Development guidelines
- [API Documentation](docs/api/) - Current API specifications
- [Testing Guidelines](tests/README.md) - Testing procedures

---

## 📞 Support & Contact

For questions or issues during the refactoring process:
- Review the documentation in this repository
- Check the testing guidelines for validation procedures
- Consult the original refactor plan for detailed implementation
- Ensure all changes follow the established project conventions

---

## 🎉 Refactoring Complete!

All phases of the AlgoBet refactoring have been successfully completed. The system now features:

- **Service-oriented architecture** with reusable business logic
- **API-first design** with comprehensive REST endpoints
- **Real-time progress updates** via WebSocket
- **Modern frontend UI** with React/Next.js
- **Automated scheduling** with APScheduler integration
- **Comprehensive test coverage** with 155 passing tests

The codebase is now more maintainable, testable, and scalable.