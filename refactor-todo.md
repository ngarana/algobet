# AlgoBet Refactoring Roadmap

**Status**: 📋 Planning Phase
**Created**: 2026-02-05
**Last Updated**: 2026-02-05

## 🎯 Executive Summary

This document outlines a systematic refactoring plan to transform AlgoBet from a CLI-heavy architecture to a modern service-oriented architecture with API-first design, real-time progress updates, and automated scheduling capabilities.

## 📊 Progress Overview

### Overall Progress: 55% Complete
- **Phase 1**: Service Layer Foundation (✅ 3/3 tasks completed)
- **Phase 2**: API Scraping Endpoints (✅ 4/4 tasks completed)
- **Phase 3**: WebSocket Progress Updates (✅ 2/2 tasks completed)
- **Phase 4**: Frontend Integration (0/4 tasks)
- **Phase 5**: CLI Deprecation (0/2 tasks)
- **Phase 6**: Automated/Scheduled Scraping (0/11 tasks)
- **Testing & Validation**: (0/4 tasks)

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
**Duration**: 2-3 days | **Priority**: High | **Status**: 🔴 Not Started

### Objective
Create frontend interface for scraping management with real-time updates.

### Tasks

#### 4.1 Scraping Dashboard Component
- [ ] **Files**:
  - [ ] `frontend/app/scraping/page.tsx`
  - [ ] `frontend/components/scraping/ScrapingJobCard.tsx`
  - [ ] `frontend/components/scraping/ScrapingProgress.tsx`
  - [ ] `frontend/components/scraping/ScrapeForm.tsx`
  - [ ] `frontend/components/scraping/JobHistoryTable.tsx`
- [ ] **Test Coverage**: Component unit tests
- [ ] **Validation**: UI/UX testing with users

#### 4.2 API Client Functions
- [ ] **File**: `frontend/lib/api/scraping.ts`
- [ ] **Description**: TypeScript API client for scraping operations
- [ ] **Test Coverage**: API client tests with mock server
- [ ] **Validation**: Error handling and type safety

#### 4.3 WebSocket Progress Hook
- [ ] **File**: `frontend/hooks/useScrapingProgress.ts`
- [ ] **Description**: React hook for WebSocket progress updates
- [ ] **Test Coverage**: Hook testing with mock WebSocket
- [ ] **Validation**: Real-time update reliability

#### 4.4 Integration Testing
- [ ] **Description**: End-to-end testing of scraping workflow
- [ ] **Test Coverage**: Full user journey testing
- [ ] **Validation**: Performance and reliability testing

---

## 🗂️ Phase 5: CLI Deprecation
**Duration**: 1 day | **Priority**: Low | **Status**: 🔴 Not Started

### Objective
Move CLI functionality to development tools and update entry points.

### Tasks

#### 5.1 Move CLI to Dev Tools
- [ ] **File**: `algobet/cli/dev_tools.py`
- [ ] **Description**: Refactor CLI for development use only
- [ ] **Commands**:
  - [ ] `init` - Initialize database
  - [ ] `reset_db` - Reset database (with confirmation)
  - [ ] `db_stats` - Show database statistics
- [ ] **Test Coverage**: CLI command tests
- [ ] **Validation**: Ensure no production dependencies

#### 5.2 Update Configuration
- [ ] **File**: `pyproject.toml`
- [ ] **Description**: Update entry points and dependencies
- [ ] **Validation**: Verify installation and command availability

---

## ⏰ Phase 6: Automated/Scheduled Scraping
**Duration**: 2-3 days | **Priority**: High | **Status**: 🔴 Not Started

### Objective
Implement automated scraping with multiple scheduling options and comprehensive monitoring.

### Tasks

#### 6.1 Database Models
- [ ] **File**: `algobet/models.py` (additions)
- [ ] **Models**:
  - [ ] `ScheduledTask` - Task configuration
  - [ ] `TaskExecution` - Execution history
- [ ] **Test Coverage**: Model validation tests
- [ ] **Validation**: Database migration and data integrity

#### 6.2 Scheduler Service
- [ ] **File**: `algobet/services/scheduler_service.py`
- [ ] **Description**: APScheduler-based task management
- [ ] **Features**:
  - [ ] Task CRUD operations
  - [ ] Cron expression validation
  - [ ] Task execution orchestration
  - [ ] Execution history tracking
- [ ] **Test Coverage**:
  - [ ] Unit tests for task management
  - [ ] Integration tests with APScheduler
  - [ ] Task execution tests
- [ ] **Validation**: Concurrent task handling

#### 6.3 CLI Runner for Cron Jobs
- [ ] **File**: `algobet/cli/scheduled_runner.py`
- [ ] **Description**: CLI entry point for cron-based execution
- [ ] **Features**:
  - [ ] Task execution by name
  - [ ] Quick upcoming scrape mode
  - [ ] Task listing and status
  - [ ] Comprehensive logging
- [ ] **Test Coverage**: CLI argument parsing and execution
- [ ] **Validation**: Cron compatibility testing

#### 6.4 API Endpoints for Schedule Management
- [ ] **File**: `algobet/api/routers/schedules.py`
- [ ] **Description**: REST API for schedule management
- [ ] **Endpoints**:
  - [ ] `POST /api/v1/schedules` - Create schedule
  - [ ] `GET /api/v1/schedules` - List schedules
  - [ ] `GET /api/v1/schedules/{id}` - Get schedule
  - [ ] `PATCH /api/v1/schedules/{id}` - Update schedule
  - [ ] `DELETE /api/v1/schedules/{id}` - Delete schedule
  - [ ] `POST /api/v1/schedules/{id}/run` - Run task now
  - [ ] `GET /api/v1/schedules/{id}/history` - Get execution history
- [ ] **Test Coverage**: Full endpoint testing
- [ ] **Validation**: Authorization and validation

#### 6.5 Docker Configuration
- [ ] **Files**:
  - [ ] `docker-compose.scheduler.yml`
  - [ ] `Dockerfile.cron`
  - [ ] `docker/crontab`
- [ ] **Description**: Containerized scheduling options
- [ ] **Test Coverage**: Container build and execution tests
- [ ] **Validation**: Production deployment testing

#### 6.6 APScheduler Worker Process
- [ ] **File**: `algobet/scheduler/worker.py`
- [ ] **Description**: Standalone scheduler worker process
- [ ] **Features**:
  - [ ] Signal handling for graceful shutdown
  - [ ] Task loading on startup
  - [ ] Error handling and recovery
- [ ] **Test Coverage**: Worker lifecycle tests
- [ ] **Validation**: Long-running stability testing

#### 6.7 Default Scheduled Tasks
- [ ] **File**: `algobet/cli/seed_schedules.py`
- [ ] **Description**: Seed default scheduling configurations
- [ ] **Default Tasks**:
  - [ ] Daily upcoming scrape (6 AM)
  - [ ] Evening upcoming scrape (6 PM)
  - [ ] Daily predictions (7 AM)
  - [ ] Weekly results scrape (Monday 3 AM)
- [ ] **Test Coverage**: Seed script testing
- [ ] **Validation**: Task creation and activation

#### 6.8 FastAPI Integration
- [ ] **File**: `algobet/api/main.py`
- [ ] **Description**: Scheduler integration with FastAPI lifecycle
- [ ] **Features**:
  - [ ] Optional scheduler startup
  - [ ] Graceful shutdown handling
  - [ ] Router registration
- [ ] **Test Coverage**: Integration testing
- [ ] **Validation**: Startup/shutdown behavior

#### 6.9 Frontend Schedule Management
- [ ] **Files**:
  - [ ] `frontend/app/schedules/page.tsx`
  - [ ] `frontend/components/schedules/ScheduleCard.tsx`
  - [ ] `frontend/components/schedules/ScheduleForm.tsx`
  - [ ] `frontend/components/schedules/CronExpressionInput.tsx`
  - [ ] `frontend/components/schedules/ExecutionHistory.tsx`
- [ ] **Test Coverage**: Component and integration tests
- [ ] **Validation**: User experience testing

#### 6.10 API Client for Schedules
- [ ] **File**: `frontend/lib/api/schedules.ts`
- [ ] **Description**: TypeScript client for schedule management
- [ ] **Test Coverage**: API client testing
- [ ] **Validation**: Type safety and error handling

#### 6.11 Integration Testing
- [ ] **Description**: End-to-end testing of scheduling system
- [ ] **Test Coverage**: Full scheduling workflow testing
- [ ] **Validation**: Production-like scenario testing

---

## 🧪 Testing & Validation
**Duration**: 1-2 days | **Priority**: High | **Status**: 🔴 Not Started

### Objective
Ensure comprehensive test coverage and validate all refactoring changes.

### Tasks

#### T.1 Comprehensive Test Suite
- [ ] **Unit Tests**: All new service methods
- [ ] **Integration Tests**: API endpoint testing
- [ ] **End-to-End Tests**: Full workflow testing
- [ ] **Performance Tests**: Load and stress testing
- [ ] **Test Coverage Target**: >85% coverage

#### T.2 Regression Testing
- [ ] **Existing Functionality**: Ensure no breaking changes
- [ ] **CLI Compatibility**: Verify deprecated CLI still works
- [ ] **API Compatibility**: Ensure existing API contracts
- [ ] **Database Compatibility**: Verify data integrity

#### T.3 Performance Validation
- [ ] **Response Times**: API endpoint performance
- [ ] **Memory Usage**: Service layer efficiency
- [ ] **Concurrent Operations**: Threading and async handling
- [ ] **Resource Utilization**: CPU and memory profiling

#### T.4 Production Readiness
- [ ] **Error Handling**: Comprehensive error scenarios
- [ ] **Logging**: Proper logging and monitoring
- [ ] **Documentation**: Updated API documentation
- [ ] **Deployment**: Production deployment procedures

---

## 📋 Migration Checklist

### Pre-Migration Preparation
- [ ] Create comprehensive backup of production database
- [ ] Document current API contracts and behavior
- [ ] Set up staging environment for testing
- [ ] Prepare rollback procedures
- [ ] Notify stakeholders of migration timeline

### Migration Execution
- [ ] Deploy Phase 1 (Service Layer) to staging
- [ ] Deploy Phase 2 (API Updates) to staging
- [ ] Deploy Phase 3 (WebSocket) to staging
- [ ] Deploy Phase 4 (Frontend) to staging
- [ ] Deploy Phase 5 (CLI) to staging
- [ ] Deploy Phase 6 (Scheduling) to staging
- [ ] Perform comprehensive staging testing
- [ ] Deploy to production with monitoring

### Post-Migration Validation
- [ ] Verify all API endpoints are functioning
- [ ] Test WebSocket connections and progress updates
- [ ] Validate scheduled task execution
- [ ] Monitor system performance and errors
- [ ] Update documentation and notify users

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
- [ ] **Code Duplication Reduction**: Target 80% reduction
- [ ] **Test Coverage**: Target >85% coverage
- [ ] **Cyclomatic Complexity**: Reduce average complexity
- [ ] **Technical Debt**: Measurable reduction in debt

### Performance Metrics
- [ ] **API Response Time**: <200ms for standard requests
- [ ] **WebSocket Latency**: <3s for progress updates
- [ ] **Scraping Efficiency**: Maintain or improve current performance
- [ ] **Resource Utilization**: Optimize memory and CPU usage

### User Experience Metrics
- [ ] **Frontend Accessibility**: All scraping operations via UI
- [ ] **Real-time Updates**: Progress visibility for long operations
- [ ] **Schedule Management**: Full CRUD operations via frontend
- [ ] **Error Handling**: Clear user feedback for all operations

### Operational Metrics
- [ ] **Automated Scraping**: Daily execution without manual intervention
- [ ] **Execution History**: Complete audit trail of all operations
- [ ] **Error Monitoring**: Comprehensive logging and alerting
- [ ] **System Reliability**: 99.9% uptime for scheduled operations

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

**Remember**: This is a comprehensive refactoring that will significantly improve the system's maintainability, testability, and user experience. Take time to implement each phase thoroughly and test comprehensively.
