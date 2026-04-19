# AlgoBet Docker Makefile

.PHONY: help build up down logs clean rebuild restart backend scheduler scheduler-init db init-db

# Default target
help:
	@echo "AlgoBet Docker Commands:"
	@echo ""
	@echo "  make build        - Build all Docker containers"
	@echo "  make up           - Start all containers (api + scheduler-worker)"
	@echo "  make down         - Stop all containers"
	@echo "  make logs         - View logs from all containers"
	@echo "  make clean        - Remove containers and volumes"
	@echo "  make rebuild      - Rebuild and restart all containers"
	@echo "  make restart      - Restart all containers"
	@echo ""
	@echo "  make backend      - Start backend (db + api) only"
	@echo "  make scheduler    - Start full stack with scheduler worker"
	@echo "  make scheduler-init - Initialize scheduler with default tasks"
	@echo "  make db           - Start database only"
	@echo "  make init-db      - Initialize database tables"
	@echo ""
	@echo "  make logs-backend - View backend logs"
	@echo "  make logs-scheduler - View scheduler worker logs"

# Build all containers
build:
	docker-compose build

# Start all containers (api + scheduler-worker)
up:
	docker-compose up -d

# Start backend only (db + api)
backend:
	docker-compose up -d

# Start scheduler stack (db + api + scheduler-worker)
scheduler:
	docker-compose up -d

# Start database only
db:
	docker-compose up -d db

# Stop all containers
down:
	docker-compose down

# View logs from all containers
logs:
	docker-compose logs -f

# View backend logs
logs-backend:
	docker-compose logs -f api db

# View scheduler worker logs
logs-scheduler:
	docker-compose logs -f scheduler-worker

# Remove containers and volumes
clean:
	docker-compose down -v

# Initialize database tables
init-db:
	docker exec algobet-api python -c "from algobet.database import init_db; init_db()"

# Initialize scheduler with default tasks
scheduler-init:
	docker exec algobet-scheduler-worker python -c "from algobet.services.scheduler_tasks import register_default_tasks; from algobet.services.scheduler_service import SchedulerService; from algobet.database import session_scope; register_default_tasks(); print('Task types registered')"
	@echo "Scrape upcoming task will be loaded on next scheduler-worker restart"

# Rebuild and restart all containers
rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# Restart all containers
restart:
	docker-compose restart
