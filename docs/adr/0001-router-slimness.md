# ADR 0001: Thin Routers and Service Orchestrators

## Status

Accepted

## Context

The architecture audit flagged large FastAPI routers that mix HTTP parsing,
business orchestration, persistence, model execution, and response shaping. This
made endpoint behavior hard to test and caused similar logic to be copied across
routers and services.

## Decision

FastAPI routers must stay thin. A router may:

- declare route metadata and dependency injection;
- validate HTTP-facing request parameters;
- call one orchestrator/use-case method;
- return an API schema response.

Routers must not own model training, scraping, backtest execution, pagination
query construction, retry logic, or persistence workflows. Those belong in
service orchestrators and domain collaborators.

## Consequences

New endpoint logic should start in `algobet/services/` or a domain package below
it, then be exposed through a small router adapter. Existing large routers should
be migrated incrementally by extracting one route/use-case at a time while
preserving public API contracts.
