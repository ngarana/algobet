# AGENTS.md - Algobet Frontend Guide

## Overview

This document provides essential information for agentic coding agents working with the Algobet frontend (located in the `frontend/` directory). It covers build/lint/test commands, code style guidelines, testing patterns, project structure, and actual codebase architecture.

## Table of Contents

- [Build/Lint/Test Commands](#buildlinttest-commands)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Patterns](#testing-patterns)
- [Project Structure](#project-structure)
- [Configuration Details](#configuration-details)
- [Docker Setup](#docker-setup)
- [Best Practices](#best-practices)

## Build/Lint/Test Commands

### Core Commands (via pnpm)

```bash
# Development server
pnpm dev

# Build for production
pnpm build

# Start production server
pnpm start
```

### Quality Gates

```bash
# Run all quality checks
pnpm quality-gates

# Individual quality checks:
pnpm typecheck          # TypeScript type checking
pnpm lint               # ESLint code linting
pnpm lint:fix           # Auto-fix ESLint issues
pnpm format             # Prettier formatting
pnpm format:check       # Check formatting compliance
pnpm test               # Run all tests once
pnpm test:watch         # Run tests in watch mode
pnpm test:ui            # Open Vitest UI dashboard
pnpm test:coverage      # Generate coverage report
```

### Running a Single Test

```bash
# Run tests matching a pattern
pnpm test -- --grep "button"

# Run tests in a specific file
pnpm test -- components/ui/__tests__/button.test.tsx

# Run tests with verbose output
pnpm test -- --reporter=verbose

# Run specific test with coverage
pnpm test -- --coverage components/ui/__tests__/button.test.tsx
```

### Test Patterns

```bash
# Run tests for a specific component
pnpm test -- components/button/__tests__/

# Run tests for hooks
pnpm test -- hooks/__tests__/

# Exclude specific test files
pnpm test -- --exclude "**/__tests__/legacy/**"
```

## Code Style Guidelines

### Import Conventions

1. **Order of Imports**
   - React and React hooks first
   - Third-party libraries second
   - Local imports last
   - Group related imports together

2. **Import Syntax**

   ```typescript
   // ✅ Good - grouped and ordered
   import * as React from "react";
   import { useState, useEffect } from "react";

   import { Button } from "@/components/ui/button";
   import { cn } from "@/lib/utils";
   ```

3. **Path Aliases**
   - Use `@/` for root directory
   - `@/components` for components
   - `@/lib` for utility functions
   - `@/hooks` for custom hooks
   - `@/types` for TypeScript types
   - `@/utils` for utility modules
   - `@/stores` for Zustand stores

### Formatting Standards

1. **Prettier Configuration**
   - Semi-colons: Yes
   - Quotes: Single quotes
   - Tab width: 2 spaces
   - Trailing comma: ES5 (always except last in objects)
   - Print width: 88 characters
   - Plugin: prettier-plugin-tailwindcss

2. **Code Formatting**

   ```bash
   # Format all files
   pnpm format

   # Check formatting
   pnpm format:check
   ```

3. **File Structure**
   - Component files: `ComponentName.tsx`
   - Test files: `ComponentName.test.tsx` or `ComponentName.spec.tsx`
   - Utility files: `utils.ts`
   - Type definitions: `types.ts`
   - Index files: `index.ts` for exports

### TypeScript Type Guidelines

1. **Type Definitions**
   - Use `interface` for object shapes
   - Use `type` for unions, aliases, and complex types
   - Prefer `const` assertions for literal types
   - Use generics for reusable components

2. **Type Safety**
   - Avoid `any` types (ESLint warns)
   - Use proper null checking
   - Define fallback values for optional properties
   - Use discriminated unions for state management

3. **Component Props**
   ```typescript
   interface ComponentProps extends React.ComponentProps<typeof Base> {
     customProp?: string;
     asChild?: boolean;
   }
   ```

### Naming Conventions

1. **Files and Directories**
   - Components: `PascalCase` (Button.tsx)
   - Test files: `ComponentName.test.tsx`
   - Hooks: `useFeatureName.ts`
   - Utility functions: `snake_case` or `camelCase`
   - Constants: `UPPER_SNAKE_CASE`

2. **Variables and Functions**
   - Components: `PascalCase`
   - Hooks: `useFeatureName`
   - Helper functions: `camelCase`
   - Private variables: `_variableName` (to ignore in ESLint)

3. **CSS Classes**
   - Use Tailwind CSS utility classes where possible
   - For custom classes: `kebab-case`
   - Component-specific: `ComponentName__Part`

### Error Handling

1. **Error Boundaries**
   - Implement ErrorBoundary for component error catching
   - Log errors to monitoring services
   - Provide user-friendly fallback UI
   - Include error debugging information in development

2. **API Error Handling**

   ```typescript
   try {
     const data = await apiCall();
   } catch (error) {
     console.error("API Error:", error);
     // Handle specific error types
     if (error instanceof TypeError) {
       // Handle network errors
     }
   }
   ```

3. **Console Usage**
   - Avoid `console.log` in production code
   - Use `console.warn` and `console.error` for diagnostics
   - Mock console methods in tests

### Code Organization

1. **Component Structure**
   - Keep components focused and single-responsibility
   - Extract reusable logic into hooks
   - Use composition over inheritance
   - Colocate tests with components

2. **File Size**
   - Keep components under 300 lines
   - Split large components into smaller ones
   - Extract utilities into separate files

## Testing Patterns

### Test Setup

1. **Test Configuration**
   - Environment: JSDOM
   - Test runner: Vitest
   - Setup file: `src/test/setupTests.ts`
   - Coverage provider: V8

2. **Mocking**

   ```typescript
   import { vi } from "vitest";

   // Mock Next.js navigation
   vi.mock("next/navigation", () => ({
     useRouter() {
       return { push: vi.fn() };
     },
   }));

   // Mock environment variables
   process.env.NEXT_PUBLIC_API_URL = "http://test-api.com";
   ```

### Testing Best Practices

1. **Use Screen Queries**

   ```typescript
   // ✅ Good
   expect(screen.getByRole('button')).toBeInTheDocument();

   // ❌ Avoid
   const { container } = render(<Component />);
   expect(container.querySelector('button')).toBeInTheDocument();
   ```

2. **Prefer User Event**

   ```typescript
   // ✅ Good
   const user = userEvent.setup();
   await user.click(button);

   // ❌ Less realistic
   fireEvent.click(button);
   ```

3. **Test Behavior, Not Implementation**

   ```typescript
   // ✅ Good - tests what user sees
   it("should submit form on click", async () => {
     await user.click(screen.getByRole("button", { name: /submit/i }));
     expect(onSubmit).toHaveBeenCalled();
   });

   // ❌ Bad - tests implementation detail
   it("should call handleSubmit function", () => {
     expect(wrapper.instance().handleSubmit).toHaveBeenCalled();
   });
   ```

4. **Keep Tests Independent**
   - Each test should set up its own state
   - Use `beforeEach` for common setup
   - Avoid test-to-test dependencies

### Testing Utilities

1. **Custom Render**

   ```typescript
   const renderWithProviders = (ui: ReactElement) => {
     const queryClient = new QueryClient({
       defaultOptions: { queries: { retry: false } },
     });

     return render(
       <QueryClientProvider client={queryClient}>
         {ui}
       </QueryClientProvider>
     );
   };
   ```

2. **Async Testing**

   ```typescript
   import { waitFor } from '@testing-library/react';

   it('should fetch data on mount', async () => {
     render(<UserProfile />);

     await waitFor(() => {
       expect(screen.getByText('John Doe')).toBeInTheDocument();
     });
   });
   ```

## Project Structure

```
frontend/
├── app/                          # Next.js app router (feature pages)
│   ├── app/
│   │   ├── backtest/
│   │   ├── calibrate/
│   │   ├── matches/
│   │   ├── models/
│   │   ├── predictions/
│   │   ├── schedules/
│   │   ├── scraping/
│   │   ├── teams/
│   │   └── value-bets/
│   └── providers.tsx             # Global providers
├── components/                    # Reusable components
│   ├── __tests__/                # Component tests
│   ├── backtest/
│   ├── charts/
│   ├── dashboard/
│   ├── error-boundary.tsx
│   ├── layout/
│   ├── matches/
│   ├── schedules/
│   ├── scraping/
│   ├── skeletons/
│   └── ui/
│       ├── badge.tsx
│       ├── button.tsx
│       ├── card.tsx
│       ├── checkbox.tsx
│       ├── input.tsx
│       └── table.tsx
├── hooks/                        # Custom hooks
│   ├── useFetchProgress.ts
│   ├── useJobFocus.ts
│   ├── useJobOperations.ts
│   ├── useLiveLog.ts
│   └── useScrapingProgress.test.tsx
├── lib/                          # Shared utilities
│   ├── api/                      # API clients
│   │   ├── client.ts
│   │   ├── fetch.ts
│   │   ├── index.ts
│   │   ├── matches.ts
│   │   ├── ml-operations.ts
│   │   ├── models.ts
│   │   ├── predictions.ts
│   │   ├── schedules.ts
│   │   ├── scraping.ts
│   │   ├── teams.ts
│   │   ├── tournaments.ts
│   │   └── value-bets.ts
│   ├── constants/
│   │   └── fetch.ts
│   ├── queries/                  # TanStack Query hooks
│   │   ├── index.ts
│   │   ├── use-dashboard-stats.ts
│   │   ├── use-fetch.ts
│   │   ├── use-matches.ts
│   │   ├── use-ml-operations.ts
│   │   ├── use-models.ts
│   │   ├── use-predictions.ts
│   │   ├── use-teams.ts
│   │   └── use-tournaments.ts
│   ├── types/
│   │   ├── api.ts
│   │   ├── ml-operations.ts
│   │   └── schemas.ts
│   └── utils.ts
├── next-env.d.ts
├── next.config.js
├── package.json
├── pnpm-lock.yaml
├── postcss.config.js
├── tailwind.config.js
├── tsconfig.json
├── vitest.config.ts
└── vitest.config.ts
```

## Configuration Details

### ESLint Configuration

**File**: `.eslintrc.json`

**Key Rules**:

- `@typescript-eslint/no-unused-vars`: Warn (allow `_` prefix)
- `@typescript-eslint/no-explicit-any`: Warn
- `prefer-const`: Error
- `no-console`: Warn (allow `warn`, `error`)
- `jest/prefer-to-be`: Warn
- `testing-library/prefer-screen-queries`: Warn

**Test-Specific Rules**:

- Override for test files: `testing-library/render-result-naming-convention: off`
- Override for test files: `testing-library/no-unnecessary-act: error`

### Prettier Configuration

**File**: `.prettierrc`

- Semicolons: `true`
- Single Quote: `false` (double quotes)
- Tab Width: `2`
- Trailing Comma: `es5`
- Print Width: `88`
- Plugins: `prettier-plugin-tailwindcss`

### Vitest Configuration

**File**: `vitest.config.ts`

- Environment: `jsdom`
- Globals: `true`
- Coverage: `v8` provider
- Include: `**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}`
- Exclude: `node_modules`, `.next`, `out`, `playwright-report`, `coverage`, app router pages

**Coverage Thresholds**:

- Lines: 70%
- Statements: 70%
- Functions: 70%
- Branches: 70%

### Lint-Staged Configuration

**File**: `.lintstagedrc.json`

Applies to staged files only:

- `*.{js,jsx,ts,tsx}` → ESLint + Prettier
- `*.{json,css,scss,md}` → Prettier only

### Husky Configuration

- Pre-commit hooks run ESLint and Prettier
- Commit message validation enforced
- Location: `.husky/`

## Docker Setup

The frontend can run inside a Docker container alongside the backend services. This provides a consistent development environment and eliminates port mapping issues.

### Running Frontend in Docker

1. **Start all services including frontend:**

```bash
# From project root
docker-compose up -d

# Or with all services (including scheduler)
docker-compose -f docker-compose.all.yml up -d
```

2. **Access the application:**
   - Frontend: http://localhost:3001
   - API: http://localhost:8010

### Docker Configuration

The frontend Docker setup includes:

- **Dockerfile**: Multi-stage build supporting both development and production
  - `runner-dev` target: For development with hot reloading
  - `runner` target: For production with optimized build

- **API Communication**: Uses Next.js rewrites to proxy requests
  - Client-side requests go to `/api/*` which are rewritten to the API service
  - Server-side requests use the internal Docker network (`http://api:8010`)

- **Port Mapping**: Container port 3001 is mapped to host port 3001

### Environment Variables in Docker

When running in Docker, the following environment variables are set automatically:

| Variable | Value | Purpose |
|----------|-------|---------|
| `NEXT_PUBLIC_API_URL` | `/api/v1` | Client-side API path (uses rewrites) |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8010` | WebSocket URL |
| `API_INTERNAL_URL` | `http://api:8010` | Server-side API URL |

### Troubleshooting Docker Issues

1. **"NetworkError when attempting to fetch resource"**
   - This error occurs when the frontend cannot reach the API
   - Solution: Ensure all services are running with `docker-compose ps`
   - Check API health: `curl http://localhost:8010/health`

2. **Hot reload not working**
   - Volume mounts may need refreshing
   - Restart the frontend container: `docker-compose restart frontend`

3. **Port conflicts**
   - If port 3001 is in use, modify the port mapping in `docker-compose.yml`
   - Change `"3001:3001"` to `"3002:3001"` to use port 3002 on host

### Running Frontend Outside Docker (Development)

To run the frontend directly on the host machine:

1. Ensure `.env.local` points to the API:
```
NEXT_PUBLIC_API_URL=http://localhost:8010/api/v1
API_INTERNAL_URL=http://localhost:8010
```

2. Start the frontend:
```bash
cd frontend
pnpm dev
```

3. Access at http://localhost:3001

## Best Practices

### Before Pushing

1. **Run Quality Gates Locally**

   ```bash
   pnpm quality-gates
   ```

2. **Fix Issues Incrementally**
   - Start with TypeScript errors
   - Then fix linting issues
   - Finally ensure tests pass

3. **Use IDE Integrations**
   - ESLint plugin for real-time feedback
   - Prettier plugin for auto-formatting on save

### Test Writing

1. **Write Tests for New Features**
   - Don't add features without tests
   - Maintain good coverage

2. **Update Tests When Refactoring**
   - Keep tests in sync with implementation
   - Ensure tests still pass after changes

3. **Maintain Coverage**
   - Aim for 70%+ coverage in all categories
   - Run `pnpm test:coverage` regularly

### Code Reviews

1. **Check Quality Gates**
   - Verify all quality checks pass
   - Ensure no new linting errors

2. **Review Test Coverage**
   - New code should have tests
   - Check coverage reports

3. **Validate Code Style**
   - Ensure consistent formatting
   - Follow naming conventions

### CI/CD Integration

- Quality gates run automatically on:
  - Push to `main` or `develop` branches
  - Pull requests targeting `main` or `develop`
- Configuration: `.github/workflows/frontend-ci.yml`

## Troubleshooting

### Pre-commit Hook Fails

```bash
# See what files are failing
git status

# Fix linting issues
pnpm lint:fix

# Fix formatting
pnpm format

# Try commit again
```

### Tests Fail

```bash
# Run tests in watch mode for faster iteration
pnpm test:watch

# Run specific test file
pnpm test -- path/to/test.test.tsx

# Update snapshots if needed
pnpm test -- -u
```

### Type Errors

```bash
# Run type check in watch mode
pnpm typecheck --watch
```

### Common Issues

1. **"ReferenceError: window is not defined"**
   - Ensure tests run in jsdom environment (configured in vitest.config.ts)

2. **Mock Not Working**
   - Make sure mocks are set up before imports
   - Use `vi.mock()` at top of file

3. **Async Tests Timeout**
   - Use `waitFor` or increase timeout: `it('test', async () => { ... }, 10000)`

## Available Scripts Reference

| Script               | Description               |
| -------------------- | ------------------------- |
| `pnpm dev`           | Start development server  |
| `pnpm build`         | Build for production      |
| `pnmb start`         | Start production server   |
| `pnpm quality-gates` | Run all quality checks    |
| `pnpm typecheck`     | TypeScript type checking  |
| `pnpm lint`          | Run ESLint                |
| `pnpm lint:fix`      | Auto-fix ESLint issues    |
| `pnpm format`        | Format code with Prettier |
| `pnpm format:check`  | Check formatting          |
| `pnpm test`          | Run tests once            |
| `pnpm test:watch`    | Run tests in watch mode   |
| `pnmb test:ui`       | Open Vitest UI            |
| `pnpm test:coverage` | Generate coverage report  |

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [User Event](https://testing-library.com/docs/user-event/intro/)
- [ESLint](https://eslint.org/)
- [Prettier](https://prettier.io/)
- [Husky](https://typicode.github.io/husky/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Zustand](https://zustand-demo.pmndrs.rs/)
- [Next.js App Router](https://nextjs.org/docs/app)
