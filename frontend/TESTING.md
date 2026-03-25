# Testing Guide

This document provides comprehensive information about testing in the Algobet frontend project.

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Structure](#test-structure)
- [Best Practices](#best-practices)
- [Quality Gates](#quality-gates)

## Overview

The Algobet frontend uses a comprehensive testing strategy with multiple quality gates to ensure code quality and reliability.

## Technologies

- **Vitest**: Fast, Vite-native test runner
- **React Testing Library**: For component testing
- **User Event**: For realistic user interaction testing
- **JSDOM**: For DOM simulation in Node.js
- **ESLint plugins**: jest and testing-library plugins for linting tests

## Running Tests

### Basic Commands

```bash
# Run all tests once
pnpm test

# Run tests in watch mode (re-runs on file changes)
pnpm test:watch

# Open Vitest UI dashboard
pnpm test:ui

# Run tests with coverage report
pnpm test:coverage
```

### Test Patterns

```bash
# Run tests matching a pattern
pnpm test -- --grep "button"

# Run tests in a specific file
pnpm test -- components/ui/__tests__/button.test.tsx

# Run tests with verbose output
pnpm test -- --reporter=verbose
```

## Writing Tests

### Component Tests

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from '@/components/ui/button';

describe('Button', () => {
  it('should render correctly', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByRole('button')).toBeInTheDocument();
  });

  it('should handle clicks', async () => {
    const onClick = vi.fn();
    const user = userEvent.setup();

    render(<Button onClick={onClick}>Click me</Button>);
    await user.click(screen.getByRole('button'));

    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
```

### Hook Tests

```typescript
import { renderHook, act } from "@testing-library/react";
import { useScrapingProgress } from "@/hooks/useScrapingProgress";

describe("useScrapingProgress", () => {
  it("should initialize with default values", () => {
    const { result } = renderHook(() => useScrapingProgress({ jobId: "test-123" }));

    expect(result.current.isConnected).toBe(false);
  });

  it("should update on progress changes", async () => {
    const { result } = renderHook(() =>
      useScrapingProgress({ jobId: "test-123", enabled: true })
    );

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });
  });
});
```

### Utility Function Tests

```typescript
import { describe, it, expect } from "vitest";
import { cn } from "@/lib/utils";

describe("cn utility", () => {
  it("should merge tailwind classes", () => {
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("should handle conditional classes", () => {
    const isActive = true;
    expect(cn("btn", isActive && "btn-active")).toBe("btn btn-active");
  });
});
```

### API Client Tests

```typescript
import { vi } from "vitest";
import { apiGet, apiPost } from "@/lib/api/client";

// Mock fetch
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

describe("apiGet", () => {
  it("should fetch data", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 1 }),
    });

    const result = await apiGet("/users/1");
    expect(result).toEqual({ id: 1 });
  });
});
```

## Test Structure

### File Organization

Tests should be colocated with the code they test:

```
frontend/
├── components/
│   ├── ui/
│   │   ├── button.tsx
│   │   └── __tests__/
│   │       └── button.test.tsx
├── hooks/
│   ├── useScrapingProgress.ts
│   └── __tests__/
│       └── useScrapingProgress.test.tsx
├── lib/
│   ├── utils.ts
│   └── __tests__/
│       └── utils.test.ts
└── src/
    └── test/
        └── setupTests.ts
```

### Naming Conventions

- Test files: `*.test.ts` or `*.test.tsx`
- Test suites: `describe('<ComponentName>', ...)`
- Test cases: `it('should do something', ...)`

## Best Practices

### 1. Use Screen Queries

```typescript
// ✅ Good
expect(screen.getByRole('button')).toBeInTheDocument();

// ❌ Avoid
const { container } = render(<Button />);
expect(container.querySelector('button')).toBeInTheDocument();
```

### 2. Prefer User Event over FireEvent

```typescript
// ✅ Good
const user = userEvent.setup();
await user.click(button);

// ❌ Less realistic
fireEvent.click(button);
```

### 3. Test Behavior, Not Implementation

```typescript
// ✅ Good - tests behavior
it("should submit form on click", async () => {
  await user.click(screen.getByRole("button", { name: /submit/i }));
  expect(onSubmit).toHaveBeenCalled();
});

// ❌ Bad - tests implementation
it("should call handleSubmit function", () => {
  expect(wrapper.instance().handleSubmit).toHaveBeenCalled();
});
```

### 4. Keep Tests Independent

```typescript
// ✅ Good - each test is independent
it('should render', () => { ... });
it('should handle click', async () => { ... });

// ❌ Bad - tests depend on each other
it('should render and then handle click', () => { ... });
```

### 5. Use Descriptive Test Names

```typescript
// ✅ Good
it("should disable button when loading is true");

// ❌ Bad
it("should work");
```

### 6. Mock External Dependencies

```typescript
// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter() {
    return { push: vi.fn() };
  },
}));

// Mock environment variables
process.env.NEXT_PUBLIC_API_URL = "http://test-api.com";
```

## Quality Gates

### Pre-commit Hooks

The project uses Husky with lint-staged to run quality checks before each commit:

- **ESLint**: Code linting
- **Prettier**: Code formatting

### CI Pipeline

Run all quality gates locally before pushing:

```bash
# Run the complete quality gate script
./scripts/quality-gates.sh

# Or manually:
pnpm typecheck    # TypeScript type checking
pnpm lint         # ESLint
pnpm format:check # Prettier format check
pnpm test         # Run tests
```

### Coverage Requirements

The project enforces minimum coverage thresholds:

- **Lines**: 70%
- **Statements**: 70%
- **Functions**: 70%
- **Branches**: 70%

View coverage report:

```bash
pnpm test:coverage
# Open coverage/index.html in browser
```

## Common Patterns

### Testing Async Operations

```typescript
import { waitFor } from '@testing-library/react';

it('should fetch data on mount', async () => {
  render(<UserProfile />);

  await waitFor(() => {
    expect(screen.getByText('John Doe')).toBeInTheDocument();
  });
});
```

### Testing Context Providers

```typescript
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

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

### Testing Custom Hooks

```typescript
import { renderHook, act } from "@testing-library/react";

it("should update state", () => {
  const { result } = renderHook(() => useCounter());

  act(() => {
    result.current.increment();
  });

  expect(result.current.count).toBe(1);
});
```

## Troubleshooting

### Common Issues

**Issue**: Tests fail with "ReferenceError: window is not defined"

**Solution**: Ensure tests run in jsdom environment (configured in vitest.config.ts)

**Issue**: Mock not working

**Solution**: Make sure mocks are set up before imports, use `vi.mock()` at top of file

**Issue**: Async tests timeout

**Solution**: Use `waitFor` or increase timeout: `it('test', async () => { ... }, 10000)`

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
- [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)
- [User Event](https://testing-library.com/docs/user-event/intro/)
