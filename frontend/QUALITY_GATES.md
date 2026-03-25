# Frontend Quality Gates

This document describes the comprehensive quality gates implemented for the Algobet frontend.

## Overview

The project enforces multiple quality checks to ensure code consistency, correctness, and maintainability.

## Quality Gates

### 1. TypeScript Type Checking

Ensures type safety throughout the codebase.

```bash
pnpm typecheck
```

**What it checks:**

- Type correctness
- Interface implementations
- Generic constraints
- Null/undefined handling

### 2. ESLint

Enforces coding standards and best practices.

```bash
pnpm lint          # Check for issues
pnpm lint:fix      # Auto-fix where possible
```

**Rules include:**

- No unused variables (except prefixed with `_`)
- No explicit `any` types (warns)
- Prefer `const` over `let`
- Testing Library best practices
- Jest/ESLint recommended rules

### 3. Code Formatting (Prettier)

Ensures consistent code style.

```bash
pnpm format        # Format all files
pnpm format:check  # Check formatting
```

**Configuration:**

- Semi-colons: Yes
- Quotes: Single
- Tab width: 2 spaces
- Trailing comma: ES5 (always except last in objects)
- Print width: 88 characters
- Tailwind CSS plugin enabled

### 4. Unit Tests

Runs test suite with coverage tracking.

```bash
pnpm test           # Run tests once
pnpm test:watch     # Watch mode
pnpm test:ui        # Open Vitest UI
pnpm test:coverage  # Generate coverage report
```

**Coverage thresholds:**

- Lines: 70%
- Statements: 70%
- Functions: 70%
- Branches: 70%

### 5. Pre-commit Hooks

Automatically runs checks before each commit using Husky + lint-staged.

**Pre-commit:**

- Runs ESLint on staged `.ts`, `.tsx`, `.js`, `.jsx` files
- Runs Prettier on staged files

**Commit message validation:**

- Enforces conventional commit format
- Example: `feat(auth): add user login`

## Running All Quality Gates

Run all quality gates in sequence:

```bash
# Using the script
pnpm quality-gates

# Or directly
./scripts/quality-gates.sh
```

This runs:

1. TypeScript type check
2. ESLint
3. Format check
4. Tests

## CI/CD Integration

Quality gates are automatically run in GitHub Actions on:

- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`

See `.github/workflows/frontend-ci.yml` for configuration.

## File Structure

```
frontend/
├── .eslintrc.json           # ESLint configuration
├── .prettierrc              # Prettier configuration
├── .lintstagedrc.json       # Lint-staged configuration
├── vitest.config.ts         # Vitest test configuration
├── scripts/
│   └── quality-gates.sh     # Quality gate runner script
├── src/test/
│   └── setupTests.ts        # Test setup file
├── .husky/
│   ├── pre-commit           # Pre-commit hook
│   └── commit-msg           # Commit message validation
└── **/__tests__/            # Test files colocated with source
```

## Test Files

Test files follow this naming convention:

- `*.test.ts` - Utility/function tests
- `*.test.tsx` - Component/hook tests

Examples:

- `lib/__tests__/utils.test.ts`
- `components/ui/__tests__/button.test.tsx`
- `hooks/__tests__/useScrapingProgress.test.tsx`

## Configuration Summary

### ESLint Plugins

- `eslint-plugin-jest` - Jest/ESLint best practices
- `eslint-plugin-testing-library` - React Testing Library rules

### Vitest Configuration

- Environment: JSDOM
- Globals: Enabled
- Coverage provider: V8
- Setup file: `src/test/setupTests.ts`

### Lint-staged

Applies to staged files only:

- `*.{js,jsx,ts,tsx}` → ESLint + Prettier
- `*.{json,css,scss,md}` → Prettier only

## Troubleshooting

### Pre-commit hook fails

```bash
# See what files are failing
git status

# Fix linting issues
pnpm lint:fix

# Fix formatting
pnpm format

# Try commit again
```

### Tests fail

```bash
# Run tests in watch mode for faster iteration
pnpm test:watch

# Run specific test file
pnpm test -- path/to/test.test.tsx

# Update snapshots if needed
pnpm test -- -u
```

### Type errors

```bash
# Run type check in watch mode
pnpm typecheck --watch
```

## Best Practices

1. **Run quality gates locally before pushing**

   ```bash
   pnpm quality-gates
   ```

2. **Fix issues incrementally**
   - Start with type errors
   - Then linting
   - Finally tests

3. **Use IDE integrations**
   - ESLint plugin for real-time feedback
   - Prettier plugin for auto-formatting on save

4. **Keep tests up to date**
   - Write tests for new features
   - Update tests when refactoring
   - Maintain good coverage

## Resources

- [Vitest](https://vitest.dev/)
- [Testing Library](https://testing-library.com/)
- [ESLint](https://eslint.org/)
- [Prettier](https://prettier.io/)
- [Husky](https://typicode.github.io/husky/)
