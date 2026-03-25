#!/bin/bash

# Quality Gate Script for Frontend
# This script runs all quality checks in sequence

set -e  # Exit on error

echo "========================================="
echo "Running Frontend Quality Gates"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0

# Step 1: Type Check
echo -e "${YELLOW}[1/4] Running TypeScript type check...${NC}"
if pnpm typecheck; then
  echo -e "${GREEN}✓ TypeScript type check passed${NC}"
else
  echo -e "${RED}✗ TypeScript type check failed${NC}"
  FAILURES=$((FAILURES + 1))
fi
echo ""

# Step 2: Lint
echo -e "${YELLOW}[2/4] Running ESLint...${NC}"
if pnpm lint; then
  echo -e "${GREEN}✓ ESLint passed${NC}"
else
  echo -e "${RED}✗ ESLint failed${NC}"
  FAILURES=$((FAILURES + 1))
fi
echo ""

# Step 3: Format Check
echo -e "${YELLOW}[3/4] Checking code formatting...${NC}"
if pnpm format:check; then
  echo -e "${GREEN}✓ Code formatting is correct${NC}"
else
  echo -e "${RED}✗ Code formatting issues found${NC}"
  echo -e "${YELLOW}Run 'pnpm format' to fix formatting issues${NC}"
  FAILURES=$((FAILURES + 1))
fi
echo ""

# Step 4: Tests
echo -e "${YELLOW}[4/4] Running tests...${NC}"
if pnpm test; then
  echo -e "${GREEN}✓ All tests passed${NC}"
else
  echo -e "${RED}✗ Some tests failed${NC}"
  FAILURES=$((FAILURES + 1))
fi
echo ""

# Summary
echo "========================================="
if [ $FAILURES -eq 0 ]; then
  echo -e "${GREEN}✓ All quality gates passed!${NC}"
  exit 0
else
  echo -e "${RED}✗ $FAILURES quality gate(s) failed${NC}"
  exit 1
fi
