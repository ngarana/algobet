# Frontend Dockerfile for Next.js application
# This uses a multi-stage build for optimal production image size

# =============================================================================
# Base stage with common setup
# =============================================================================
FROM node:20-alpine AS base

# Update package index and install system dependencies
RUN apk update && apk add --no-cache libc6-compat git

# Install pnpm
RUN npm install -g pnpm@10.29.3

# Set working directory
WORKDIR /app

# =============================================================================
# Dependencies stage
# =============================================================================
FROM base AS deps

# Copy package files
COPY package.json pnpm-lock.yaml* ./

# Install dependencies
# --ignore-scripts skips husky install which requires .git
RUN pnpm install --ignore-scripts

# =============================================================================
# Development stage
# =============================================================================
FROM base AS runner-dev

# Set environment
ENV NODE_ENV=development
ENV NEXT_TELEMETRY_DISABLED=1

# Expose port
EXPOSE 3001

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/package.json ./package.json

# Copy source code
COPY . .

# Start development server
CMD ["pnpm", "dev"]

# =============================================================================
# Build stage for production
# =============================================================================
FROM deps AS builder

# Copy source code
COPY . .

# Build the application
RUN pnpm build

# =============================================================================
# Production stage
# =============================================================================
FROM base AS runner

# Set environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose port
EXPOSE 3000

# Start production server
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
