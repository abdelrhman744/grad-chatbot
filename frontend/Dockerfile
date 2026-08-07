# syntax=docker/dockerfile:1

# ── Dependencies ─────────────────────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ── Build ────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# `next build` calls next.config.js's rewrites() ONCE and freezes the
# resolved destination into .next/routes-manifest.json — the standalone
# server replays that manifest at runtime, it does NOT re-invoke
# rewrites() per request. So BACKEND_INTERNAL_URL must be correct at BUILD
# time (a Docker build ARG), not just set via docker-compose's runtime
# `environment:` block (which is too late — the manifest is already baked).
ARG BACKEND_INTERNAL_URL=http://backend:8000
ENV BACKEND_INTERNAL_URL=$BACKEND_INTERNAL_URL
RUN npm run build

# ── Runtime ──────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Next.js standalone output already bundles only the production
# node_modules it needs — no npm install in this stage.
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

# 127.0.0.1, not localhost: Alpine's resolver returns ::1 (IPv6) first for
# "localhost", but Next.js's HOSTNAME=0.0.0.0 bind is IPv4-only — wget has
# no fallback, so it would try ::1 and get an immediate connection refused
# even though the server is up and reachable on IPv4.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD wget -q -O- http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
