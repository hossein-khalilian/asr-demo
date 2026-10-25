# syntax=docker/dockerfile:1.7
#
# Frontend image: Next.js (standalone output).
#
#   dev    – deps installed; source is bind-mounted and `next dev` hot-reloads
#   prod   – only .next/standalone + static assets; no node_modules, no pnpm
#
# Configuration is runtime-only (API_URL, ASR_API_KEY are read by the server-side
# proxy per request), so one prod image works against any backend.

ARG REGISTRY=docker.arvancloud.ir/

FROM ${REGISTRY}node:24-slim AS base
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1 \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0
# pnpm via corepack, at the version package.json's "packageManager" pins.
RUN corepack enable

# ---- dev ------------------------------------------------------------------------
FROM base AS dev
ARG UID=1000
ARG GID=1000
RUN groupmod -g ${GID} node && usermod -u ${UID} -g ${GID} node \
    && mkdir -p /app/.next /app/node_modules && chown -R node:node /app
USER node
COPY --chown=node:node package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
EXPOSE 3000
CMD ["pnpm", "run", "dev"]

# ---- builder --------------------------------------------------------------------
FROM base AS builder
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile
COPY . .
RUN mkdir -p public && pnpm run build

# ---- prod -----------------------------------------------------------------------
FROM ${REGISTRY}node:24-slim AS prod
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
COPY --from=builder --chown=node:node /app/public ./public
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
