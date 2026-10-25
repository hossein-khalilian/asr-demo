# asr-demo operations
#
#   make dev            start the dev stack (hot reload)      → http://localhost:3000
#   make dev-logs       tail dev logs (SERVICE=backend for one service)
#   make dev-down       stop the dev stack
#
#   make deploy         git pull + rebuild + up on the prod stack
#   make prod           build and start the prod stack
#
#   make test           backend tests + frontend lint/typecheck (no Docker, no NeMo)
#
# Overrides:
#   SERVICE=<name>      scope logs/restart/build to backend or frontend
#   GPU=1               layer docker-compose.gpu.yml (CUDA PyTorch + the host GPU)

GPU ?= 0
GPU_FILE := $(if $(filter 1,$(GPU)),-f docker-compose.gpu.yml)

DEV_COMPOSE  := docker compose -f docker-compose.dev.yml $(GPU_FILE)
PROD_COMPOSE := docker compose -f docker-compose.prod.yml $(GPU_FILE)

SERVICE ?=

.PHONY: help env \
        dev dev-up dev-down dev-restart dev-build dev-logs dev-ps \
        prod prod-up prod-down prod-restart prod-build prod-logs prod-ps \
        deploy pull test test-backend test-frontend

help:
	@awk 'BEGIN{FS=":.*##"; printf "targets:\n"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

env: ## create .env and backend/.env from the examples if missing, and the proxy network
	@test -f .env || cp .env.example .env
	@test -f backend/.env || cp backend/.env.example backend/.env
	@docker network inspect proxy >/dev/null 2>&1 || docker network create proxy >/dev/null

# ── Dev stack ───────────────────────────────────────────────────────────────

dev: dev-up ## start dev stack in the background

dev-up: env ## start dev stack in the background
	$(DEV_COMPOSE) up -d --build $(SERVICE)

dev-down: ## stop dev stack (keeps the model cache)
	$(DEV_COMPOSE) down

dev-restart: ## restart dev services (SERVICE=name to scope)
	$(DEV_COMPOSE) restart $(SERVICE)

dev-build: ## rebuild dev images (SERVICE=name to scope)
	$(DEV_COMPOSE) build $(SERVICE)

dev-logs: ## tail dev logs (SERVICE=name to scope)
	$(DEV_COMPOSE) logs -f --tail=200 $(SERVICE)

dev-ps: ## list dev containers
	$(DEV_COMPOSE) ps

# ── Prod stack ──────────────────────────────────────────────────────────────

prod: prod-up ## build and start prod stack

prod-up: env ## build and start prod stack in the background
	$(PROD_COMPOSE) up -d --build $(SERVICE)

prod-down: ## stop prod stack (keeps the model cache)
	$(PROD_COMPOSE) down

prod-restart: ## restart prod services (SERVICE=name to scope)
	$(PROD_COMPOSE) restart $(SERVICE)

prod-build: ## rebuild prod images (SERVICE=name to scope)
	$(PROD_COMPOSE) build $(SERVICE)

prod-logs: ## tail prod logs (SERVICE=name to scope)
	$(PROD_COMPOSE) logs -f --tail=200 $(SERVICE)

prod-ps: ## list prod containers
	$(PROD_COMPOSE) ps

# ── Deploy ──────────────────────────────────────────────────────────────────

pull: ## git pull the current branch
	git pull --ff-only

deploy: pull prod-build prod-up ## pull latest, rebuild prod images, restart stack
	$(PROD_COMPOSE) ps

# ── Checks ──────────────────────────────────────────────────────────────────

test: test-backend test-frontend ## run all checks locally

test-backend: ## ruff + pytest (needs ffmpeg; creates backend/.venv)
	cd backend && test -d .venv || python3 -m venv .venv
	cd backend && .venv/bin/pip install -q -r requirements-dev.txt
	cd backend && .venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests
	cd backend && .venv/bin/pytest -q

test-frontend: ## eslint + tsc + prettier
	cd frontend && pnpm install --frozen-lockfile
	cd frontend && pnpm lint && pnpm typecheck && pnpm format:check
