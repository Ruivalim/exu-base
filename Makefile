# exu-base: train encoder-only models that answer with calibrated distributions
#
# `make` or `make help` lists the targets. This Makefile is a facade: every
# target delegates to the real tool (uv, ruff, pytest, node). A new target needs
# `## description` on the same line to show up in help.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --warn-undefined-variables --no-builtin-rules --no-print-directory
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

# ---------------------------------------------------------------------------
# Variables (?= lets you override: `make smoke FIXTURES=...`)
# ---------------------------------------------------------------------------

APP        ?= exu
VENV       ?= .venv
PY         ?= $(VENV)/bin/python
SITE_DIR   ?= _site
ARTIFACTS  ?= artifacts
FIXTURES   ?= tests/fixtures/smoke.jsonl

BOLD  :=
CYAN  :=
GREEN :=
RESET :=
ifneq ($(shell [ -t 1 ] && echo tty),)
  BOLD  := $(shell tput bold 2>/dev/null)
  CYAN  := $(shell tput setaf 6 2>/dev/null)
  GREEN := $(shell tput setaf 2 2>/dev/null)
  RESET := $(shell tput sgr0 2>/dev/null)
endif

##@ Geral

.PHONY: help
help: ## Lista os alvos disponíveis
	@awk 'BEGIN { FS = ":.*##"; printf "\n$(BOLD)$(APP)$(RESET)\n\nUso: make $(CYAN)<alvo>$(RESET)\n" } \
	  /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) } \
	  /^[a-zA-Z0-9_.\/-]+:.*?##/ { printf "  $(CYAN)%-18s$(RESET) %s\n", $$1, $$2 } \
	  END { printf "\n" }' $(MAKEFILE_LIST)

.PHONY: setup
setup: ## Instala dependências e prepara o ambiente local
	@echo "$(GREEN)▸ setup$(RESET)"
	uv sync --extra dev

##@ Qualidade

.PHONY: fmt
fmt: ## Formata o código (modifica arquivos)
	@echo "$(GREEN)▸ fmt$(RESET)"
	uv run ruff format .

.PHONY: lint
lint: ## Lint estático, sem modificar nada
	@echo "$(GREEN)▸ lint$(RESET)"
	uv run ruff check .
	uv run ruff format --check .

.PHONY: test
test: ## Roda os testes
	@echo "$(GREEN)▸ test$(RESET)"
	uv run pytest

.PHONY: check
check: lint test build ## Tudo que o CI roda: lint, test, build
	@echo "$(GREEN)✓ check ok$(RESET)"

##@ Build

.PHONY: build
build: ## Empacota wheel e sdist em dist/
	@echo "$(GREEN)▸ build$(RESET)"
	uv build

.PHONY: clean
clean: ## Remove artefatos gerados
	@echo "$(GREEN)▸ clean$(RESET)"
	rm -rf dist $(SITE_DIR) $(ARTIFACTS) .pytest_cache .ruff_cache

##@ Site

.PHONY: site
site: ## Constrói o site explicativo em _site (precisa de Node 20+)
	@echo "$(GREEN)▸ site$(RESET)"
	node scripts/build-site.mjs

.PHONY: serve
serve: site ## Serve _site em http://localhost:8000
	@echo "$(GREEN)▸ serve em :8000$(RESET)"
	$(PY) -m http.server --directory $(SITE_DIR) 8000

##@ Exemplos

.PHONY: smoke
smoke: ## Treina e avalia um modelo minúsculo nos fixtures (CPU, poucos segundos)
	@echo "$(GREEN)▸ smoke$(RESET)"
	bash scripts/smoke.sh
