.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests frontend frontend_install frontend_build media_server dev

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	python -m pytest $(TEST_FILE)

integration_tests:
	python -m pytest tests/integration_tests 

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# FRONTEND (React + Vite)
######################

frontend_install:
	cd frontend && npm install

frontend_build:
	cd frontend && npm run build

media_server:
	python -m src.agent.multimedia.static_server

# 一键启动：LangGraph Server + 媒体服务 + 前端 dev（各进程独立运行）
dev:
	@echo '[dev] 启动 LangGraph Server (:2024) ...'
	start "langgraph" cmd /c "langgraph dev"
	@echo '[dev] 启动媒体静态服务 (:8900) ...'
	start "media" cmd /c "python -m src.agent.multimedia.static_server"
	@echo '[dev] 启动前端 dev server (:5173) ...'
	start "frontend" cmd /c "cd frontend && npm run dev"
	@echo '[dev] 已启动三个独立窗口，分别关闭即可停止。'

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'
	@echo '----'
	@echo 'frontend_install             - install frontend deps (npm install in frontend/)'
	@echo 'frontend_build               - build frontend production bundle (npm run build)'
	@echo 'media_server                 - start FastAPI media server on :8900'
	@echo 'dev                          - start LangGraph Server + media server + frontend dev'

