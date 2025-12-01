.PHONY: help install install-dev install-hooks clean lint format type-check test test-quick pre-commit-all pre-commit-update

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

install-hooks:  ## Install pre-commit git hooks
	pre-commit install
	@echo "✅ Pre-commit hooks installed successfully!"

clean:  ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage .tox/
	@echo "✅ Cleaned build artifacts and caches"

lint:  ## Run linting checks (flake8)
	@echo "Running Flake8..."
	flake8 model_checkpoint/ tests/ examples/
	@echo "✅ Linting passed!"

format:  ## Format code with black and isort
	@echo "Running Black..."
	black model_checkpoint/ tests/ examples/ setup.py run_tests.py
	@echo "Running isort..."
	isort model_checkpoint/ tests/ examples/ setup.py run_tests.py
	@echo "✅ Code formatted!"

type-check:  ## Run type checking with mypy
	@echo "Running MyPy..."
	mypy model_checkpoint/
	@echo "✅ Type checking passed!"

test:  ## Run all tests
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "✅ All tests passed!"

test-quick:  ## Run tests without coverage
	@echo "Running quick tests..."
	python run_tests.py
	@echo "✅ Quick tests completed!"

test-coverage:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest --cov=model_checkpoint --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

pre-commit-all:  ## Run all pre-commit hooks on all files
	@echo "Running all pre-commit hooks..."
	pre-commit run --all-files
	@echo "✅ Pre-commit checks completed!"

pre-commit-update:  ## Update pre-commit hook versions
	@echo "Updating pre-commit hooks..."
	pre-commit autoupdate
	@echo "✅ Pre-commit hooks updated!"

check:  ## Run all checks (format, lint, type-check, test)
	@echo "🔍 Running all checks..."
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) test
	@echo "✅ All checks passed!"

dev-setup:  ## Complete development environment setup
	@echo "🚀 Setting up development environment..."
	@$(MAKE) install-dev
	@$(MAKE) install-hooks
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make test' to verify everything works"
	@echo "  2. Run 'make pre-commit-all' to check code quality"
	@echo "  3. See 'make help' for more commands"

security:  ## Run security checks with bandit
	@echo "Running Bandit security checks..."
	bandit -r model_checkpoint/ -c pyproject.toml
	@echo "✅ Security checks passed!"

deps-check:  ## Check dependencies for security vulnerabilities
	@echo "Checking dependencies with pip-audit..."
	pip-audit
	@echo "✅ Dependencies are secure!"

all-checks:  ## Run all checks including security
	@echo "🔍 Running comprehensive checks..."
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security
	@$(MAKE) test
	@echo "✅ All comprehensive checks passed!"
