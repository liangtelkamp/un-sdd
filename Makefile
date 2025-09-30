.PHONY: help install install-dev test test-cov lint format format-check security clean pre-commit

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r tests/requirements-test.txt
	pip install -e ".[dev,test,security]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=llm_model --cov-report=html --cov-report=xml

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v --no-cov

lint: ## Run all linters
	flake8 llm_model/ tests/
	black --check .
	isort --check-only .
	mypy llm_model/

format: ## Format code with black and isort
	black .
	isort .

format-check: ## Check code formatting
	black --check .
	isort --check-only .

security: ## Run security checks
	safety check --file requirements.txt
	bandit -r llm_model/ -ll

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

ci: format-check lint test-cov security ## Run all CI checks locally

setup: install-dev ## Setup development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make format' to format code"
	@echo "Run 'make lint' to run linters"
