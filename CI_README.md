# CI/CD Pipeline Documentation

This document describes the continuous integration and deployment setup for the LLM Model Strategy Pattern project.

## Overview

The project uses GitHub Actions for CI/CD with the following workflows:

- **CI Pipeline** (`ci.yml`) - Runs tests, linting, and code quality checks
- **Auto Format** (`format.yml`) - Automatically formats code on pull requests
- **Dependabot** (`dependabot.yml`) - Automatically updates dependencies

## CI Pipeline Features

### Test Matrix
- Tests run on Python 3.9, 3.10, and 3.11
- Runs on Ubuntu latest
- Caches pip dependencies for faster builds

### Code Quality Checks
- **Black** - Code formatting
- **isort** - Import sorting
- **flake8** - Linting and style checking
- **mypy** - Type checking
- **pytest** - Testing with coverage
- **bandit** - Security linting
- **safety** - Dependency vulnerability scanning

### Coverage Reporting
- Generates HTML and XML coverage reports
- Uploads coverage to Codecov
- Coverage threshold can be configured in `pyproject.toml`

## Local Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Or manually
pip install -r requirements.txt
pip install -r tests/requirements-test.txt
pip install -e ".[dev,test,security]"
pre-commit install
```

### Running Checks Locally

```bash
# Run all CI checks
make ci

# Run specific checks
make test          # Run tests
make lint          # Run linters
make format        # Format code
make security      # Run security checks
make format-check  # Check formatting without changing files
```

### Pre-commit Hooks

Pre-commit hooks are configured to run automatically on git commits:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black
pre-commit run flake8
```

## Configuration Files

### `pyproject.toml`
- Project metadata and dependencies
- Tool configurations (Black, isort, mypy, pytest, coverage)
- Development dependencies

### `.pre-commit-config.yaml`
- Pre-commit hook configurations
- Automatic code formatting and linting

### `.github/workflows/`
- GitHub Actions workflow definitions
- CI/CD pipeline configuration

### `Makefile`
- Convenient commands for local development
- Standardized build and test processes

## Workflow Details

### CI Pipeline (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Jobs:**
1. **test** - Runs on multiple Python versions
2. **format-check** - Validates code formatting
3. **security** - Runs security scans

**Steps:**
1. Checkout code
2. Setup Python environment
3. Cache dependencies
4. Install dependencies
5. Run linting (flake8, Black, isort, mypy)
6. Run tests with coverage
7. Upload coverage to Codecov
8. Run security checks (safety, bandit)

### Auto Format (`format.yml`)

**Triggers:**
- Pull requests with Python file changes

**Features:**
- Automatically formats code with Black and isort
- Commits formatted code back to the PR
- Comments on PR when formatting is applied

### Dependabot (`dependabot.yml`)

**Features:**
- Automatically updates Python dependencies weekly
- Updates GitHub Actions weekly
- Auto-merges patch version updates
- Creates PRs for major/minor updates

## Quality Gates

The CI pipeline enforces the following quality gates:

1. **Code Formatting** - All code must be formatted with Black
2. **Import Sorting** - Imports must be sorted with isort
3. **Linting** - Code must pass flake8 checks
4. **Type Checking** - Code must pass mypy type checking
5. **Tests** - All tests must pass
6. **Security** - No high-severity security issues
7. **Coverage** - Maintain test coverage (configurable threshold)

## Badges

Add these badges to your README:

```markdown
![CI](https://github.com/liangtelkamp/un-sdd/workflows/CI%20Pipeline/badge.svg)
![Codecov](https://codecov.io/gh/liangtelkamp/un-sdd/branch/main/graph/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
```

## Troubleshooting

### Common Issues

1. **Black formatting errors**
   ```bash
   make format
   git add .
   git commit -m "Format code with black"
   ```

2. **Import sorting errors**
   ```bash
   isort .
   git add .
   git commit -m "Sort imports with isort"
   ```

3. **Linting errors**
   ```bash
   make lint
   # Fix issues manually
   ```

4. **Test failures**
   ```bash
   make test
   # Check test output and fix issues
   ```

5. **Type checking errors**
   ```bash
   mypy llm_model/
   # Add type annotations as needed
   ```

### Debugging CI Issues

1. Check the Actions tab in GitHub
2. Review the workflow logs
3. Run the same commands locally
4. Check for environment differences

## Contributing

When contributing to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make ci` to ensure all checks pass
5. Commit your changes
6. Push to your fork
7. Create a pull request

The CI pipeline will automatically run all checks on your PR.

## Security

The CI pipeline includes several security measures:

- **Dependency scanning** with safety
- **Code security analysis** with bandit
- **Automated dependency updates** with Dependabot
- **Pre-commit hooks** to catch issues early

## Performance

The CI pipeline is optimized for performance:

- **Dependency caching** to speed up builds
- **Parallel job execution** where possible
- **Conditional execution** based on file changes
- **Efficient test discovery** and execution

## Monitoring

- **Coverage reports** are generated and uploaded to Codecov
- **Test results** are displayed in the GitHub Actions interface
- **Security scan results** are available in the Actions logs
- **Dependency updates** are tracked via Dependabot PRs

