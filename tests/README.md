# LLM Model Strategy Pattern Tests

This directory contains comprehensive tests for the LLM Model Strategy Pattern implementation.

## Test Structure

- `test_azure_strategy.py` - Tests for Azure OpenAI Strategy
- `conftest.py` - Shared fixtures and configuration
- `run_tests.py` - Test runner script
- `requirements-test.txt` - Testing dependencies

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Running All Tests

```bash
# Using the test runner script
python run_tests.py

# Or using pytest directly
pytest -v

# Run specific test file
pytest test_azure_strategy.py -v

# Run with coverage
pytest --cov=llm_model --cov-report=html
```

### Running Individual Test Classes

```bash
# Run only Azure strategy tests
pytest test_azure_strategy.py::TestAzureOpenAIStrategy -v

# Run only integration tests
pytest test_azure_strategy.py::TestAzureOpenAIStrategyIntegration -v
```

## Test Coverage

The tests cover:

### Azure OpenAI Strategy Tests
- ✅ Initialization with valid credentials
- ✅ Initialization from environment variables
- ✅ Error handling for missing credentials
- ✅ Text generation for standard models
- ✅ Text generation for O3 models
- ✅ Exception handling during generation
- ✅ Configuration retrieval
- ✅ Model component access
- ✅ Ready state checking
- ✅ Integration workflow testing

### Mocking Strategy
- All external dependencies are properly mocked
- Azure OpenAI client is mocked to avoid real API calls
- Environment variables are mocked for consistent testing
- API responses are mocked with realistic data

## Test Data

The tests use mock data to avoid:
- Real API calls to Azure OpenAI
- Network dependencies
- API key requirements
- Rate limiting issues

## Adding New Tests

When adding new strategy tests:

1. Create a new test file: `test_[strategy_name].py`
2. Follow the existing pattern with proper mocking
3. Include both unit tests and integration tests
4. Use the shared fixtures from `conftest.py`
5. Add comprehensive error handling tests

## Example Test Structure

```python
class TestNewStrategy:
    @pytest.fixture
    def mock_config(self):
        return {...}
    
    def test_initialization_success(self, mock_config):
        # Test successful initialization
        pass
    
    def test_generation(self, mock_config):
        # Test text generation
        pass
    
    def test_error_handling(self, mock_config):
        # Test error scenarios
        pass
```
