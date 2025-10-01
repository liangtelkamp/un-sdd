# Classifier Test Suite

This directory contains a comprehensive pytest test suite for the three classifier classes in the un-sdd project.

## Test Structure

```
tests/
├── __init__.py                          # Test package initialization
├── conftest.py                          # Shared fixtures and test configuration
├── test_pii_classifier.py               # Tests for PIIClassifier
├── test_pii_sensitivity_classifier.py  # Tests for PIISensitivityClassifier
├── test_non_pii_classifier.py          # Tests for NonPIIClassifier
├── run_tests.py                        # Test runner script
└── README.md                           # This file
```

## Test Coverage

### PIIClassifier Tests
- ✅ No alphanumeric content detection (returns "None")
- ✅ Successful PII entity detection
- ✅ Case-insensitive entity matching
- ✅ AGE entity prioritization
- ✅ Undetermined output handling
- ✅ Exception handling during _run_prompt
- ✅ Version and k-value parameter passing
- ✅ Edge cases (empty inputs, None values)
- ✅ Context parameter validation

### PIISensitivityClassifier Tests
- ✅ "None" PII entity handling (returns "NON_SENSITIVE")
- ✅ Sensitivity level mapping (NON_SENSITIVE, MEDIUM_SENSITIVE, HIGH_SENSITIVE, SEVERE_SENSITIVE)
- ✅ Undetermined output handling
- ✅ Exception handling during _run_prompt
- ✅ Version and max_tokens parameter passing
- ✅ Edge cases (empty inputs, None values)
- ✅ Context parameter validation
- ✅ Mixed case keyword handling

### NonPIIClassifier Tests
- ✅ Sensitivity level mapping
- ✅ Undetermined output handling
- ✅ Exception handling during _run_prompt
- ✅ Version and max_tokens parameter passing
- ✅ Edge cases (empty inputs, None values)
- ✅ Context parameter validation
- ✅ Complex table context handling
- ✅ Complex ISP rules handling
- ✅ Large input handling
- ✅ Special character handling

## Running Tests

### Prerequisites
```bash
pip install pytest pytest-cov
```

### Run All Tests
```bash
# From project root
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=classifiers --cov-report=term-missing

# Using the test runner script
python tests/run_tests.py
```

### Run Specific Test Files
```bash
# Test PIIClassifier only
python -m pytest tests/test_pii_classifier.py -v

# Test PIISensitivityClassifier only
python -m pytest tests/test_pii_sensitivity_classifier.py -v

# Test NonPIIClassifier only
python -m pytest tests/test_non_pii_classifier.py -v
```

### Run Specific Test Classes or Methods
```bash
# Run specific test class
python -m pytest tests/test_pii_classifier.py::TestPIIClassifier -v

# Run specific test method
python -m pytest tests/test_pii_classifier.py::TestPIIClassifier::test_classify_no_alphanumeric_content -v
```

## Test Features

### Mocking Strategy
- All tests use mocks to avoid hitting real LLMs
- `_run_prompt` method is mocked to return predictable responses
- `_standardize_output` and `_map_sensitivity` methods are mocked as needed
- PII_ENTITIES_LIST is mocked to a deterministic small list

### Fixtures
- `pii_classifier`: PIIClassifier instance
- `pii_sensitivity_classifier`: PIISensitivityClassifier instance  
- `non_pii_classifier`: NonPIIClassifier instance
- `mock_run_prompt`: Mock for _run_prompt method
- `sample_standardized_output`: Sample successful output
- `sample_error_output`: Sample error output

### Parameterized Tests
- Sensitivity mapping tests use `@pytest.mark.parametrize` for comprehensive coverage
- Input validation tests cover various edge cases
- Case-insensitive matching tests cover different input formats

### Error Handling
- All tests include error path testing
- Exception scenarios are properly mocked and tested
- Graceful degradation is verified

## Test Quality Assurance

### Coverage Requirements
- 100% logical branch coverage for classify methods
- All edge cases covered (empty inputs, None values, exceptions)
- Both happy path and error path testing
- Parameter validation testing

### Test Organization
- Tests are organized by classifier class
- Each test file is independently runnable
- Clear test names and docstrings
- Comprehensive comments explaining test purposes

### Best Practices
- No real LLM calls during testing
- Deterministic test data
- Proper fixture usage
- Clean test isolation
- Comprehensive assertions

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the classifiers module is in the Python path
2. **Mock Failures**: Check that mock patches are applied correctly
3. **Test Timeouts**: Increase timeout for complex test scenarios

### Debug Mode
```bash
# Run tests with detailed output
python -m pytest tests/ -v -s --tb=long

# Run single test with debugging
python -m pytest tests/test_pii_classifier.py::TestPIIClassifier::test_classify_no_alphanumeric_content -v -s
```

## Contributing

When adding new tests:
1. Follow the existing naming conventions
2. Use appropriate fixtures from conftest.py
3. Include both positive and negative test cases
4. Add docstrings explaining test purpose
5. Ensure tests are deterministic and don't depend on external services