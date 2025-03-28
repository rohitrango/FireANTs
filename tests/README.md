# FireANTs Test Suite

This directory contains comprehensive tests for the FireANTs (Fast Iterative RiEmannian ANTs) library. The tests verify the correctness and functionality of various registration methods and tools.

## Test Structure

The test suite is organized into several files:

- `test_basic_registration.py`: Tests for moments, affine and rigid registration methods
- `test_deformable_registration.py`: Tests for deformable registration methods (greedy and SyN)
- `test_ants_compatibility.py`: Tests for ANTs format compatibility and inverse transformations
- `test_advanced_features.py`: Tests for multi-resolution behavior, similarity metrics, and robustness
- `test_diffeomorphic_properties.py`: Tests for diffeomorphic properties and Jacobian determinant

## Test Data

The test suite uses synthetic test data created on-the-fly for most tests. For file-based tests, images are temporarily saved to the `test_data` directory.

## Running Tests

### Prerequisites

Before running the tests, make sure you have:

1. Installed FireANTs
2. Installed test dependencies (`pytest`, `numpy`, `torch`, `SimpleITK`)
3. For ANTs compatibility tests: installed ANTs and made it available in the PATH

### Running All Tests

To run all tests:

```bash
cd /path/to/fireants  # Navigate to the FireANTs root directory
pytest -v tests/
```

### Running Specific Test Files

To run specific test files:

```bash
pytest -v tests/test_basic_registration.py
```

### Running Specific Test Classes or Methods

To run a specific test class:

```bash
pytest -v tests/test_basic_registration.py::TestAffineRegistration
```

To run a specific test method:

```bash
pytest -v tests/test_basic_registration.py::TestAffineRegistration::test_affine_registration_tensor
```

## Test Parameters

The tests use a smaller image size (32x32x32) and fewer iterations than would be used in a real-world scenario to ensure tests run quickly. For performance testing in real-world conditions, adjust these parameters accordingly.

## Skipped Tests

Some tests may be skipped if the necessary dependencies are not available:

- ANTs compatibility tests will be skipped if ANTs is not installed or not available in the PATH

## Adding New Tests

When adding new tests:

1. Follow the existing pattern for test organization
2. Use the provided pytest fixtures (`create_test_images`, `save_test_images`, `compute_similarity`) for consistency
3. Ensure tests are independent and don't rely on the state from other tests
4. For slow tests, consider adding the `@pytest.mark.slow` decorator

## Continuous Integration

These tests are designed to be run in a CI/CD pipeline to ensure code changes don't break existing functionality. 