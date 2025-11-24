# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Created `CONTRIBUTING.md` with development guidelines and contribution instructions
- Created `src/config.py` for centralized configuration management
- Created `docs/API.md` with comprehensive API documentation
- Created `.python-version` file to specify Python 3.11 as the standard version
- Added module-level docstrings to key Python files
- Added comprehensive docstrings to all API endpoints
- Added error handling for model loading in Streamlit app
- Added input validation schema with examples in FastAPI
- Expanded test suite from 2 to 5 tests
- Added linting job to GitHub Actions CI/CD pipeline
- Added pytest execution to CI/CD pipeline
- Added pull request trigger to CI/CD workflow

### Changed
- Updated MLflow from 2.15.1 to 2.17.2 (fixes 4 security vulnerabilities)
- Standardized scikit-learn version to 1.5.2 across both requirements files
- Added `imbalanced-learn==0.12.4` to requirements_train.txt
- Updated Python version to 3.11 in CI/CD pipeline for consistency
- Improved all error messages with better context

### Fixed
- Fixed all 20 PEP8 style violations identified by flake8
- Removed 6 unused imports across multiple files
- Fixed trailing whitespace issues in app_streamlit.py
- Fixed missing newlines at end of files
- Fixed f-string placeholder issues in 5 files
- Fixed improper spacing between functions and classes
- Fixed duplicate import and redefinition in fastapi_deploy.py

### Security
- Updated MLflow to version 2.17.2 to address:
  - Weak password requirements authentication bypass
  - Model creation directory traversal RCE vulnerability
  - Local file read/path traversal in dbfs
  - Excessive directory permissions allowing privilege escalation
- No critical security issues found in code (bandit scan clean)

## [1.0.0] - Previous Version

### Added
- Initial release with MLOps pipeline
- Streamlit web application
- FastAPI deployment
- MLflow experiment tracking
- XGBoost model training with SMOTE balancing
- Hyperparameter tuning
- CI/CD pipeline with GitHub Actions
- Comprehensive README documentation
