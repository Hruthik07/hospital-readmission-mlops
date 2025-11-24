# Contributing to Hospital Readmission MLOps Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/hruthik07/hospital-readmission-mlops.git
   cd hospital-readmission-mlops
   ```

2. **Set up Python environment**
   ```bash
   # Use Python 3.11
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # For development and deployment
   pip install -r requirements.txt
   
   # For training (includes MLflow, additional ML tools)
   pip install -r requirements_train.txt
   
   # For development (linting, testing)
   pip install pytest flake8 pylint bandit
   ```

## Code Quality Standards

### Style Guidelines
- Follow PEP 8 style guide
- Maximum line length: 120 characters
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules

### Before Submitting Code

1. **Run linting checks**
   ```bash
   flake8 src/ tests/ --max-line-length=120
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v
   ```

3. **Check for security issues**
   ```bash
   bandit -r src/
   ```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Maintain or improve test coverage
- Place tests in the `tests/` directory

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following code quality standards
4. Commit your changes with clear messages
5. Push to your fork (`git push origin feature/amazing-feature`)
6. Open a Pull Request with a clear description

## Project Structure

```
hospital-readmission-mlops/
├── .github/workflows/     # CI/CD pipelines
├── data/                  # Data files
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/               # Trained models
├── reports/              # Model reports and visualizations
├── src/                  # Source code
│   ├── app_streamlit.py           # Streamlit web app
│   ├── fastapi_deploy.py          # FastAPI deployment
│   ├── data_preprocessing.py      # Data preprocessing
│   ├── train_*.py                 # Training scripts
│   └── mlflow_*.py               # MLflow tracking scripts
├── tests/                # Test files
└── requirements*.txt     # Dependencies
```

## Questions?

Feel free to open an issue for any questions or concerns!
