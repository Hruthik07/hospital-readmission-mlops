"""
Test suite for Hospital Readmission MLOps Project.

Tests project structure, requirements, and basic model functionality.
"""

import os


def test_project_structure():
    """Basic test to check if important folders exist."""
    assert os.path.exists("src"), "src folder missing!"
    assert os.path.exists("models"), "models folder missing!"
    assert os.path.exists("reports"), "reports folder missing!"
    assert os.path.exists("data"), "data folder missing!"


def test_requirements_file():
    """Check if requirements.txt file exists."""
    assert os.path.exists("requirements.txt"), "requirements.txt missing!"
    assert os.path.exists("requirements_train.txt"), "requirements_train.txt missing!"


def test_source_files_exist():
    """Verify that key source files exist."""
    key_files = [
        "src/app_streamlit.py",
        "src/data_preprocessing.py",
        "src/fastapi_deploy.py",
        "src/train_xgboost_balanced.py"
    ]
    for file_path in key_files:
        assert os.path.exists(file_path), f"{file_path} missing!"


def test_gitignore_exists():
    """Check if .gitignore file exists."""
    assert os.path.exists(".gitignore"), ".gitignore missing!"


def test_readme_exists():
    """Check if README.md file exists and has content."""
    assert os.path.exists("README.md"), "README.md missing!"
    with open("README.md", "r") as f:
        content = f.read()
        assert len(content) > 0, "README.md is empty!"
        assert "Hospital Readmission" in content, "README.md missing project title!"
