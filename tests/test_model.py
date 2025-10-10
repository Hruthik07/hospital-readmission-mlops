# tests/test_model.py

import os
import pytest

def test_project_structure():
    """Basic test to check if important folders exist"""
    assert os.path.exists("src"), "src folder missing!"
    assert os.path.exists("models"), "models folder missing!"
    assert os.path.exists("reports"), "reports folder missing!"

def test_requirements_file():
    """Check if requirements.txt file exists"""
    assert os.path.exists("requirements.txt"), "requirements.txt missing!"
