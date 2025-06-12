# tests/test_data.py
import os

def test_data_exists():
    assert os.path.isdir("data"), "data/ directory is missing"
