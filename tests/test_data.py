import os
import pytest


def test_data_exists():
    assert os.path.isdir("data"), "data/ directory is missing"
