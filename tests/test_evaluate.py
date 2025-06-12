import os
import subprocess
import json
import pytest


def test_evaluate_and_metrics():
    # Ensure train has run
    subprocess.run(["python", "train.py"], check=True)

    # Run evaluation
    result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # Verify metrics file
    metrics_path = os.path.join("artifacts", "metrics.json")
    assert os.path.exists(metrics_path)

    with open(metrics_path) as f:
        data = json.load(f)

    assert "accuracy" in data
    assert 0.0 <= data["accuracy"] <= 1.0
