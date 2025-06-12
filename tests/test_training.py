import os
import subprocess
import pytest


def test_train_runs():
    # Run training
    result = subprocess.run(
        ["python", "train.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    # Check artifact created
    model_path = os.path.join(os.getcwd(), "artifacts", "model.pkl")
    assert os.path.isfile(model_path)
