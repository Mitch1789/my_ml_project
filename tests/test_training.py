import subprocess
import os

def test_train_runs():
    # 1. Run train.py
    result = subprocess.run(
        ["python", "train.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, result.stderr

    # 2. Assert the model artifact exists in ./artifacts/model.pkl
    model_path = os.path.join(os.getcwd(), "artifacts", "model.pkl")
    assert os.path.isfile(model_path), f"Expected model artifact at {model_path}"
