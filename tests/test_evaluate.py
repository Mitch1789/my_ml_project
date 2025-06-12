import subprocess

def test_evaluate_and_metrics(tmp_path):
    # Ensure train has run
    subprocess.run(["python", "train.py"], check=True)
    # Run evaluation
    result = subprocess.run(
        ["python", "evaluate.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, result.stderr
    # Load metrics.json
    import json, os
    metrics_path = os.path.join("artifacts", "metrics.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path) as f:
        data = json.load(f)
    assert "accuracy" in data
    assert 0.0 <= data["accuracy"] <= 1.0
