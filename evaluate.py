import os
import sys

import joblib
import json

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    model_path = "artifacts/model.pkl"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    clf = joblib.load(model_path)

    X, y = load_iris(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f)

    # Optionally fail if below absolute minimum
    # if acc < 0.0:
    #     sys.exit(1)


if __name__ == "__main__":
    main()
