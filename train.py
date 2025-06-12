import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    # 1. Load data
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train a simple model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # 3. Save artifact
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(clf, "artifacts/model.pkl")
    print("Model trained and saved to artifacts/model.pkl")


if __name__ == "__main__":
    main()
