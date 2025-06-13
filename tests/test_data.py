from sklearn.datasets import load_iris


def test_load_iris():
    """
    Ensure that the Iris dataset can be loaded via scikit-learn.
    """
    X, y = load_iris(return_X_y=True)
    assert X.shape[0] > 0 and X.shape[1] > 0
    assert len(y) > 0
