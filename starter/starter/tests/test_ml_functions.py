import pandas as pd
from ml.model import train_model, inference
from ml.data import process_data
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("tests/census_subset.csv")
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data():
    X_train, y_train, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape == (9, 33)
    assert y_train.shape == (9,)


def _is_fitted(model, X_train) -> bool:
    from sklearn.exceptions import NotFittedError

    try:
        model.predict(X_train)
        return True
    except NotFittedError as e:
        return False


def test_train_model():
    X_train, y_train, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)
    assert _is_fitted(model, X_train)


def test_inference():
    X_train, y_train, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    y_pred = inference(model, X_train)

    assert y_pred.shape == (9,)
    assert y_pred[y_pred == 0].shape == (7,)
    assert y_pred[y_pred == 1].shape == (2,)
