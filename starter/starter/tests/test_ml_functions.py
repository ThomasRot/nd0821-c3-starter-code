import pandas as pd
from ml.model import train_model, inference
from ml.data import process_data
from sklearn.linear_model import LogisticRegression


def test_train_model():
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
    X_train, y_train, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    assert X_train.shape == (9, 33)
    assert y_train.shape == (9,)

    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)

    y_pred = inference(model, X_train)
    assert y_pred.shape == (9,)
