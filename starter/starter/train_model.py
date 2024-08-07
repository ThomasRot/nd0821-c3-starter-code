# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

from ml.model import inference, train_model
from ml.data import process_data
from ml.model import compute_model_metrics


# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = train_model(X_train=X_train, y_train=y_train)

with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../model/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("../model/label_binarizer.pkl", "wb") as f:
    pickle.dump(lb, f)


y_pred = inference(model, X_test)
precision, recall, f1 = compute_model_metrics(y_test, y_pred)
print(f"The model achieved {precision=}, {recall=} and {f1=}")

with open("slice_output.txt", "w") as f:
    for cat_feature in cat_features:
        for option in test[cat_feature].unique():
            df = test.copy()
            df[cat_feature] = option
            X_test, y_test, _, _ = process_data(
                df,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False,
            )
            y_pred = inference(model, X_test)
            precision, recall, f1 = compute_model_metrics(y_test, y_pred)
            print(
                f"For slice with {cat_feature}={option} the model achieved {precision=}, {recall=} and {f1=}",
                file=f,
            )
