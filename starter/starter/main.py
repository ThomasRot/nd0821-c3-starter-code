from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
import pickle

from ml.data import process_data
from ml.model import inference

app = FastAPI()

with open("../model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("../model/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
with open("../model/label_binarizer.pkl", "rb") as f:
    lb = pickle.load(f)

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


class InferenceData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(..., alias="capital-gain")
    capital_loss: float = Field(..., alias="capital-loss")
    hours_per_week: float = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "FastAPI model inference API for udacity!"}


@app.post("/predict")
async def predict(data: InferenceData) -> Dict[str, int]:
    X = pd.DataFrame([data.dict(by_alias=True)])
    X, *_ = process_data(
        X, categorical_features=cat_features, encoder=encoder, lb=lb, training=False
    )
    print("==" * 8, X)
    prediction = inference(model, X)
    return {"prediction": prediction}
