from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "FastAPI model inference API for udacity!"}


def test_predict_valid_data():
    response = client.post(
        "/predict",
        json={
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        },
    )
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}


def test_predict_invalid_valid_data():
    response = client.post("/predict", json={"one": 2.5, "two": 4.5})
    assert response.status_code == 422
    assert "detail" in response.json()
