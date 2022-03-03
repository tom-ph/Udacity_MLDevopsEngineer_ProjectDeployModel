from catboost import CatBoostClassifier
from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
import pickle
from pydantic import BaseModel
import yaml

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

with open('starter/config.yaml') as stream:
    config = yaml.safe_load(stream)
label_encoder_path = config["model_training"]["label_encoder"]
model_path = config["model_training"]["trained_model_path"]
cat_features = config["data"]["categorical_features"]
model = CatBoostClassifier()
model.load_model(model_path)
with open(label_encoder_path, 'rb') as pkl_file:
    label_encoder = pickle.load(pkl_file)


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 29,
                "workclass": "Private",
                "fnlgt": 174391,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Never-married",
                "occupation": "Handlers-cleaners",
                "relationship": "Own-child",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "Italy"
            }
        }

# Default page
@app.get("/")
async def greetings():
    message = "Hello stranger, welcome to the Udacity Machine Learning DevOps Engineer Model Deploy exercise!"
    return {"message": message}

@app.post("/predict/")
async def predict(input_data: Union[InputData, list]):
    if isinstance(input_data, InputData):
        input_data = [input_data]
    input_data = [jsonable_encoder(row) for row in input_data]
    X = pd.DataFrame(input_data)
    X_prep, _, _ = process_data(X, cat_features, training=False, lb=label_encoder)
    preds = inference(model, X_prep)
    decoded_preds = list(label_encoder.inverse_transform(preds))
    return {"predictions": decoded_preds}


if __name__=="__main__":
    pass
    