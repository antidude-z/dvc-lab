import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.pkl")


class InputData(BaseModel):
    input: list[float]


@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.input).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": pred.tolist()}
