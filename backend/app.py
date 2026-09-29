from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
import joblib
import os


app = FastAPI(
    title="House Price Prediction API",
    description="House price prediction using Linear Regression",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model.pkl"
)

model = joblib.load(MODEL_PATH)


class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    age: float


@app.get("/")
def home():

    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "frontend",
        "index.html"
    )

    return FileResponse(frontend_path)


@app.post("/predict")
def predict_price(data: HouseData):

    input_data = [[
        data.area,
        data.bedrooms,
        data.bathrooms,
        data.age
    ]]

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }