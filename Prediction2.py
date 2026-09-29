import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("priceM.pkl")


class ModelRequestData(BaseModel):
    total_square: float
    floor: float


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.get("/predict_get", response_model=Result)
def predict_get(total_square: float, floor: float):
    input_df = pd.DataFrame(
        {"total_square": total_square, "floor": floor}, index=[0]
    )
    result = model.predict(input_df)[0]
    return Result(result=float(result))


@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_df = pd.DataFrame(
        {"total_square": data.total_square, "floor": data.floor}, index=[0]
    )
    result = model.predict(input_df)[0]
    return Result(result=float(result))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
