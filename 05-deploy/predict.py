import pickle
from typing import Any, Dict, Literal
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field

app = FastAPI(title="churn-prediction")


# Defining request
class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

# Defining Response
class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool   

C = 1.0
output_file = f'model_C={C}.bin'

# Loading the model

with open(output_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
async def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

# # Testing the model with a random customer

# customer = {
#     "customerid": "8879-zkjof",
#     "gender": "female",
#     "seniorcitizen": 0,
#     "partner": "no",
#     "dependents": "no",
#     "tenure": 1,
#     "phoneservice": "no",
#     "multiplelines": "no",
#     "internetservice": "dsl",
#     "onlinesecurity": "no",
#     "onlinebackup": "yes",
#     "deviceprotection": "no",
#     "techsupport": "no",
#     "streamingtv": "no",
#     "streamingmovies": "no",
#     "contract": "month-to-month",
#     "paperlessbilling": "yes",
#     "paymentmethod": "electronic_check",
#     "monthlycharges": 29.85,
#     "totalcharges": 29.85
# }

# # X = dv.transform([customer])
# y_pred = pipeline.predict_proba(customer)[0, 1]

# print('input', customer)
# print('churn probability =', y_pred)