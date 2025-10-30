import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(title="prediction application")

output_file = 'pipeline_v1.bin'


# Defining request
class Record(BaseModel):
    lead_source: Literal["referral", "events", "paid_ads", "social_media"]
    # industry: Literal["retail", "technology", "finance", "other", "NA"]
    # employment_status: Literal["self_employed", "employed", "unemployed", "student", "NA"]
    # location: Literal["australia", "north_america", "asia", "europe", "middle_east", "south_america"]
    number_of_courses_viewed: int = Field(ge=0)
    annual_income: float = Field(ge=0.0)
    # interaction_count: int = Field(ge=0.0)
    # lead_score: float = Field(ge=0.0)

# Defining Response
class PredictResponse(BaseModel):
    conversion_probability: float
    conversion: bool 

with open(output_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)


def single_pred(record):
    y_pred = pipeline.predict_proba(record)[0, 1]
    return float(y_pred)

@app.post("/predict")
async def pred(record: Record) -> PredictResponse:
    prob = single_pred(record.model_dump())

    return PredictResponse(
        conversion_probability=prob,
        conversion=bool(prob >= 0.5)
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)