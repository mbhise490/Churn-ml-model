from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data_ingestion import DataIngestion
from model_prediction import ChurnPredictor


app = FastAPI(
    title="Telecom Churn Prediction API",
    description="OOP based Telecom Customer Churn Prediction API",
    version="1.0.0")

data_ingestion = DataIngestion()

predictor = ChurnPredictor(model_path="churn_logistic_regression.joblib")


class CustomerData(BaseModel):

    tenure: int

    onlinesecurity: str
    techsupport: str

    contract: str

    paperlessbilling: str
    paymentmethod: str

    monthlycharges: float


@app.get("/")
def home():

    return {"message": "Telecom Churn Prediction API",
            "status": "running"}


@app.get("/health")
def health():

    return {"status": "healthy",
        "model": predictor.model_name,
        "version": predictor.model_version}


@app.post("/predict")
def predict(customer: CustomerData):

    try:
        customer_data = customer.model_dump()

        customer_df = data_ingestion.prepare_data(customer_data)

        result = predictor.predict(customer_df)

        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))