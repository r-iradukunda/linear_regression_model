from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import numpy as np

# Load the saved models
with open('encoding.pkl', 'rb') as enc_file:
    encoding = pickle.load(enc_file)

with open('random_forest.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize FastAPI instance
app = FastAPI()

# Middleware for CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Selling Price Prediction API! Visit /docs for Swagger UI."}

# Pydantic models for request and response validation
class PredictionRequest(BaseModel):
    name: str = Field(..., description="The name of the vehicle (categorical)")
    km_driven: int = Field(..., ge=0, description="Kilometers driven (non-negative integer)")
    fuel: str = Field(..., description="The type of fuel used (categorical)")
    vehicle_age: int = Field(..., ge=0, le=50, description="Vehicle age in years (0-50)")

class PredictionResponse(BaseModel):
    selling_price: float = Field(..., description="Predicted selling price of the vehicle")
    details: Optional[dict] = Field(None, description="Details of the input after encoding")

# Endpoint for prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Encode categorical features
        if request.name not in encoding['name'].classes_:
            raise HTTPException(status_code=400, detail=f"Name '{request.name}' is not recognized.")
        if request.fuel not in encoding['fuel'].classes_:
            raise HTTPException(status_code=400, detail=f"Fuel type '{request.fuel}' is not recognized.")

        encoded_name = encoding['name'].transform([request.name])[0]
        encoded_fuel = encoding['fuel'].transform([request.fuel])[0]

        # Prepare input array
        input_data = np.array([[encoded_name, request.km_driven, encoded_fuel, request.vehicle_age]])

        # Make prediction
        selling_price = model.predict(input_data)[0]

        return PredictionResponse(
            selling_price=float(selling_price),
            details={
                "encoded_name": encoded_name,
                "encoded_fuel": encoded_fuel,
                "km_driven": request.km_driven,
                "vehicle_age": request.vehicle_age
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {e}")
