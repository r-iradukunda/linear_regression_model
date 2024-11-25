# Import libraries
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import pickle
import uvicorn
import numpy as np

# Load the trained model
model_filename = "random_forest.sav"  # Change this if using another model
with open(model_filename, "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI(title="Car Selling Price Prediction API")

# Define the request body structure using Pydantic
class PredictionRequest(BaseModel):
    vehicle_age: int = Field(..., ge=0, le=50, description="Age of the vehicle in years")
    km_driven: int = Field(..., ge=0, description="Kilometers driven by the vehicle")
    name: str = Field(..., description="Name of the car brand/model")
    fuel: str = Field(..., description="Type of fuel used by the car (e.g., diesel, petrol)")

# CORS middleware configuration
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the encoders
with open('encoding.pkl', 'rb') as f:
    encoding = pickle.load(f)

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Encode categorical features
    encoded_name = encoding['name'].transform([request.name])[0]
    encoded_fuel = encoding['fuel'].transform([request.fuel])[0]
    
    # Prepare the input data for prediction
    input_data = np.array([
        request.vehicle_age,
        request.km_driven,
        encoded_name,
        encoded_fuel
    ]).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(input_data)[0]
    
    # Return the prediction
    return {"predicted_selling_price": prediction}

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)