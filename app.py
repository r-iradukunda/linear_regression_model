# Import libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pickle
import uvicorn
import numpy as np
import sklearn

# Load the trained model
model_filename = "random_forest.sav"  # Change this if using another model
try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file '{model_filename}' not found. Please check the file path.")

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
try:
    with open('encoding.pkl', 'rb') as f:
        encoding = pickle.load(f)
except FileNotFoundError:
    raise Exception("Encoding file 'encoding.pkl' not found. Please check the file path.")

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Encode categorical features
        if request.name not in encoding['name'].classes_:
            raise HTTPException(status_code=400, detail=f"Car name '{request.name}' not recognized. Available options: {list(encoding['name'].classes_)}")
        if request.fuel not in encoding['fuel'].classes_:
            raise HTTPException(status_code=400, detail=f"Fuel type '{request.fuel}' not recognized. Available options: {list(encoding['fuel'].classes_)}")
        
        encoded_name = encoding['name'].transform([request.name])[0]
        encoded_fuel = encoding['fuel'].transform([request.fuel])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Prepare the input data for prediction
    input_data = np.array([
        request.vehicle_age,
        request.km_driven,
        encoded_name,
        encoded_fuel
    ]).reshape(1, -1)
    
    # Make the prediction
    try:
        prediction = model.predict(input_data)[0]
    except sklearn.exceptions.NotFittedError as e:
        raise HTTPException(status_code=500, detail="Model is not fitted. Please check the model file.")
    
    # Return the prediction
    return {"predicted_selling_price": prediction}

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)