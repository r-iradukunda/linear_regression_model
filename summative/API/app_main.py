import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and encodings
MODEL_PATH = os.path.join("models", "random_forest.sav")
ENCODING_PATH = os.path.join("models", "encoding.pkl")

try:
    # Load the trained model
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    # Load the encoding mappings
    with open(ENCODING_PATH, "rb") as encoding_file:
        encoding = pickle.load(encoding_file)
except Exception as e:
    raise ValueError(f"Error loading model or encodings: {e}")


# Define the input schema
class CarDetails(BaseModel):
    name: str
    fuel: str
    km_driven: int
    vehicle_age: int


@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.post("/predict/")
async def car_price(car_details: CarDetails):
    try:
        # Extract input features
        features = [
            car_details.km_driven,
            car_details.vehicle_age,
            encoding["name"].get(car_details.name, -1),
            encoding["fuel"].get(car_details.fuel, -1),
        ]

        # Check for unknown categories
        if -1 in features:
            raise HTTPException(
                status_code=400,
                detail="Unknown value encountered in categorical features.",
            )

        # Make prediction
        prediction = model.predict([features])[0]
        return {"Car_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
