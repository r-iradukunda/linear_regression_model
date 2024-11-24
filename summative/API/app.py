from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import logging
import pandas as pd
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import RedirectResponse


# Initialize FastAPI app
app = FastAPI()


# Load encoders and the trained model
with open('encoding.pkl', 'rb') as f:
    encoding = pickle.load(f)

car_model = pickle.load(open('random_forest.sav', 'rb'))

# Define the input model
class CarDetails(BaseModel):
    name: str
    km_driven: float = Field(..., ge=-50)
    fuel: str
    vehicle_age: float = Field(..., ge=-50)

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Define the POST endpoint
@app.post('/predict')
def car_pred(input_parameters: CarDetails):
    try:
        # Transform categorical
        input_data = {
            'encoded_name': encoding['name'].transform([input_parameters.name])[0]
                               if input_parameters.name in encoding['name'].classes_
                               else -1,
            'km_driven': input_parameters.km_driven,
            'encoded_fuel': encoding['fuel'].transform([input_parameters.fuel])[0]
                             if input_parameters.fuel in encoding['fuel'].classes_
                             else -1,
            'vehicle_age': input_parameters.vehicle_age
        }

        input_data_df = pd.DataFrame([input_data])

        logging.info(f"Encoded input data: {input_data_df}")

        prediction = car_model.predict(input_data_df)

        return {"prediction": prediction[0]}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"error": "Prediction failed.", "details": str(e)}
