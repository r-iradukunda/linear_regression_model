# Car Selling Price Prediction Project:
This repository contains a Selling price Prediction project that consists of three main components: a machine learning model, a FastAPI backend, and a Flutter frontend.

**Project Structure**
summative/
├── API/
├── flutter_app/
└── linear_regression/
## 1. API
The API directory contains the FastAPI backend that serves the machine learning model and handles prediction requests.

Key Files
app/
main.py: The main FastAPI application file.
linear_regression_model.pkl: The trained linear regression model.
scaler.pkl: The scaler used for preprocessing.
poly.pkl: The polynomial features transformer.
label_encoders.pkl: The label encoders for categorical features.
Other necessary files for model loading and preprocessing.
How to Run Locally
Navigate to the API directory.
Install the required dependencies:
```pip install -r requirements.txt```
Start the FastAPI server:
```uvicorn app.main:app --reload```
## 2. flutter_app
The flutter_app directory contains the Flutter frontend application that allows users to input TV features and get price predictions.

Key Files
lib/
main.dart: The main entry point of the Flutter application.
How to Run
Ensure you have Flutter installed on your system.
Navigate to the flutter_app directory.
Get the Flutter dependencies:
```flutter pub get```
Run the Flutter application:
```flutter run```
## 3. linear_regression
The linear_regression directory contains the Jupyter notebooks and scripts used for training and evaluating the machine learning model.

Key Files
notebooks/
train_model.ipynb: The notebook used for training the linear regression model.
evaluate_model.ipynb: The notebook used for evaluating the model.
How to Use
Navigate to the linear_regression directory.
Install the required dependencies:
```bash
pip install -r requirements.txt
```
Open the Jupyter notebooks to train and evaluate the model:
jupyter notebook
Usage
FastAPI Backend
The FastAPI backend is deployed on Render and provides an endpoint for predicting TV prices based on input features. The endpoint is:

POST /predict: Predicts the selling price of car given its features.
Example request:

```bash
curl -X 'POST' \
  'https://linear-regression-model-0jqg.onrender.com/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "name": "Maruti Swift Dzire VDI",
  "fuel": "Pretol",
  "km_driven": 4500,
  "vehicle_age": 2
}'
```

## Flutter Frontend
The Flutter application provides a user-friendly interface for inputting car details and displaying the predicted Selling price.

Make sure the FastAPI backend URL in the Flutter application is set to the correct Render address:

```bash
final String apiUrl = 'https://tv-prices-api.onrender.com/predict';
```

Click for here [YouTube_Demo_Video](https://youtu.be/mysYXTmumiA)