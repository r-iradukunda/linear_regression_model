# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np


def load_and_inspect_data(filepath):
    """Load and inspect the dataset."""
    data = pd.read_csv(filepath)
    print(data.head())
    print(data.info())
    return data


def visualize_data(data):
    """Visualize feature relationships with the target variable."""
    plt.figure(figsize=(16, 6))

    # Scatter plot for km_driven vs selling_price
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x="km_driven", y="selling_price", data=data, color="green", marker="o"
    )
    plt.title("KM Driven vs Selling Price", fontsize=14, pad=15)
    plt.xlabel("Kilometers Driven", fontsize=12)
    plt.ylabel("Selling Price", fontsize=12)

    # Boxplot for fuel types vs selling_price
    plt.subplot(1, 2, 2)
    sns.boxplot(x="fuel", y="selling_price", data=data, palette="Set2", dodge=False)
    plt.title("Fuel Types vs Selling Price", fontsize=14, pad=15)
    plt.xlabel("Fuel Type", fontsize=12)
    plt.ylabel("Selling Price", fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.show()


def preprocess_data(data):
    """Preprocess the dataset by calculating vehicle age and encoding categorical features."""
    # Calculate vehicle age
    data["vehicle_age"] = datetime.now().year - data["year"]

    # Drop unnecessary columns
    data.drop(columns=["owner", "seller_type", "year", "transmission"], inplace=True)

    # Encode categorical features
    encoding = {}
    categorical_cols = ["name", "fuel"]
    le = LabelEncoder()
    for col in categorical_cols:
        data["encoded_" + col] = le.fit_transform(data[col])
        encoding[col] = le  # Save the LabelEncoder for each column

    # Drop original categorical columns
    data.drop(columns=["name", "fuel"], inplace=True)

    # Save the encoders
    with open("encoding.pkl", "wb") as f:
        pickle.dump(encoding, f)

    return data


def split_data(data):
    """Split the data into training and testing sets."""
    X = data.drop(columns=["selling_price"])
    y = data["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train Linear Regression, Random Forest, and Decision Tree models."""
    linear_reg = LinearRegression()
    random_forest = RandomForestRegressor()
    decision_tree = DecisionTreeRegressor()

    linear_reg.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)
    decision_tree.fit(X_train, y_train)

    return linear_reg, random_forest, decision_tree


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate models using RMSE and R2 score metrics."""
    results = {}
    for model_name, model in models.items():
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        r2 = r2_score(y_test, model.predict(X_test))

        results[model_name] = {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "r2_score": r2,
        }

        print(
            f"{model_name} - Training RMSE: {train_rmse}, Test RMSE: {test_rmse}, R2 Score: {r2}"
        )

    return results


def plot_predictions(model, X_test, y_test):
    """Plot the predicted vs actual values for the given model."""
    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid()
    plt.show()


def save_best_model(models, results):
    """Save the best performing model based on RMSE."""
    best_model_name = min(results, key=lambda k: results[k]["test_rmse"])
    best_model = models[best_model_name]

    # Save the model for future use
    with open(f"{best_model_name}.sav", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model saved: {best_model_name}")


def main(filepath):
    # Step 1: Load and inspect data
    data = load_and_inspect_data(filepath)

    # Step 2: Visualize data
    visualize_data(data)

    # Step 3: Preprocess data
    data = preprocess_data(data)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(data)

    # Step 5: Train models
    models = train_models(X_train, y_train)
    models_dict = {
        "Linear Regression": models[0],
        "Random Forest": models[1],
        "Decision Tree": models[2],
    }

    # Step 6: Evaluate models
    results = evaluate_models(models_dict, X_train, X_test, y_train, y_test)

    # Step 7: Plot predictions for the best model (Random Forest as an example)
    plot_predictions(models_dict["Random Forest"], X_test, y_test)

    # Step 8: Save the best model
    save_best_model(models_dict, results)


# Run the main function with the dataset path
if __name__ == "__main__":
    main("/content/drive/MyDrive/ML_SUMMATIVE/CAR_DETAILS_FROM_CAR_DEKHO.csv")
