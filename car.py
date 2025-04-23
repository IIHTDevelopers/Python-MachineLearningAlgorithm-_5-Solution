import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


# ------------------------ #
#  Function 1: Load & Prep
# ------------------------ #
def load_and_prepare_data(path="car_resale.csv"):
    df = pd.read_csv(path)

    # Fill missing values
    df['fuel_type'] = df['fuel_type'].fillna('Petrol')  # default fallback
    df['brand_score'] = df['brand_score'].fillna(df['brand_score'].median())

    # Label encode categorical feature
    le = LabelEncoder()
    df['fuel_type'] = le.fit_transform(df['fuel_type'])  # Petrol = 1, Diesel = 0, etc.

    print("✅ Data loaded and preprocessed.")
    return df


# ------------------- #
# Function 2: EDA
# ------------------- #
def explore_data(df):
    max_price = df['resale_price'].max()
    mean_price = df['resale_price'].mean()
    print(f"\n💰 Max Resale Price: ₹{max_price}")
    print(f"📊 Avg Resale Price: ₹{mean_price:.2f}")


# ----------------------------------- #
# Function 3: Prediction Demo (Linear)
# ----------------------------------- #
def prediction_demo(model, X_sample):
    prediction = model.predict([X_sample])
    print(f"\n🚘 Predicted Resale Price: ₹{int(prediction[0])}")


# ----------------------------- #
# Function 4: Custom Cost (MSE)
# ----------------------------- #
def cost_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ----------------------------- #
# Function 5: Train & Evaluate
# ----------------------------- #
def train_and_evaluate(X_train, y_train, X_test, y_test, path="car_resale_model.pkl"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f"\n✅ Model trained and saved to '{path}'")

    y_pred = model.predict(X_test)
    cost = cost_function(y_test.values, y_pred)

    print(f"\n📉 Custom MSE: {cost:.2f}")
    print("🔍 Sample Predictions:", y_pred[:5])


# --------- Main Program ---------
if __name__ == "__main__":
    df = load_and_prepare_data("car_resale.csv")

    explore_data(df)

    features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
    X = df[features]
    y = df['resale_price']

    # Optional scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, y_train, X_test, y_test)

    # Sample prediction demo
    sample_input = X_test[0]  # Using the first test sample
    model = joblib.load("car_resale_model.pkl")
    prediction_demo(model, sample_input)
