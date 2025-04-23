import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


# ---------------------------------- #
# 1. Load and Preprocess the Dataset
# ---------------------------------- #
def load_and_prepare_data(path="credit_risk_dataset.csv"):
    df = pd.read_csv(path)
    df.fillna(method="ffill", inplace=True)

    # Encode categorical features
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print(" Data loaded and preprocessed.")
    return df


# ---------------------------- #
# 2. Apply SMOTE to balance classes
# ---------------------------- #
def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(" SMOTE applied.")
    return X_res, y_res


# --------------------------------- #
# 3. Hypothesis Function (SVM scores)
# --------------------------------- #
def hypothesis(model, X):
    return model.decision_function(X)  # Raw decision boundary scores


# ------------------------------- #
# 4. Custom Cost Function (Hinge Loss)
# ------------------------------- #
def hinge_loss(y_true, scores):
    y_signed = np.where(y_true == 1, 1, -1)
    loss = np.maximum(0, 1 - y_signed * scores)
    return np.mean(loss)


# -------------------------------- #
# 5. Train the Model and Predict
# -------------------------------- #
def train_and_predict(df):
    X = df.drop(columns='loan_status')
    y = df['loan_status']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_resampled, y_resampled = apply_smote(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Use hypothesis (decision scores) and calculate hinge loss
    decision_scores = hypothesis(model, X_test)
    y_pred = model.predict(X_test)
    loss = hinge_loss(y_test.values, decision_scores)

    print(" Sample Predictions:", y_pred[:10])
    print(" Hinge Loss (Custom Cost):", loss)


# ---------------- #
# Main Execution
# ---------------- #
if __name__ == "__main__":
    df_credit = load_and_prepare_data("credit_risk_dataset.csv")
    train_and_predict(df_credit)
