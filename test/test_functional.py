import unittest
import os
import sys
import numpy as np
import pandas as pd

# Adjusting path to import TestUtils and your module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test.TestUtils import TestUtils
from main import load_and_prepare_data as load_credit_data, apply_smote, hypothesis, hinge_loss


class TestCreditRiskSVM(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = load_credit_data("credit_risk_dataset.csv")

    def test_data_loading_and_preprocessing(self):
        try:
            if not self.df.empty and all(col in self.df.columns for col in [
                'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
            ]):
                self.test_obj.yakshaAssert("TestDataLoadingAndPreprocessing", True, "functional")
                print("TestDataLoadingAndPreprocessing = Passed")
            else:
                self.test_obj.yakshaAssert("TestDataLoadingAndPreprocessing", False, "functional")
                print("TestDataLoadingAndPreprocessing = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestDataLoadingAndPreprocessing", False, "functional")
            print("TestDataLoadingAndPreprocessing = Failed")

    def test_smote_application(self):
        try:
            from sklearn.preprocessing import StandardScaler

            X = self.df.drop(columns='loan_status')
            y = self.df['loan_status']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_res, y_res = apply_smote(X_scaled, y)

            if len(np.unique(y_res, return_counts=True)[1]) == 2 and len(X_res) == len(y_res):
                self.test_obj.yakshaAssert("TestSMOTEApplication", True, "functional")
                print("TestSMOTEApplication = Passed")
            else:
                self.test_obj.yakshaAssert("TestSMOTEApplication", False, "functional")
                print("TestSMOTEApplication = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestSMOTEApplication", False, "functional")
            print("TestSMOTEApplication = Failed")

    def test_model_predictions_match_expected(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            X = self.df.drop(columns='loan_status')
            y = self.df['loan_status']
            X_scaled = StandardScaler().fit_transform(X)
            X_res, y_res = apply_smote(X_scaled, y)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            expected_preds = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0])

            if np.array_equal(y_pred[:10], expected_preds):
                self.test_obj.yakshaAssert("TestModelPredictionsMatchExpected", True, "functional")
                print("TestModelPredictionsMatchExpected = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelPredictionsMatchExpected", False, "functional")
                print("TestModelPredictionsMatchExpected = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestModelPredictionsMatchExpected", False, "functional")
            print("TestModelPredictionsMatchExpected = Failed")

    def test_hinge_loss_output(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            X = self.df.drop(columns='loan_status')
            y = self.df['loan_status']
            X_scaled = StandardScaler().fit_transform(X)
            X_res, y_res = apply_smote(X_scaled, y)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            model.fit(X_train, y_train)

            scores = hypothesis(model, X_test)
            loss = hinge_loss(y_test.values, scores)
            expected_loss = 0.36837249806112676

            if np.isclose(loss, expected_loss, atol=1e-5):
                self.test_obj.yakshaAssert("TestHingeLossOutput", True, "functional")
                print("TestHingeLossOutput = Passed")
            else:
                self.test_obj.yakshaAssert("TestHingeLossOutput", False, "functional")
                print("TestHingeLossOutput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestHingeLossOutput", False, "functional")
            print("TestHingeLossOutput = Failed")
            
    def test_csv_loaded_successfully(self):
        try:
            required_columns = [
                'person_home_ownership',
                'loan_intent',
                'loan_grade',
                'cb_person_default_on_file',
                'loan_status'
            ]
            if not self.df.empty and all(col in self.df.columns for col in required_columns):
                self.test_obj.yakshaAssert("TestCSVLoadedSuccessfully", True, "functional")
                print("TestCSVLoadedSuccessfully = Passed")
            else:
                self.test_obj.yakshaAssert("TestCSVLoadedSuccessfully", False, "functional")
                print("TestCSVLoadedSuccessfully = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestCSVLoadedSuccessfully", False, "functional")
            print("TestCSVLoadedSuccessfully = Failed")


import unittest
import os
import sys
import numpy as np
import pandas as pd
import joblib

# Adjust path to access main code and TestUtils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test.TestUtils import TestUtils
from car import load_and_prepare_data, explore_data, train_and_evaluate, prediction_demo, cost_function

class TestCarResaleModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = load_and_prepare_data("car_resale.csv")

    def test_csv_loaded_and_preprocessed(self):
        try:
            required_cols = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score', 'resale_price']
            if not self.df.empty and all(col in self.df.columns for col in required_cols):
                self.test_obj.yakshaAssert("TestCSVLoadedAndPreprocessed", True, "functional")
                print("TestCSVLoadedAndPreprocessed = Passed")
            else:
                self.test_obj.yakshaAssert("TestCSVLoadedAndPreprocessed", False, "functional")
                print("TestCSVLoadedAndPreprocessed = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestCSVLoadedAndPreprocessed", False, "functional")
            print("TestCSVLoadedAndPreprocessed = Failed")

    def test_explore_data_output(self):
        try:
            max_price = self.df['resale_price'].max()
            mean_price = self.df['resale_price'].mean()
            if max_price == 2500000 and np.isclose(mean_price, 1091152.55, atol=1):
                self.test_obj.yakshaAssert("TestExploreDataOutput", True, "functional")
                print("TestExploreDataOutput = Passed")
            else:
                self.test_obj.yakshaAssert("TestExploreDataOutput", False, "functional")
                print("TestExploreDataOutput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestExploreDataOutput", False, "functional")
            print("TestExploreDataOutput = Failed")

    def test_model_training_and_saving(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression

            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = self.df[features]
            y = self.df['resale_price']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model_path = "car_resale_model.pkl"
            train_and_evaluate(X_train, y_train, X_test, y_test, path=model_path)

            if os.path.exists(model_path):
                self.test_obj.yakshaAssert("TestModelTrainingAndSaving", True, "functional")
                print("TestModelTrainingAndSaving = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelTrainingAndSaving", False, "functional")
                print("TestModelTrainingAndSaving = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestModelTrainingAndSaving", False, "functional")
            print("TestModelTrainingAndSaving = Failed")

    def test_custom_mse_output(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LinearRegression

            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = self.df[features]
            y = self.df['resale_price']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = cost_function(y_test.values, y_pred)

            if np.isclose(mse, 19433096169.02, atol=1e2):
                self.test_obj.yakshaAssert("TestCustomMSEOutput", True, "functional")
                print("TestCustomMSEOutput = Passed")
            else:
                self.test_obj.yakshaAssert("TestCustomMSEOutput", False, "functional")
                print("TestCustomMSEOutput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestCustomMSEOutput", False, "functional")
            print("TestCustomMSEOutput = Failed")

    def test_prediction_demo_output(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = self.df[features]
            y = self.df['resale_price']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            model = joblib.load("car_resale_model.pkl")
            pred = model.predict([X_test[0]])

            if int(pred[0]) == 380276:  # From your console output
                self.test_obj.yakshaAssert("TestPredictionDemoOutput", True, "functional")
                print("TestPredictionDemoOutput = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictionDemoOutput", False, "functional")
                print("TestPredictionDemoOutput = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPredictionDemoOutput", False, "functional")
            print("TestPredictionDemoOutput = Failed")
