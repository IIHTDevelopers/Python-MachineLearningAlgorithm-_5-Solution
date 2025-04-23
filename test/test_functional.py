import unittest
from test.TestUtils import TestUtils
import pandas as pd
import numpy as np
import io
import sys
import os
import joblib
from car import load_and_prepare_data as car_load_data, explore_data, prediction_demo, cost_function as car_cost_function, train_and_evaluate
from main import load_and_prepare_data as main_load_data, apply_smote, hypothesis, hinge_loss, train_and_predict
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TestCarFunctions(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        """
        Test case for car.py load_and_prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = car_load_data("car_resale.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded correctly
            expected_columns = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score', 'resale_price']
            
            if (isinstance(df, pd.DataFrame) and 
                all(col in df.columns for col in expected_columns) and
                "✅ Data loaded and preprocessed." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestCarLoadAndPrepareData", True, "functional")
                print("TestCarLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestCarLoadAndPrepareData", False, "functional")
                print("TestCarLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCarLoadAndPrepareData", False, "functional")
            print(f"TestCarLoadAndPrepareData = Failed | Exception: {e}")

    def test_explore_data(self):
        """
        Test case for car.py explore_data() function.
        """
        try:
            # Load the actual dataset
            df = car_load_data("car_resale.csv")
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with the actual dataset
            explore_data(df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if the output contains expected information
            output = captured_output.getvalue()
            
            # Create a simple DataFrame with just the resale_price column for testing
            # This is simpler than converting to numpy and back
            simple_df = pd.DataFrame({'resale_price': df['resale_price'].values})
            
            # Redirect stdout again
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with the simple DataFrame
            explore_data(simple_df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check output from the second test
            output_simple = captured_output.getvalue()
            
            if ("💰 Max Resale Price: ₹" in output and
                "📊 Avg Resale Price: ₹" in output):
                self.test_obj.yakshaAssert("TestExploreData", True, "functional")
                print("TestExploreData = Passed")
            else:
                self.test_obj.yakshaAssert("TestExploreData", False, "functional")
                print("TestExploreData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestExploreData", False, "functional")
            print(f"TestExploreData = Failed | Exception: {e}")

    def test_prediction_demo(self):
        """
        Test case for car.py prediction_demo() function.
        """
        try:
            # Load the actual dataset
            df = car_load_data("car_resale.csv")
            
            # Prepare features and target
            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = df[features]
            y = df['resale_price']
            
            # Optional scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train a model on the actual data
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            # Use a sample from the actual dataset
            sample_input = X_scaled[0]  # First row of the dataset
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            prediction_demo(model, sample_input)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if the output contains a prediction
            output = captured_output.getvalue()
            
            if "🚘 Predicted Resale Price: ₹" in output:
                self.test_obj.yakshaAssert("TestPredictionDemo", True, "functional")
                print("TestPredictionDemo = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictionDemo", False, "functional")
                print("TestPredictionDemo = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictionDemo", False, "functional")
            print(f"TestPredictionDemo = Failed | Exception: {e}")

    def test_cost_function(self):
        """
        Test case for car.py cost_function() function.
        """
        try:
            # Load the actual dataset
            df = car_load_data("car_resale.csv")
            
            # Prepare features and target
            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = df[features]
            y = df['resale_price']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train a model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get actual values as numpy array
            y_true = y_test.values
            
            # Call the function with real data
            cost = car_cost_function(y_true, y_pred)
            
            # Calculate expected MSE manually
            expected_cost = np.mean((y_true - y_pred) ** 2)
            
            if abs(cost - expected_cost) < 1e-10:  # Check if the values are close enough
                self.test_obj.yakshaAssert("TestCarCostFunction", True, "functional")
                print("TestCarCostFunction = Passed")
            else:
                self.test_obj.yakshaAssert("TestCarCostFunction", False, "functional")
                print("TestCarCostFunction = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestCarCostFunction", False, "functional")
            print(f"TestCarCostFunction = Failed | Exception: {e}")

    def test_train_and_evaluate(self):
        """
        Test case for car.py train_and_evaluate() function.
        """
        try:
            # Load the actual dataset
            df = car_load_data("car_resale.csv")
            
            # Prepare features and target
            features = ['age', 'original_price', 'mileage', 'fuel_type', 'num_owners', 'brand_score']
            X = df[features]
            y = df['resale_price']
            
            # Optional scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            
            # Create a temporary model file path
            temp_model_path = "temp_car_model.pkl"
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with actual data
            train_and_evaluate(X_train, y_train, X_test, y_test, temp_model_path)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if the model file was created and output contains expected information
            output = captured_output.getvalue()
            model_exists = os.path.exists(temp_model_path)
            
            # Clean up the temporary model file
            if model_exists:
                os.remove(temp_model_path)
            
            # Also test with numpy array for y_test
            y_test_np = y_test.values
            
            # Create another temporary model file path
            temp_model_path2 = "temp_car_model2.pkl"
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with numpy array
            train_and_evaluate(X_train, y_train, X_test, y_test_np, temp_model_path2)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if the model file was created and output contains expected information
            output_np = captured_output.getvalue()
            model_exists2 = os.path.exists(temp_model_path2)
            
            # Clean up the temporary model file
            if model_exists2:
                os.remove(temp_model_path2)
            
            if ((model_exists and
                 f"✅ Model trained and saved to '{temp_model_path}'" in output and
                 "📉 Custom MSE:" in output and
                 "🔍 Sample Predictions:" in output) or
                (model_exists2 and
                 f"✅ Model trained and saved to '{temp_model_path2}'" in output_np and
                 "📉 Custom MSE:" in output_np and
                 "🔍 Sample Predictions:" in output_np)):
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", True, "functional")
                print("TestTrainAndEvaluate = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
                print("TestTrainAndEvaluate = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print(f"TestTrainAndEvaluate = Failed | Exception: {e}")


class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        # Initialize TestUtils object for yaksha assertions
        self.test_obj = TestUtils()

    def test_load_and_prepare_data(self):
        """
        Test case for main.py load_and_prepare_data() function.
        """
        try:
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function
            df = main_load_data("credit_risk_dataset.csv")
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if data is loaded correctly
            if (isinstance(df, pd.DataFrame) and 
                "person_home_ownership" in df.columns and
                "loan_intent" in df.columns and
                "loan_grade" in df.columns and
                "cb_person_default_on_file" in df.columns and
                " Data loaded and preprocessed." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestMainLoadAndPrepareData", True, "functional")
                print("TestMainLoadAndPrepareData = Passed")
            else:
                self.test_obj.yakshaAssert("TestMainLoadAndPrepareData", False, "functional")
                print("TestMainLoadAndPrepareData = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestMainLoadAndPrepareData", False, "functional")
            print(f"TestMainLoadAndPrepareData = Failed | Exception: {e}")

    def test_apply_smote(self):
        """
        Test case for main.py apply_smote() function.
        """
        try:
            # Load the actual dataset
            df = main_load_data("credit_risk_dataset.csv")
            
            # Prepare features and target
            X = df.drop(columns='loan_status')
            y = df['loan_status']
            
            # Scale the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Take a sample of the data to make the test run faster
            # (SMOTE can be slow on large datasets)
            sample_size = min(100, len(X_scaled))
            indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[indices]
            y_sample = y.iloc[indices].values
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with actual data
            X_res, y_res = apply_smote(X_sample, y_sample)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if SMOTE balanced the classes
            unique, counts = np.unique(y_res, return_counts=True)
            balanced = len(unique) == 2 and counts[0] == counts[1]  # Equal number of each class
            
            if (balanced and
                " SMOTE applied." in captured_output.getvalue()):
                self.test_obj.yakshaAssert("TestApplySmote", True, "functional")
                print("TestApplySmote = Passed")
            else:
                self.test_obj.yakshaAssert("TestApplySmote", False, "functional")
                print("TestApplySmote = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestApplySmote", False, "functional")
            print(f"TestApplySmote = Failed | Exception: {e}")

    def test_hypothesis(self):
        """
        Test case for main.py hypothesis() function.
        """
        try:
            # Create a simple test case with a small dataset
            X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
            y = np.array([0, 0, 0, 1, 1, 1])  # Balanced classes
            
            # Train a model
            model = SVC(kernel='linear', probability=True, random_state=42)
            model.fit(X, y)
            
            # Call the function with test data
            scores = hypothesis(model, X)
            
            # Check if the function returns decision scores
            if (isinstance(scores, np.ndarray) and
                len(scores) == len(X)):
                self.test_obj.yakshaAssert("TestHypothesis", True, "functional")
                print("TestHypothesis = Passed")
            else:
                self.test_obj.yakshaAssert("TestHypothesis", False, "functional")
                print("TestHypothesis = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestHypothesis", False, "functional")
            print(f"TestHypothesis = Failed | Exception: {e}")

    def test_hinge_loss(self):
        """
        Test case for main.py hinge_loss() function.
        """
        try:
            # Create simple test data
            y_true = np.array([1, 0, 1, 0, 1, 0])
            scores = np.array([0.8, -0.5, 0.6, -0.3, 0.7, -0.4])
            
            # Call the function with test data
            loss = hinge_loss(y_true, scores)
            
            # Calculate expected hinge loss manually
            y_signed = np.where(y_true == 1, 1, -1)
            expected_loss = np.mean(np.maximum(0, 1 - y_signed * scores))
            
            if abs(loss - expected_loss) < 1e-10:  # Check if the values are close enough
                self.test_obj.yakshaAssert("TestHingeLoss", True, "functional")
                print("TestHingeLoss = Passed")
            else:
                self.test_obj.yakshaAssert("TestHingeLoss", False, "functional")
                print("TestHingeLoss = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestHingeLoss", False, "functional")
            print(f"TestHingeLoss = Failed | Exception: {e}")

    def test_train_and_predict(self):
        """
        Test case for main.py train_and_predict() function.
        """
        try:
            # Create a simple test DataFrame with the required columns
            data = {
                'person_home_ownership': [0, 1, 2, 0, 1, 0],
                'loan_intent': [0, 1, 2, 0, 1, 2],
                'loan_grade': [0, 1, 2, 0, 1, 2],
                'cb_person_default_on_file': [0, 1, 0, 1, 0, 1],
                'loan_status': [0, 1, 0, 1, 0, 1],  # Balanced classes
                'person_age': [25, 30, 35, 40, 45, 50],
                'person_income': [50000, 60000, 70000, 80000, 90000, 100000],
                'loan_amnt': [10000, 15000, 20000, 25000, 30000, 35000],
                'person_emp_length': [5, 6, 7, 8, 9, 10],
                'loan_int_rate': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'loan_percent_income': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                'cb_person_cred_hist_length': [5, 6, 7, 8, 9, 10]
            }
            df = pd.DataFrame(data)
            
            # Redirect stdout to capture print output
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Call the function with the test DataFrame
            train_and_predict(df)
            
            # Reset stdout
            sys.stdout = sys.__stdout__
            
            # Check if the output contains expected information
            output = captured_output.getvalue()
            
            if (" Sample Predictions:" in output):
                self.test_obj.yakshaAssert("TestTrainAndPredict", True, "functional")
                print("TestTrainAndPredict = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndPredict", False, "functional")
                print("TestTrainAndPredict = Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndPredict", False, "functional")
            print(f"TestTrainAndPredict = Failed | Exception: {e}")


if __name__ == '__main__':
    unittest.main()
