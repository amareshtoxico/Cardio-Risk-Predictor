import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# 1. Load the Pre-trained Model & Scaler
try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Please ensure 'best_model.pkl' and 'scaler.pkl' are in the same folder.")

# Define the exact order of columns as the model expects
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# Define which columns need scaling (based on your scaler.pkl)
SCALE_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


@app.route('/')
def home():
    # No need to pass model names anymore
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Collect Data from Form
        values = [float(request.form[col]) for col in FEATURE_COLUMNS]

        # 3. Create DataFrame (Required for Scaler)
        input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

        # 4. Apply Scaling to the specific columns
        input_df[SCALE_COLS] = scaler.transform(input_df[SCALE_COLS])

        # 5. Make Prediction
        prediction = model.predict(input_df)[0]

        # 6. Get Probability (Confidence Score)
        try:
            prob = model.predict_proba(input_df)[0][1]
            prob = round(prob * 100, 2)
            # If the prediction is 0 (Healthy), show the health probability instead
            if prediction == 0:
                prob = 100 - prob
        except:
            prob = None  # Some models don't support probability

        # 7. Send result to UI
        return render_template('result.html',
                               prediction=prediction,
                               probability=prob,
                               model="Random Forest (Best Model)")

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True)