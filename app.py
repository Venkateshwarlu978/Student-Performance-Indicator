from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# ======================================
# HOME PAGE
# ======================================
@app.route('/')
def index():
    return render_template('index.html')

# ======================================
# PREDICTION PAGE
# ======================================
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        # Collect form data safely
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            # IMPORTANT FIX based on your TypeError
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)

        # Predict using pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        print("Prediction:", results[0])

        # Show prediction on page
        return render_template('home.html', results=results[0])


# ======================================
# RUN APPLICATION
# ======================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
