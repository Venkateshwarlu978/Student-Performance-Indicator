from flask import Flask, request, render_template
import logging
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ---------------------------
# ENABLE SERVER LOGGING
# ---------------------------
logging.basicConfig(level=logging.DEBUG)

application = Flask(__name__)
app = application

# Show errors in Render logs
app.config["DEBUG"] = True
app.config["PROPAGATE_EXCEPTIONS"] = True


# ============================
# HOME PAGE
# ============================
@app.route('/')
def index():
    logging.info("➡ Loading index.html")
    return render_template('index.html')


# ============================
# PREDICTION PAGE
# ============================
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            logging.info("➡ GET request - loading home.html")
            return render_template('home.html')

        else:
            logging.info("➡ POST request - running prediction")

            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=int(request.form.get('reading_score')),
                writing_score=int(request.form.get('writing_score'))
            )

            pred_df = data.get_data_as_data_frame()
            logging.debug(f"Input DataFrame:\n{pred_df}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            logging.info(f"➡ Prediction result: {results[0]}")

            return render_template('home.html', results=results[0])

    except Exception as e:
        logging.error("🔥 ERROR OCCURRED IN /predictdata", exc_info=True)
        return "Internal Server Error", 500


# ============================
# GLOBAL ERROR HANDLER
# ============================
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error("🔥 GLOBAL SERVER ERROR:", exc_info=True)
    return "Internal Server Error", 500


# ============================
# RUN APP (LOCAL ONLY)
# ============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

