# ============================================
# IMPORT LIBRARIES
# ============================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import joblib
import os

# Import our custom modules
from database import init_db, save_prediction
from report_generator import generate_pdf_report

# ============================================
# FLASK APP SETUP
# ============================================

app = Flask(__name__)

# Allow React frontend connection
CORS(app)

# Initialize Database
init_db()


# ============================================
# LOAD BEST TRAINED MODEL
# ============================================

# Load saved best model
model = joblib.load("model.pkl")


# ============================================
# LOAD METRICS
# ============================================

# Load all classifier metrics
metrics = joblib.load("metrics.pkl")


# ============================================
# HOME ROUTE
# ============================================

@app.route("/")
def home():

    return jsonify({
        "message": "Diabetes Prediction API is Running Successfully"
    })


# ============================================
# PREDICTION ROUTE
# ============================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------
        # GET DATA FROM FRONTEND
        # ------------------------------------

        data = request.get_json()

        # Check if input exists
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract values
        inputs = data.get("data", [])
        patient_name = data.get("patient_name", "Anonymous Patient")


        # ------------------------------------
        # VALIDATION
        # ------------------------------------

        # Expecting 5 features: Insulin, Glucose, BloodPressure, BMI, Age
        if len(inputs) != 5:

            return jsonify({
                "error":
                "Expected 5 values: Insulin, Glucose, BloodPressure, BMI, Age"
            }), 400


        # ------------------------------------
        # CONVERT TO NUMPY ARRAY
        # ------------------------------------

        input_array = np.array(
            inputs,
            dtype=float
        ).reshape(1, -1)


        # ------------------------------------
        # MAKE PREDICTION
        # ------------------------------------

        prediction = model.predict(input_array)
        result = int(prediction[0])
        label = "Positive" if result == 1 else "Negative"

        # Generate Suggestion
        if result == 1:
            suggestion = "Your results indicate a high risk of diabetes. We strongly recommend reducing sugar intake, exercising regularly, monitoring your glucose levels daily, and consulting a healthcare professional immediately."
        else:
            suggestion = "Your results indicate a low risk of diabetes. To maintain this, continue a balanced diet, stay active with regular physical activity, and keep up with your annual health checkups."

        # ------------------------------------
        # SAVE TO DATABASE
        # ------------------------------------

        db_data = {
            "patient_name": patient_name,
            "insulin": inputs[0],
            "glucose": inputs[1],
            "blood_pressure": inputs[2],
            "bmi": inputs[3],
            "age": inputs[4],
            "prediction_result": result,
            "prediction_label": label,
            "suggestion": suggestion
        }
        save_prediction(db_data)


        # ------------------------------------
        # RETURN RESULT
        # ------------------------------------

        return jsonify({
            "prediction": result,
            "label": label,
            "suggestion": suggestion,
            "patient_name": patient_name
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================
# DOWNLOAD REPORT ROUTE
# ============================================

@app.route("/download-report", methods=["POST"])
def download_report():
    try:
        data = request.get_json()
        
        # Required data for PDF
        patient_data = data.get("patient_data")
        
        if not patient_data:
            return jsonify({"error": "Patient data is required to generate report"}), 400

        # Generate PDF
        pdf_filename = f"Report_{patient_data.get('patient_name', 'Patient')}.pdf"
        pdf_path = os.path.join(os.path.dirname(__file__), pdf_filename)
        
        generate_pdf_report(patient_data,  pdf_path)

        # Send file and then delete it
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# PERFORMANCE / METRICS ROUTE
# ============================================

# ============================================
# PERFORMANCE / METRICS ROUTE
# ============================================

@app.route("/performance", methods=["GET"])
def performance():

    try:

        # Get best model using F1 Score
        best_model_name = max(
            metrics,
            key=lambda model: metrics[model]["f1_score"]
        )

        best_model = {
            "name": best_model_name,
            "accuracy": metrics[best_model_name]["accuracy"],
            "precision": metrics[best_model_name]["precision"],
            "recall": metrics[best_model_name]["recall"],
            "f1_score": metrics[best_model_name]["f1_score"]
        }

        # Final response
        return jsonify({
            "SVM": metrics["SVM"],
            "KNN": metrics["KNN"],
            "NaiveBayes": metrics["NaiveBayes"],
            "BestModel": best_model
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )