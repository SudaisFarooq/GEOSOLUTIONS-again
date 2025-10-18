from flask import Flask, request, jsonify, render_template
from predict import predict_flood

app = Flask(__name__)

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/floodPrediction.html')
def flood_prediction_page():
    return render_template('floodPrediction.html')

@app.route('/index2.html')
def case_study_page():
    return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Log incoming request
        print("Incoming JSON:", request.data)

        data = request.get_json(force=True)  # force=True ensures JSON parsing even if headers are missing
        start_date = data.get("startDate")
        end_date = data.get("endDate")
        lat = float(data.get("latitude"))
        lon = float(data.get("longitude"))

        # Validate
        if not (start_date and end_date and lat and lon):
            return jsonify({"error": "Missing required parameters"}), 400

        result = predict_flood(start_date, end_date, lat, lon)
        return jsonify(result)

    except Exception as e:
        # Always return JSON on error
        import traceback
        print(traceback.format_exc())  # Log full traceback for debugging
        return jsonify({"error": str(e)}), 400
if __name__ == "__main__":
    app.run(debug=True)