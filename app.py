from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None if loading fails

# Function to determine AQI category and class
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "good"
    elif aqi_value <= 100:
        return "Moderate", "moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "sensitive"
    elif aqi_value <= 200:
        return "Unhealthy", "unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy", "very-unhealthy"
    else:
        return "Hazardous", "hazardous"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded. Please check the server logs."

    try:
        # Get inputs and convert to float
        features = [
            float(request.form["T"]),
            float(request.form["TM"]),
            float(request.form["Tm"]),
            float(request.form["SLP"]),
            float(request.form["H"]),
            float(request.form["VV"]),
            float(request.form["V"]),
            float(request.form["VM"])
        ]
        
        # Convert input to a NumPy array
        input_data = np.array([features])

        # Make prediction
        aqi_value = model.predict(input_data)[0]  # Extract single prediction

        # Determine AQI category and class
        category_text, category_class = get_aqi_category(aqi_value)

        return render_template('result.html', aqi_value=aqi_value, category_text=category_text, category_class=category_class)

    except Exception as e:
        return f"Error in prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for error tracking
