from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Trained Model and Vectorizer ---
# These files must be in the same directory as app.py
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found.")
    print("Please run the train.py script first to generate these files.")
    model = None
    vectorizer = None

# --- Define Routes ---

# Route for the main homepage
@app.route('/')
def home():
    # Renders the index.html template
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded. Please check server logs.'})

    # Get the text from the form submission
    text_data = request.json.get('text')
    if not text_data:
        return jsonify({'error': 'No text provided.'})

    # Prepare the text using the loaded vectorizer
    text_vector = vectorizer.transform([text_data])

    # Make a prediction using the loaded model
    prediction = model.predict(text_vector)

    # Determine the sentiment label
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    # Return the result as JSON
    return jsonify({'sentiment': sentiment})

# --- Run the App ---
if __name__ == '__main__':
    # The app will run on http://127.0.0.1:5000/
    app.run(debug=True)

