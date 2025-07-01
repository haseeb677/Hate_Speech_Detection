from flask import Flask, render_template, request, jsonify
import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import project modules
from src.preprocessing.text_preprocessing import TextPreprocessor
from src.features.feature_engineering import TextVectorizer
from src.models.model_training import ModelTrainer
from src.visualization.visualization import Visualizer

app = Flask(__name__, template_folder='../../templates', static_folder='../../static')

# Initialize the text preprocessor
text_preprocessor = TextPreprocessor()

# Load the model and vectorizer if they exist
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/hate_speech_model.joblib')
vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../models/vectorizer.joblib')

# Check if model and vectorizer files exist
model_exists = os.path.exists(model_path)
vectorizer_exists = os.path.exists(vectorizer_path)

# Initialize model and vectorizer
model = None
vectorizer = None

# Load model and vectorizer if they exist
if model_exists and vectorizer_exists:
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
else:
    print("Model or vectorizer not found. Please train the model first.")

# Initialize the visualizer
visualizer = Visualizer()

# Store feedback data
feedback_data = []

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Process the input text and make a prediction."""
    # Get the input text from the request
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Preprocess the text
        preprocessed_text = text_preprocessor.preprocess_text(text)
        
        # Vectorize the text
        if hasattr(vectorizer, 'transform'):
            X = vectorizer.transform([preprocessed_text])
        else:
            X = vectorizer.fit_transform([preprocessed_text])
        
        # Make a prediction
        prediction = model.predict(X)[0]
        
        # Get the probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            probability = float(max(probabilities))
        
        # Determine the result label
        if prediction == 1:
            result = "Hate Speech"
        else:
            result = "Non-Hate Speech"
        
        # Return the prediction result
        response = {
            'prediction': result,
            'probability': probability,
            'preprocessed_text': preprocessed_text
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Collect user feedback on predictions."""
    data = request.get_json()
    text = data.get('text', '')
    prediction = data.get('prediction', '')
    correct = data.get('correct', False)
    
    if not text or not prediction:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Store the feedback
    feedback_entry = {
        'text': text,
        'prediction': prediction,
        'correct': correct,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    feedback_data.append(feedback_entry)
    
    # Save feedback to a CSV file
    feedback_df = pd.DataFrame(feedback_data)
    feedback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/feedback.csv')
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    feedback_df.to_csv(feedback_path, index=False)
    
    return jsonify({'success': True, 'message': 'Feedback recorded successfully'})

@app.route('/visualize')
def visualize():
    """Render the visualization page."""
    return render_template('visualize.html')

@app.route('/get_wordcloud', methods=['GET'])
def get_wordcloud():
    """Generate and return a word cloud image."""
    try:
        # Load feedback data if available
        feedback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/feedback.csv')
        
        if os.path.exists(feedback_path):
            feedback_df = pd.read_csv(feedback_path)
            texts = feedback_df['text'].tolist()
        else:
            # Use some default texts if no feedback data is available
            texts = ["This is a sample text for the word cloud"]
        
        # Generate the word cloud
        fig = visualizer.plot_wordcloud(texts, title='Common Words in Analyzed Texts')
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image as base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return jsonify({'image': img_str})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)