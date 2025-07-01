#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Hate Speech Detection Model

This script trains a hate speech detection model using the data in the data directory
and saves the trained model to the models directory.
"""

import os
import sys
import pandas as pd
import joblib
import logging
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.text_preprocessing import preprocess_text
from src.features.feature_engineering import TextVectorizer
from src.models.model_training import ModelTrainer
from src.visualization.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main function to train the model."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting model training process")
    
    # Load the data
    try:
        train_data_path = os.path.join('data', 'train_data.csv')
        sample_data_path = os.path.join('data', 'sample_data.csv')
        
        # Check if train_data.csv exists, otherwise use sample_data.csv
        if os.path.exists(train_data_path):
            logger.info(f"Loading training data from {train_data_path}")
            data = pd.read_csv(train_data_path)
        elif os.path.exists(sample_data_path):
            logger.info(f"Training data not found. Using sample data from {sample_data_path}")
            data = pd.read_csv(sample_data_path)
        else:
            logger.error("No training or sample data found. Please add data to the data directory.")
            return
        
        logger.info(f"Loaded {len(data)} records")
        
        # Preprocess the text
        logger.info("Preprocessing text data")
        data['preprocessed_text'] = data['text'].apply(preprocess_text)
        
        # Split the data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            data['preprocessed_text'], 
            data['label'], 
            test_size=0.2, 
            random_state=42
        )
        
        logger.info(f"Split data into {len(X_train)} training and {len(X_test)} testing samples")
        
        # Create and fit the vectorizer
        logger.info("Vectorizing text data using TF-IDF")
        vectorizer = TextVectorizer(method='tfidf')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train the model
        logger.info("Training the model")
        model_trainer = ModelTrainer(model_type='logistic_regression')
        model_trainer.train(X_train_vec, y_train)
        
        # Evaluate the model
        logger.info("Evaluating the model")
        accuracy, precision, recall, f1, conf_matrix, report = model_trainer.evaluate(X_test_vec, y_test)
        
        logger.info(f"Model performance:\n"
                   f"Accuracy: {accuracy:.4f}\n"
                   f"Precision: {precision:.4f}\n"
                   f"Recall: {recall:.4f}\n"
                   f"F1 Score: {f1:.4f}\n")
        
        logger.info(f"Classification Report:\n{report}")
        
        # Create visualizations
        logger.info("Creating visualizations")
        visualizer = Visualizer()
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('static/images', exist_ok=True)
        
        # Save confusion matrix visualization
        cm_fig = visualizer.plot_confusion_matrix(conf_matrix, ['Non-Hate', 'Hate'])
        cm_fig.savefig('static/images/confusion_matrix.png')
        
        # Save ROC curve visualization
        y_prob = model_trainer.predict_proba(X_test_vec)[:, 1]
        roc_fig = visualizer.plot_roc_curve(y_test, y_prob)
        roc_fig.savefig('static/images/roc_curve.png')
        
        # Save class distribution visualization
        class_dist_fig = visualizer.plot_class_distribution(data['label'])
        class_dist_fig.savefig('static/images/class_distribution.png')
        
        # Save word cloud visualization
        hate_texts = data[data['label'] == 1]['preprocessed_text']
        non_hate_texts = data[data['label'] == 0]['preprocessed_text']
        
        if not hate_texts.empty:
            hate_wc_fig = visualizer.generate_wordcloud(' '.join(hate_texts))
            hate_wc_fig.savefig('static/images/hate_wordcloud.png')
        
        if not non_hate_texts.empty:
            non_hate_wc_fig = visualizer.generate_wordcloud(' '.join(non_hate_texts))
            non_hate_wc_fig.savefig('static/images/non_hate_wordcloud.png')
        
        # Save the model and vectorizer
        logger.info("Saving model and vectorizer")
        model_path = os.path.join('models', 'hate_speech_model.joblib')
        vectorizer_path = os.path.join('models', 'vectorizer.joblib')
        
        joblib.dump(model_trainer.model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.exception(f"An error occurred during model training: {str(e)}")


if __name__ == "__main__":
    main()