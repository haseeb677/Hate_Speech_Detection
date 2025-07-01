#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Sample Model

This script creates a sample model and vectorizer for testing purposes.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    """Create sample model and vectorizer for testing."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create a simple logistic regression model
    model = LogisticRegression()
    
    # Create a simple TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Save the model and vectorizer
    model_path = os.path.join('models', 'hate_speech_model.joblib')
    vectorizer_path = os.path.join('models', 'vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Sample model saved to {model_path}")
    print(f"Sample vectorizer saved to {vectorizer_path}")
    print("\nNote: This is just a placeholder model for testing the application.")
    print("For a real model, please run 'python scripts/train_model.py'")

if __name__ == "__main__":
    main()