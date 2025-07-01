#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the Hate Speech Detection Web Application

This script starts the Flask web application for hate speech detection.
"""

import os
import sys
import logging
from datetime import datetime

# Configure logging first
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

logger.info("Starting script execution")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
logger.info(f"System path: {sys.path}")

try:
    logger.info("Attempting to import app module")
    from src.app.app import app
    logger.info("Successfully imported app module")
    
    if __name__ == "__main__":
        logger.info("Starting Hate Speech Detection Web Application")
        
        # Check if model exists, if not, suggest training
        model_path = os.path.join('models', 'hate_speech_model.joblib')
        if not os.path.exists(model_path):
            logger.warning("Model not found. Please run scripts/train_model.py first.")
            print("\nWARNING: Model not found. Please run the following command to train the model:")
            print("python scripts/train_model.py\n")
        else:
            logger.info(f"Model found at {model_path}")
        
        # Run the app
        port = int(os.environ.get("PORT", 5000))
        logger.info(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True)
        
except Exception as e:
    logger.exception(f"An error occurred: {str(e)}")
    print(f"\nERROR: {str(e)}\n")