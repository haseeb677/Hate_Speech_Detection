import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

class ModelTrainer:
    """
    A class for training and evaluating machine learning models for hate speech detection.
    
    This class provides methods for training different types of models,
    evaluating their performance, and saving the trained models.
    """
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        """
        Initialize the ModelTrainer with the specified model type.
        
        Args:
            model_type (str): The type of model to train. 
                             Options: 'logistic_regression', 'naive_bayes', 'svm', 'random_forest'.
                             Default is 'logistic_regression'.
            random_state (int): Random seed for reproducibility. Default is 42.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        
        # Initialize the appropriate model
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif self.model_type == 'svm':
            self.model = SVC(random_state=self.random_state, probability=True)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. "
                           f"Supported types are 'logistic_regression', 'naive_bayes', 'svm', and 'random_forest'.")
    
    def train(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Train the model on the provided data.
        
        Args:
            X: The feature matrix.
            y: The target labels.
            test_size (float): The proportion of data to use for testing. Default is 0.2.
            validation_size (float): The proportion of training data to use for validation. Default is 0.1.
            
        Returns:
            self: The trained model.
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)
        
        # Further split the training data into training and validation sets
        if validation_size > 0:
            val_size = validation_size / (1 - test_size)  # Adjust validation size
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=self.random_state, stratify=y_train)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Store the classes
        self.classes_ = self.model.classes_
        
        # Evaluate the model
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        if validation_size > 0:
            val_accuracy = self.model.score(X_val, y_val)
            print(f"Validation accuracy: {val_accuracy:.4f}")
        
        test_accuracy = self.model.score(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Store the test data for later evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        return self
    
    def tune_hyperparameters(self, X, y, param_grid=None):
        """
        Tune the hyperparameters of the model using grid search.
        
        Args:
            X: The feature matrix.
            y: The target labels.
            param_grid (dict): The parameter grid to search. If None, a default grid will be used.
            
        Returns:
            self: The tuned model.
        """
        if param_grid is None:
            # Define default parameter grids for each model type
            if self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif self.model_type == 'naive_bayes':
                param_grid = {
                    'alpha': [0.1, 0.5, 1, 2, 5, 10]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                }
            elif self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
        
        # Create a grid search object
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Fit the grid search to the data
        grid_search.fit(X, y)
        
        # Print the best parameters and score
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Update the model with the best estimator
        self.model = grid_search.best_estimator_
        
        return self
    
    def evaluate(self, X=None, y=None):
        """
        Evaluate the model on the provided data or the test data.
        
        Args:
            X: The feature matrix. If None, the test data will be used.
            y: The target labels. If None, the test data will be used.
            
        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        if X is None or y is None:
            if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                X = self.X_test
                y = self.y_test
            else:
                raise ValueError("No test data available. Please provide X and y or train the model first.")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Print the evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Return the metrics as a dictionary
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions on the provided data.
        
        Args:
            X: The feature matrix.
            
        Returns:
            numpy.ndarray: The predicted labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates for each class.
        
        Args:
            X: The feature matrix.
            
        Returns:
            numpy.ndarray: The probability estimates.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"The {self.model_type} model does not support probability estimates.")
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): The path to save the model to.
            
        Returns:
            str: The filepath where the model was saved.
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): The path to load the model from.
            
        Returns:
            ModelTrainer: The loaded model.
        """
        return joblib.load(filepath)