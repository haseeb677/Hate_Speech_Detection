import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

class Visualizer:
    """
    A class for creating visualizations for hate speech detection models.
    
    This class provides methods for visualizing model performance,
    data distributions, and text features.
    """
    
    def __init__(self, figsize=(10, 6), style='whitegrid'):
        """
        Initialize the Visualizer with the specified figure size and style.
        
        Args:
            figsize (tuple): The default figure size. Default is (10, 6).
            style (str): The seaborn style to use. Default is 'whitegrid'.
        """
        self.figsize = figsize
        sns.set_style(style)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix', cmap='Blues', normalize=False):
        """
        Plot a confusion matrix.
        
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
            labels (list): The class labels. Default is None.
            title (str): The title of the plot. Default is 'Confusion Matrix'.
            cmap (str): The colormap to use. Default is 'Blues'.
            normalize (bool): Whether to normalize the confusion matrix. Default is False.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize the confusion matrix if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, cbar=True, 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        # Set the labels and title
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        # Rotate the tick labels
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curve(self, y_true, y_score, title='ROC Curve'):
        """
        Plot a ROC curve for binary classification.
        
        Args:
            y_true: The true labels (binary).
            y_score: The predicted probabilities for the positive class.
            title (str): The title of the plot. Default is 'ROC Curve'.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Compute the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the ROC curve
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Set the labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig
    
    def plot_class_distribution(self, y, labels=None, title='Class Distribution'):
        """
        Plot the distribution of classes.
        
        Args:
            y: The class labels.
            labels (list): The class names. Default is None.
            title (str): The title of the plot. Default is 'Class Distribution'.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Count the occurrences of each class
        class_counts = pd.Series(y).value_counts().sort_index()
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the class distribution
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
        
        # Set the labels and title
        if labels is not None:
            ax.set_xticklabels(labels)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title(title)
        
        # Add count labels on top of the bars
        for i, count in enumerate(class_counts.values):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, top_n=20, title='Feature Importance'):
        """
        Plot the importance of features for a model.
        
        Args:
            model: The trained model with a feature_importances_ attribute.
            feature_names (list): The names of the features.
            top_n (int): The number of top features to show. Default is 20.
            title (str): The title of the plot. Default is 'Feature Importance'.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Check if the model has feature importances
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            raise ValueError("The model does not have feature importances or coefficients.")
        
        # Get the feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
        
        # Create a DataFrame with feature names and importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance and take the top N
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the feature importances
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        
        # Set the title
        ax.set_title(title)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig
    
    def plot_wordcloud(self, texts, title='Word Cloud', max_words=100, background_color='white'):
        """
        Generate a word cloud from a list of texts.
        
        Args:
            texts (list): A list of text documents.
            title (str): The title of the plot. Default is 'Word Cloud'.
            max_words (int): The maximum number of words to include. Default is 100.
            background_color (str): The background color. Default is 'white'.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Combine all texts into a single string
        text = ' '.join(texts)
        
        # Create the word cloud
        wordcloud = WordCloud(max_words=max_words, background_color=background_color,
                             width=800, height=400, random_state=42).generate(text)
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig
    
    def plot_interactive_confusion_matrix(self, y_true, y_pred, labels=None, title='Confusion Matrix'):
        """
        Create an interactive confusion matrix using Plotly.
        
        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
            labels (list): The class labels. Default is None.
            title (str): The title of the plot. Default is 'Confusion Matrix'.
            
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object.
        """
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create the heatmap
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted Label", y="True Label", color="Count"),
                       x=labels if labels else list(range(len(cm))),
                       y=labels if labels else list(range(len(cm))),
                       title=title,
                       color_continuous_scale='Blues')
        
        # Add annotations
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append({
                    'x': j,
                    'y': i,
                    'text': str(cm[i, j]),
                    'showarrow': False,
                    'font': {'color': 'white' if cm[i, j] > cm.max() / 2 else 'black'}
                })
        
        fig.update_layout(annotations=annotations)
        
        return fig
    
    def plot_interactive_roc_curve(self, y_true, y_score, title='ROC Curve'):
        """
        Create an interactive ROC curve using Plotly.
        
        Args:
            y_true: The true labels (binary).
            y_score: The predicted probabilities for the positive class.
            title (str): The title of the plot. Default is 'ROC Curve'.
            
        Returns:
            plotly.graph_objects.Figure: The Plotly figure object.
        """
        # Compute the ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (area = {roc_auc:.2f})'
        ))
        
        # Add the diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        # Update the layout
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(x=0.7, y=0.1),
            width=700,
            height=500
        )
        
        return fig
    
    def plot_common_words(self, texts, top_n=20, title='Most Common Words'):
        """
        Plot the most common words in a list of texts.
        
        Args:
            texts (list): A list of text documents.
            top_n (int): The number of top words to show. Default is 20.
            title (str): The title of the plot. Default is 'Most Common Words'.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        # Combine all texts and split into words
        words = ' '.join(texts).split()
        
        # Count the occurrences of each word
        word_counts = Counter(words)
        
        # Get the top N words
        top_words = pd.DataFrame(word_counts.most_common(top_n), columns=['Word', 'Count'])
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot the word counts
        sns.barplot(x='Count', y='Word', data=top_words, ax=ax)
        
        # Set the title
        ax.set_title(title)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        return fig