import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

class TextVectorizer:
    """
    A class for converting text data into numerical features for machine learning models.
    
    This class provides methods for vectorizing text using different techniques:
    - Bag of Words (BoW)
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - Transformer-based embeddings (BERT, DistilBERT, etc.)
    """
    
    def __init__(self, vectorization_type='tfidf', max_features=10000, transformer_model=None):
        """
        Initialize the TextVectorizer with the specified vectorization type.
        
        Args:
            vectorization_type (str): The type of vectorization to use. 
                                      Options: 'bow', 'tfidf', 'transformer'.
                                      Default is 'tfidf'.
            max_features (int): Maximum number of features for BoW and TF-IDF.
                               Default is 10000.
            transformer_model (str): The name of the transformer model to use.
                                    Default is None. If vectorization_type is 'transformer',
                                    this should be specified (e.g., 'distilbert-base-uncased').
        """
        self.vectorization_type = vectorization_type.lower()
        self.max_features = max_features
        self.transformer_model = transformer_model
        self.vectorizer = None
        self.tokenizer = None
        self.model = None
        
        # Initialize the appropriate vectorizer
        if self.vectorization_type == 'bow':
            self.vectorizer = CountVectorizer(max_features=self.max_features)
        elif self.vectorization_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        elif self.vectorization_type == 'transformer':
            if self.transformer_model is None:
                self.transformer_model = 'distilbert-base-uncased'  # Default transformer model
            self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_model)
            self.model = AutoModel.from_pretrained(self.transformer_model)
        else:
            raise ValueError(f"Unsupported vectorization type: {self.vectorization_type}. "
                           f"Supported types are 'bow', 'tfidf', and 'transformer'.")
    
    def fit(self, texts):
        """
        Fit the vectorizer on the provided texts.
        
        Args:
            texts (list): A list of preprocessed text documents.
            
        Returns:
            self: The fitted vectorizer.
        """
        if self.vectorization_type in ['bow', 'tfidf']:
            self.vectorizer.fit(texts)
        # For transformer models, no fitting is required
        return self
    
    def transform(self, texts):
        """
        Transform the texts into numerical features.
        
        Args:
            texts (list): A list of preprocessed text documents.
            
        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: The vectorized texts.
        """
        if self.vectorization_type in ['bow', 'tfidf']:
            return self.vectorizer.transform(texts)
        elif self.vectorization_type == 'transformer':
            return self._get_transformer_embeddings(texts)
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer on the texts and transform them.
        
        Args:
            texts (list): A list of preprocessed text documents.
            
        Returns:
            numpy.ndarray or scipy.sparse.csr.csr_matrix: The vectorized texts.
        """
        if self.vectorization_type in ['bow', 'tfidf']:
            return self.vectorizer.fit_transform(texts)
        elif self.vectorization_type == 'transformer':
            self.fit(texts)  # No-op for transformers
            return self.transform(texts)
    
    def _get_transformer_embeddings(self, texts):
        """
        Get embeddings from a transformer model.
        
        Args:
            texts (list): A list of preprocessed text documents.
            
        Returns:
            numpy.ndarray: The embeddings for each text.
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Initialize a list to store the embeddings
        embeddings = []
        
        # Process each text and get its embedding
        for text in texts:
            # Tokenize the text and convert to tensor
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            # Get the embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use the [CLS] token embedding as the sentence embedding
            # This is a common approach, but there are other methods as well
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def get_feature_names(self):
        """
        Get the feature names (for BoW and TF-IDF only).
        
        Returns:
            list: The feature names.
        """
        if self.vectorization_type in ['bow', 'tfidf']:
            return self.vectorizer.get_feature_names_out()
        else:
            raise NotImplementedError("Feature names are not available for transformer models.")