import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    A class for preprocessing text data for hate speech detection.
    
    This class provides methods for cleaning and normalizing text data,
    including removing special characters, links, mentions, stopwords,
    and performing lemmatization.
    """
    
    def __init__(self, language='english'):
        """
        Initialize the TextPreprocessor with the specified language.
        
        Args:
            language (str): The language for stopwords. Default is 'english'.
        """
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def remove_urls(self, text):
        """
        Remove URLs from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with URLs removed.
        """
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def remove_mentions(self, text):
        """
        Remove mentions (e.g., @username) from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with mentions removed.
        """
        return re.sub(r'@\w+', '', text)
    
    def remove_hashtags(self, text):
        """
        Remove hashtags from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with hashtags removed.
        """
        return re.sub(r'#\w+', '', text)
    
    def remove_special_characters(self, text):
        """
        Remove special characters and punctuation from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with special characters removed.
        """
        # Keep spaces but remove other special characters
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_numbers(self, text):
        """
        Remove numbers from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with numbers removed.
        """
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text):
        """
        Remove extra whitespaces from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with extra whitespaces removed.
        """
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Text with stopwords removed.
        """
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_text)
    
    def lemmatize_text(self, text):
        """
        Lemmatize the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Lemmatized text.
        """
        word_tokens = word_tokenize(text)
        lemmatized_text = [self.lemmatizer.lemmatize(word) for word in word_tokens]
        return ' '.join(lemmatized_text)
    
    def preprocess_text(self, text):
        """
        Apply all preprocessing steps to the text.
        
        Args:
            text (str): The input text.
            
        Returns:
            str: Fully preprocessed text.
        """
        if not isinstance(text, str) or not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        
        # Remove special characters and numbers
        text = self.remove_special_characters(text)
        text = self.remove_numbers(text)
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        # Remove stopwords
        text = self.remove_stopwords(text)
        
        # Lemmatize text
        text = self.lemmatize_text(text)
        
        return text