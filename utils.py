"""
Utility functions for text preprocessing
"""
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords on module import (only downloads if not already present)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


def preprocess(text):
    """
    Preprocess text for classification.
    
    Steps:
    1. Convert to lowercase
    2. Remove non-alphabetic characters
    3. Remove stopwords
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
