import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Data files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model files
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
LOGISTIC_REGRESSION_PATH = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
NAIVE_BAYES_PATH = os.path.join(MODELS_DIR, 'naive_bayes_model.pkl')


CATEGORIES = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}


LOGISTIC_REGRESSION_MAX_ITER = 200

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Text Classification API"
API_VERSION = "1.0.0"
