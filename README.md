# Text Classification Project

A machine learning REST API for text classification.

## Project Structure

```
├── app.py              # FastAPI application
├── main.py             # Core project logic, model comparison & confusion matrix
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
├── train.csv          # Training data
├── test.csv           # Test data
└── env/               # Virtual environment
```

## Overview

This project uses two machine learning models (Logistic Regression + Naive Bayes) with ensemble averaging for accurate text classification.

**Categories:**
- World (1)
- Sports (2) 
- Business (3)
- Sci/Tech (4)

## Dataset

**Source:** [AG News Classification Dataset - Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
**Source:** [AG News Classification Dataset - Hugging Face](https://huggingface.co/datasets/sh0416/ag_news)
- **Training samples:** 120,000
- **Test samples:** 7,600
- **Features:** News article descriptions
- **Target:** Class Index (1-4)


## Quick Start

### Installation

```bash
# Clone and navigate to the repository
git clone <"https://github.com/Dielldev/text-classifier-SLM">
cd text-classifier-SLM

# Create virtual environment
python -m venv env

# Activate virtual environment
source env/Scripts/activate  # Windows
# source env/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
python app.py
```

Access the interactive API documentation at: http://localhost:8000/docs

##  API Usage

### Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "NASA launches new Mars rover"}'
```

**Response:**
```json
{
  "input_text": "NASA launches new Mars rover",
  "logistic_regression_prediction": "Sci/Tech",
  "naive_bayes_prediction": "Sci/Tech",
  "final_prediction": "Sci/Tech",
  "confidence_scores": {
    "World": 0.10,
    "Sports": 0.05,
    "Business": 0.15,
    "Sci/Tech": 0.70
  }
}
```

## Features

-  Dual model prediction (Logistic Regression + Naive Bayes)
-  Ensemble averaging for improved accuracy
-  TF-IDF vectorization
-  Automatic text preprocessing
-  Confidence scores for all categories
-  Interactive API documentation (Swagger UI)
-  Health check endpoint

## Model Details

- **Preprocessing:** Lowercase conversion, punctuation removal, stopword filtering
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Models:** Logistic Regression (max_iter=200) + Multinomial Naive Bayes
- **Ensemble Method:** Average probability distribution

## Model Comparison 

### Why Logistic Regression + Naive Bayes Ensemble?

**Logistic Regression** excels at handling feature correlations and provides reliable probability estimates with "91% accuracy score", while **Naive Bayes** is fast and works exceptionally well with high-dimensional text data despite its "naive" independence assumption with a "89 accuracy score".

For single model deployment, **Logistic Regression** generally performs better on text classification tasks due to its ability to handle feature correlations in TF-IDF vectors.

## Dependencies

All project dependencies are listed in `requirements.txt`. Key packages include:

- **fastapi** - Modern web framework for building APIs
- **uvicorn** - ASGI web server for FastAPI
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning library
- **nltk** - Natural language processing toolkit
- **pydantic** - Data validation and settings management
- **matplotlib** - Plotting and visualization
- **numpy & scipy** - Scientific computing libraries

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Development Setup

The project includes:
- **`.gitignore`** - Excludes unnecessary files from version control (virtual environments, cache files, compiled Python files, etc.)
- **`requirements.txt`** - Pinned dependency versions for reproducible installations

## Author

**Diell Govori**


