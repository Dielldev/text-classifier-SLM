from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pickle
import os
import uvicorn

from utils import preprocess
import config

app = FastAPI(title=config.API_TITLE, version=config.API_VERSION)


vectorizer = None
clf = None
nb_model = None


class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    input_text: str
    logistic_regression_prediction: str
    naive_bayes_prediction: str
    final_prediction: str
    confidence_scores: dict


def load_models():
    """Load pre-trained models from disk"""
    global vectorizer, clf, nb_model
    
    try:
        # Check if model files exist
        if not os.path.exists(config.VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Vectorizer not found at {config.VECTORIZER_PATH}. "
                "Please run main.py first to train and save models."
            )
        if not os.path.exists(config.LOGISTIC_REGRESSION_PATH):
            raise FileNotFoundError(
                f"Logistic Regression model not found at {config.LOGISTIC_REGRESSION_PATH}. "
                "Please run main.py first to train and save models."
            )
        if not os.path.exists(config.NAIVE_BAYES_PATH):
            raise FileNotFoundError(
                f"Naive Bayes model not found at {config.NAIVE_BAYES_PATH}. "
                "Please run main.py first to train and save models."
            )
        
        # Load the vectorizer
        with open(config.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Logistic Regression model
        with open(config.LOGISTIC_REGRESSION_PATH, 'rb') as f:
            clf = pickle.load(f)
        
        # Load Naive Bayes model
        with open(config.NAIVE_BAYES_PATH, 'rb') as f:
            nb_model = pickle.load(f)
        
        print("✓ Models loaded successfully from disk!")
        print(f"  - Vectorizer: {config.VECTORIZER_PATH}")
        print(f"  - Logistic Regression: {config.LOGISTIC_REGRESSION_PATH}")
        print(f"  - Naive Bayes: {config.NAIVE_BAYES_PATH}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR loading models: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    try:
        load_models()
    except Exception as e:
        print("The API will start but /predict endpoint will not work.")
     


@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    
    models_loaded = vectorizer is not None and clf is not None and nb_model is not None
    
    if not models_loaded:
        return {
            "status": "degraded",
            "models_loaded": False,
            "message": "Models not loaded. Run 'python main.py' to train and save models."
        }
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "message": "All systems operational"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_category(input_data: TextInput):
    
    if vectorizer is None or clf is None or nb_model is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please run 'python main.py' to train and save models first."
        )
    

    if not input_data.text or not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    if len(input_data.text) > 10000:
        raise HTTPException(status_code=400, detail="Input text too long (max 10000 characters)")
    
    try:
      
        clean = preprocess(input_data.text)
        vect = vectorizer.transform([clean])
        
        pred_lr = clf.predict(vect)[0]  
        pred_nb = nb_model.predict(vect)[0]  
        
        prob_lr = clf.predict_proba(vect)[0]  
        prob_nb = nb_model.predict_proba(vect)[0]  
        
        avg_prob = (prob_lr + prob_nb) / 2

        final_pred = avg_prob.argmax() + 1
        
        confidence_scores = {
            config.CATEGORIES[i+1]: float(avg_prob[i]) 
            for i in range(len(avg_prob))
        }
        
        return PredictionResponse(
            input_text=input_data.text,
            logistic_regression_prediction=config.CATEGORIES[pred_lr],
            naive_bayes_prediction=config.CATEGORIES[pred_nb],
            final_prediction=config.CATEGORIES[final_pred],
            confidence_scores=confidence_scores
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)


