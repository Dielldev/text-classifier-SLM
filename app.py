from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import uvicorn


nltk.download('stopwords')

app = FastAPI(title="Text Classification API", version="1.0.0")

categories = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}


class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    input_text: str
    logistic_regression_prediction: str
    naive_bayes_prediction: str
    final_prediction: str
    confidence_scores: dict


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def load_models():
    """Load and train models on startup"""
    global vectorizer, clf, nb_model
    
  
    train_df = pd.read_csv('train.csv')
    train_df['clean_text'] = train_df['Description'].apply(preprocess)
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    y_train = train_df['Class Index']
    
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    print("Models loaded and trained successfully!")


#  The event handler to load the models when the API starts 
@app.on_event("startup")
async def startup_event():
    """Load models when the API starts"""
    load_models()


# This this basically ridirects the root URL to the API documentation page
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")


# To check if the APi is working fine
@app.get("/health")
async def health_check():
    # Check if models are initialized, return error if not
    if vectorizer is None or clf is None or nb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "healthy", "models_loaded": True}


# Main prediction Function 
@app.post("/predict", response_model=PredictionResponse)
async def predict_category(input_data: TextInput):
    """
    Predict the category of input text using both Logistic Regression and Naive Bayes models.
    Returns predictions from both models and a final combined prediction.
    """
    
    if vectorizer is None or clf is None or nb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
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
            categories[i+1]: float(avg_prob[i]) 
            for i in range(len(avg_prob))
        }
        
        #  This is the responses we get, teh model name and responds and also the confidence scores
        return PredictionResponse(
            input_text=input_data.text,
            logistic_regression_prediction=categories[pred_lr],
            naive_bayes_prediction=categories[pred_nb],
            final_prediction=categories[final_pred],
            confidence_scores=confidence_scores
        )
    
    except Exception as e:
       # This is just for error handling
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


