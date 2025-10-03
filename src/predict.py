import pickle
import os
from utils import preprocess
import config


def load_models():
    
    if not os.path.exists(config.VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Models not found. Please train models first by running main.py"
        )
    
  
    with open(config.VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Load Logistic Regression model
    with open(config.LOGISTIC_REGRESSION_PATH, 'rb') as f:
        clf = pickle.load(f)
    
    # Load Naive Bayes model
    with open(config.NAIVE_BAYES_PATH, 'rb') as f:
        nb_model = pickle.load(f)
    
    print("Models loaded successfully!")
    return vectorizer, clf, nb_model


def predict_category(text, vectorizer, clf, nb_model, verbose=True):
   

    clean = preprocess(text)
    
    vect = vectorizer.transform([clean])
    
    pred_lr = clf.predict(vect)[0]
    pred_nb = nb_model.predict(vect)[0]
    
    prob_lr = clf.predict_proba(vect)[0]
    prob_nb = nb_model.predict_proba(vect)[0]
    
    avg_prob = (prob_lr + prob_nb) / 2
    final_pred = avg_prob.argmax() + 1  
    
    if verbose:
        print(f"\nInput text: '{text}'")
        print(f"Logistic Regression prediction: {config.CATEGORIES[pred_lr]}")
        print(f"Naive Bayes prediction: {config.CATEGORIES[pred_nb]}")
        print(f"Final ensemble prediction: {config.CATEGORIES[final_pred]}")
    
    return config.CATEGORIES[final_pred]


def predict_with_loaded_models(text):
   
    vectorizer, clf, nb_model = load_models()
    return predict_category(text, vectorizer, clf, nb_model)
