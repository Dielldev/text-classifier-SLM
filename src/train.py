import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import config


def train_logistic_regression(X_train, y_train):
 
    print("\nTraining Logistic Regression model...")
    clf = LogisticRegression(max_iter=config.LOGISTIC_REGRESSION_MAX_ITER)
    clf.fit(X_train, y_train)
    print("Logistic Regression training complete!")
    
    return clf


def train_naive_bayes(X_train, y_train):

    print("\nTraining Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    print("Naive Bayes training complete!")
    
    return nb_model


def save_models(vectorizer, clf, nb_model):

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    with open(config.VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save Logistic Regression model
    with open(config.LOGISTIC_REGRESSION_PATH, 'wb') as f:
        pickle.dump(clf, f)
    
    # Save Naive Bayes model  
    with open(config.NAIVE_BAYES_PATH, 'wb') as f:
        pickle.dump(nb_model, f)
    
    print("\n" + "="*50)
    print("All models saved successfully!")
    print(f"Models saved in: {config.MODELS_DIR}/")
    print(f"  - {os.path.basename(config.VECTORIZER_PATH)}")
    print(f"  - {os.path.basename(config.LOGISTIC_REGRESSION_PATH)}")
    print(f"  - {os.path.basename(config.NAIVE_BAYES_PATH)}")
    print("="*50)
