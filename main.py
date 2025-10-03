import pandas as pd 
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from utils import preprocess
import config

# Loading the dataset from Kaggle
train_df = pd.read_csv(config.TRAIN_DATA_PATH)
test_df = pd.read_csv(config.TEST_DATA_PATH)
train_df.head()

#Testing the data if it works correctly
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Cleaning the data
train_df['clean_text'] = train_df['Description'].apply(preprocess)
test_df['clean_text'] = test_df['Description'].apply(preprocess)
train_df[['Description', 'clean_text']].head()

# Vectorization and model training
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])
y_train = train_df['Class Index']
y_test = test_df['Class Index']

# 1st Model: Logistic Regression
clf = LogisticRegression(max_iter=config.LOGISTIC_REGRESSION_MAX_ITER)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title('Confusion Matrix Logic Regression')
plt.show()

#2nd Model: Naive Bayes

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
print('Naive Bayes Accuracy:', accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_nb)
plt.title('Confusion Matrix Naive Bayes')
plt.show()

# Saving all models in the models folder
os.makedirs(config.MODELS_DIR, exist_ok=True)

# Saving the vectorizer
with open(config.VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

# Logistic Regression model
with open(config.LOGISTIC_REGRESSION_PATH, 'wb') as f:
    pickle.dump(clf, f)

# Naive Bayes model  
with open(config.NAIVE_BAYES_PATH, 'wb') as f:
    pickle.dump(nb_model, f)

print("\n" + "="*50)
print("All models saved successfully!")
print(f"Models saved in: {config.MODELS_DIR}/")
print(f"  - {os.path.basename(config.VECTORIZER_PATH)}")
print(f"  - {os.path.basename(config.LOGISTIC_REGRESSION_PATH)}")
print(f"  - {os.path.basename(config.NAIVE_BAYES_PATH)}")
print("="*50)


# Prediction function - Both Models to compare results

def predict_category(text):
    
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    
    pred_lr = clf.predict(vect)[0]
    pred_nb = nb_model.predict(vect)[0]
    
    prob_lr = clf.predict_proba(vect)[0]
    prob_nb = nb_model.predict_proba(vect)[0]
    
    avg_prob = (prob_lr + prob_nb) / 2
    final_pred = avg_prob.argmax() + 1  
    
    print(f"\nInput text: '{text}'")
    print(f"Logistic Regression prediction: {config.CATEGORIES[pred_lr]}")
    print(f"Naive Bayes prediction: {config.CATEGORIES[pred_nb]}")
    print(f"Final ensemble prediction: {config.CATEGORIES[final_pred]}")
    
    return config.CATEGORIES[final_pred]


# Test the prediction function
if __name__ == "__main__":
    result = predict_category("ronaldo has scored a last minute goal")
    print(f"\nFinal result: {result}")