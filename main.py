import pandas as pd 
import re 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

nltk.download('stopwords')

# Loading the dataset I got from kaggle
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()

#Testing the data if it works correctly
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Performing basic preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


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
clf = LogisticRegression(max_iter=200)
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


# Prediction function - Both Models to compare results

def predict_category(text):
    categories = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    clean = preprocess(text)
    vect = vectorizer.transform([clean])
    
    
    pred_lr = clf.predict(vect)[0]
    pred_nb = nb_model.predict(vect)[0]
    
    
    prob_lr = clf.predict_proba(vect)[0]
    prob_nb = nb_model.predict_proba(vect)[0]
    
    
    avg_prob = (prob_lr + prob_nb) / 2
    final_pred = avg_prob.argmax() + 1  
    
    print(f"Input text: '{text}'")
    print(f"Logistic Regression prediction: {categories[pred_lr]}")
    print(f"Naive Bayes prediction: {categories[pred_nb]}")
  
    
    return categories[final_pred]

result = predict_category(" Trojans Open Up No")
print(f"\nFinal result: {result}") 


#This is my base idea of how my logic would work for the Text Classification task. 
#This is just a testing file, and just filling the tasks based on the requirements for the assignment!