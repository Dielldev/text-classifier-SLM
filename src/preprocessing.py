import pandas as pd
from utils import preprocess
import config


def load_data():
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, test_df


def preprocess_data(train_df, test_df):
 
    train_df['clean_text'] = train_df['Description'].apply(preprocess)
    test_df['clean_text'] = test_df['Description'].apply(preprocess)
    
    print("\nPreprocessing complete!")
    print("Sample of cleaned data:")
    print(train_df[['Description', 'clean_text']].head())
    
    return train_df, test_df


def vectorize_data(train_df, test_df, vectorizer):

    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])
    y_train = train_df['Class Index']
    y_test = test_df['Class Index']
    
    print(f"\nVectorization complete!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, vectorizer
