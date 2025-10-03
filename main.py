from sklearn.feature_extraction.text import TfidfVectorizer

# Import functions from src modules
from src.preprocessing import load_data, preprocess_data, vectorize_data
from src.train import train_logistic_regression, train_naive_bayes, save_models
from src.evaulate import evaluate_all_models
from src.predict import predict_category


def main():
    print("="*60)
    print("TEXT CLASSIFICATION - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    train_df, test_df = load_data()
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    train_df, test_df = preprocess_data(train_df, test_df)
    
    # Step 3: Vectorize data
    print("\n[Step 3] Vectorizing data...")
    vectorizer = TfidfVectorizer()
    X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
        train_df, test_df, vectorizer
    )
    
    # Step 4: Train models
    print("\n[Step 4] Training models...")
    clf = train_logistic_regression(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Step 5: Evaluate models
    print("\n[Step 5] Evaluating models...")
    results = evaluate_all_models(clf, nb_model, X_test, y_test, show_plots=True)
    
    # Step 6: Save models
    print("\n[Step 6] Saving models...")
    save_models(vectorizer, clf, nb_model)
    
    # Step 7: Test prediction
    print("\n[Step 7] Testing prediction function...")
    test_text = "ronaldo has scored a last minute goal"
    result = predict_category(test_text, vectorizer, clf, nb_model)
    print(f"\nFinal result: {result}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()