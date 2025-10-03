from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, model_name="Model"):
  
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation")
    print(f"{'='*50}")
    print(f'Accuracy: {accuracy:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, y_pred


def display_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
   
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(title)
    plt.show()


def evaluate_all_models(clf, nb_model, X_test, y_test, show_plots=True):
   
    # Evaluate Logistic Regression
    lr_accuracy, y_pred_lr = evaluate_model(clf, X_test, y_test, "Logistic Regression")
    if show_plots:
        display_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression")
    
    # Evaluate Naive Bayes
    nb_accuracy, y_pred_nb = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    if show_plots:
        display_confusion_matrix(y_test, y_pred_nb, "Confusion Matrix - Naive Bayes")
    
    results = {
        'logistic_regression': {
            'accuracy': lr_accuracy,
            'predictions': y_pred_lr
        },
        'naive_bayes': {
            'accuracy': nb_accuracy,
            'predictions': y_pred_nb
        }
    }
    
    return results
