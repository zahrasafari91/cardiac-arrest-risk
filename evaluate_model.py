import os
from tensorflow.keras.models import load_model
from datasets import load_test_data, ECGSequence
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model_on_test_data(model_path, music_dir, icare_dir, batch_size=32):
    # Load combined test data
    X_test, y_test = load_test_data(music_dir, icare_dir)

    # Create test sequence
    test_seq = ECGSequence(X_test, y_test, batch_size=batch_size, is_training=False)

    # Load the trained model
    model = load_model(model_path, compile=True)

    # Evaluate the model
    print("Evaluating the model on test data...")
    test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_seq)
    print(f"Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, "
          f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, AUC: {test_auc:.4f}")

    # Confusion matrix
    y_pred = model.predict(test_seq)
    y_pred_labels = (y_pred > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred_labels)
    print(f"Confusion Matrix:\n{cm}")

    return {
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "auc": test_auc,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    test_music_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Test/Music"
    test_icare_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Test/I_Care"
    model_path = "./results/final_model.h5"

    test_results = evaluate_model_on_test_data(model_path, test_music_dir, test_icare_dir)
    print(test_results)
