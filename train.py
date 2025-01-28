#*********************** Train for new model (training from scratch) ***********************

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
# from tensorflow.keras.metrics import Precision, Recall, AUC
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.models import Model
# from model import load_pretrained_model
# from datasets import load_and_prepare_data
# import os
# import numpy as np
# import pickle
# import logging
# from tensorflow.keras.callbacks import CSVLogger

# # Configure logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# def add_dropout_to_model(model, dropout_rate=0.5):
#     """Add a dropout layer to the pre-trained model."""
#     x = model.layers[-2].output  # Get the output of the second-to-last layer
#     x = Dropout(dropout_rate)(x)  # Add a dropout layer
#     output = model.layers[-1](x)  # Connect to the final output layer
#     return Model(inputs=model.input, outputs=output)



# def ensure_directory_exists(filepath):
#     """Ensure the directory for a given file path exists."""
#     directory = os.path.dirname(filepath)
#     if not os.path.exists(directory):
#         os.makedirs(directory)


# def main():
#     # Paths
#     music_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/Music"
#     icare_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/I_Care"
#     weights_path = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/model/model.hdf5"
#     results_dir = "./results"

#     # Hyperparameters
#     learning_rate = 0.001
#     epochs = 50
#     batch_size = 64
#     dropout_rate = 0.5

#     # Create results directory
#     os.makedirs(results_dir, exist_ok=True)
#     logging.info(f"Results will be saved in: {results_dir}")

#     # Ensure CSV log directory exists
#     csv_log_path = os.path.join(results_dir, "training_log.csv")
#     ensure_directory_exists(csv_log_path)

#     # Load datasets
#     logging.info("Loading and preparing datasets...")
#     train_seq, val_seq, y_train, y_val = load_and_prepare_data(music_dir, icare_dir, val_split=0.1, batch_size=batch_size)

#     # Load the pre-trained model
#     if not os.path.exists(weights_path):
#         raise FileNotFoundError(f"Pre-trained weights not found at {weights_path}")

#     model = load_pretrained_model(weights_path, n_classes=1)  # For binary classification
#     model = add_dropout_to_model(model, dropout_rate=dropout_rate)  # Add dropout layer
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss="binary_crossentropy",
#         metrics=["accuracy", Precision(), Recall(), AUC()]
#     )
#     logging.info(f"Model compiled with learning rate {learning_rate} and dropout rate {dropout_rate}")

#     # Define callbacks
#     callbacks = [
#         ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
#         EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
#         ModelCheckpoint(os.path.join(results_dir, "best_model.h5"), save_best_only=True, verbose=1),
#         CSVLogger(os.path.join(results_dir, "training_log.csv"))
#     ]
#     # Class weights
#     # class_weight = {0: 1.0, 1: 3.0}
#     # class_weight = {0: 1.0, 1: 1.5}  


#     # Train the model
#     logging.info("Starting model training...")
#     history = model.fit(
#         train_seq,                    # Training sequence
#         validation_data=val_seq,      # Validation sequence
#         epochs=epochs,                # Number of epochs
#         callbacks=callbacks,          # Training callbacks
#         verbose=1,                   # Verbose output
#         # class_weight=class_weight
#     )

#     # Save the final trained model
#     final_model_path = os.path.join(results_dir, "final_model.h5")
#     model.save(final_model_path)
#     logging.info(f"Final model saved at {final_model_path}")

#     # Save training history
#     history_path = os.path.join(results_dir, "training_history.pkl")
#     with open(history_path, "wb") as f:
#         pickle.dump(history.history, f)
#     logging.info(f"Training history saved at {history_path}")

#     # Evaluate the model on validation data
#     logging.info("Evaluating the model on validation data...")
#     val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_seq)
#     logging.info(f"Validation Metrics - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, "
#                  f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")

#     # Save validation metrics
#     metrics_path = os.path.join(results_dir, "validation_metrics.txt")
#     with open(metrics_path, "w") as f:
#         f.write(f"Validation Metrics:\n")
#         f.write(f" - Loss: {val_loss:.4f}\n")
#         f.write(f" - Accuracy: {val_acc:.4f}\n")
#         f.write(f" - Precision: {val_prec:.4f}\n")
#         f.write(f" - Recall: {val_rec:.4f}\n")
#         f.write(f" - AUC: {val_auc:.4f}\n")
#     logging.info(f"Validation metrics saved at {metrics_path}")

# if __name__ == "__main__":
#     main()


#*********************** Train for transfer learning ***********************
import os
import pickle
import logging
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
)
from tensorflow.keras.metrics import Precision, Recall, AUC

from model import load_pretrained_model  
from datasets import load_and_prepare_data  


#Utility functions           

def plot_metrics(history, results_dir):
    """Plot training and validation metrics."""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for metric in metrics:
        if f'val_{metric}' in history.history:
            plt.figure()
            plt.plot(history.history[metric], label=f'Training {metric}')
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.title(f'Training and Validation {metric.capitalize()}')
            plt.legend()
            plt.savefig(os.path.join(results_dir, f"{metric}_plot.png"))
            plt.close()

def ensure_directory_exists(filepath):
    """Ensure the directory for a given file path exists."""
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

#Main Training Script        

def main():
    # ----------------- Paths -----------------
    music_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/Music"
    icare_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/I_Care"
    old_weights_path = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/model/model.hdf5"
    results_dir = "./results"

    # ----------------- Hyperparams -----------------
    learning_rate = 0.001  
    epochs = 100
    batch_size = 64

    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(f"Results will be saved in: {results_dir}")

    # ----------------- Load Data -----------------
    logging.info("Loading and preparing datasets...")
    train_seq, val_seq, y_train, y_val = load_and_prepare_data(
        music_dir, icare_dir, 
        val_split=0.1, 
        batch_size=batch_size
    )

    # ----------------- Load + Transfer Model -----------------
    if not os.path.exists(old_weights_path):
        raise FileNotFoundError(f"Pre-trained weights not found at {old_weights_path}")

    logging.info("Loading and setting up transfer model...")
    # freeze_until=5 means freeze the first 5 layers of the new model
    model = load_pretrained_model(old_weights_path, n_classes=1, freeze_until=5)

    # ----------------- Compile -----------------
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc")
        ]
    )
    logging.info(f"Model compiled with learning rate {learning_rate}")

    # ----------------- Callbacks -----------------

    csv_log_path = os.path.join(results_dir, "training_log.csv")
    if not os.path.exists(os.path.dirname(csv_log_path)):
        os.makedirs(os.path.dirname(csv_log_path))

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(results_dir, "best_model.h5"), save_best_only=True, verbose=1),
        CSVLogger(csv_log_path)
    ]

    # ----------------- Class Weights (Example) -----------------
    # class_weight = {0: 1.0, 1: 3.0}  # adjust as needed for imbalance

    # ----------------- Train -----------------
    logging.info("Starting model fine-tuning...")
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        # class_weight=class_weight,
        verbose=1
    )

    # ----------------- Save Final Model -----------------
    final_model_path = os.path.join(results_dir, "final_model.h5")
    model.save(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")

    # ----------------- Save History -----------------
    history_path = os.path.join(results_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    logging.info(f"Training history saved to {history_path}")

    # ----------------- Plot Metrics -----------------
    plot_metrics(history, results_dir)

    # ----------------- Evaluation -----------------
    logging.info("Evaluating the model on validation data...")
    val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(val_seq, verbose=1)
    logging.info(f"Validation Metrics - Loss: {val_loss:.4f}, "
                 f"Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, "
                 f"Recall: {val_rec:.4f}, AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()
