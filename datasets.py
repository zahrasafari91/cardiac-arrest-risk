import os
import numpy as np
import logging
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ECGSequence(Sequence):
    """Sequence class for batch loading of ECG data."""
    def __init__(self, X_combined, y, batch_size=32, is_training=False):
        self.X_combined = X_combined
        self.y = y
        self.batch_size = batch_size
        self.is_training = is_training

    def __len__(self) -> int:
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx: int):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.y))
        return self.X_combined[start:end], self.y[start:end]


def normalize_data(data):
    """Normalize data to have zero mean and unit variance."""
    return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)


def load_and_prepare_data(music_dir: str, icare_dir: str, val_split: float = 0.1, batch_size: int = 32):
    music_data, music_labels = [], []
    icare_data, icare_labels = [], []
    skipped_files = []

    # Load MUSIC data
    for file_name in os.listdir(music_dir):
        if file_name.endswith(".npy"):
            try:
                file_data = np.load(os.path.join(music_dir, file_name), allow_pickle=True).item()
                music_data.append(file_data["ecg_segments"])  # Shape: (4096, 3)
                music_labels.append(file_data["cardiac_arrest_risk"])
            except Exception as e:
                logging.error(f"Error loading MUSIC file {file_name}: {e}")
                skipped_files.append(file_name)

    # Load I-Care data
    for file_name in os.listdir(icare_dir):
        if file_name.endswith(".npy"):
            try:
                file_data = np.load(os.path.join(icare_dir, file_name), allow_pickle=True).item()
                ecg_segments = file_data["ecg_segments"]

                # Skip files with 2-lead signals
                if ecg_segments.shape[-1] == 2:
                    skipped_files.append(file_name)
                    continue

                # Zero-pad 1-lead signals to 3 leads
                if ecg_segments.shape[-1] == 1:
                    ecg_segments = np.concatenate([ecg_segments, np.zeros((ecg_segments.shape[0], 2))], axis=-1)

                icare_data.append(ecg_segments)
                icare_labels.append(1)  # All I-Care samples are "at-risk"
            except Exception as e:
                logging.error(f"Error loading I-Care file {file_name}: {e}")
                skipped_files.append(file_name)

    # Convert to numpy arrays
    music_data = np.array(music_data)  # Shape: (num_samples_music, 4096, 3)
    icare_data = np.array(icare_data)  # Shape: (num_samples_icare, 4096, 3)
    music_labels = np.array(music_labels)
    icare_labels = np.array(icare_labels)

    # Pad channels to match (4096, 6)
    music_data_padded = np.concatenate((music_data, np.zeros_like(music_data)), axis=-1)  # MUSIC (4096, 6)
    icare_data_padded = np.concatenate((np.zeros_like(icare_data), icare_data), axis=-1)  # I-Care (4096, 6)

    # Combine datasets along the sample axis
    X_combined = np.concatenate((music_data_padded, icare_data_padded), axis=0)
    y_combined = np.concatenate((music_labels, icare_labels), axis=0)


    # Log combined dataset information
    logging.info(f"Combined Dataset Label Distribution: {np.bincount(y_combined)}")
    logging.info(f" - No-risk (0): {np.bincount(y_combined)[0]}")
    logging.info(f" - At-risk (1): {np.bincount(y_combined)[1]}")

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_combined, y_combined, test_size=val_split, stratify=y_combined, random_state=42)

    # Log training and validation set distribution
    logging.info(f"Training Set Label Distribution: {np.bincount(y_train)}")
    logging.info(f" - No-risk (0): {np.bincount(y_train)[0]}")
    logging.info(f" - At-risk (1): {np.bincount(y_train)[1]}")
    logging.info(f"Validation Set Label Distribution: {np.bincount(y_val)}")
    logging.info(f" - No-risk (0): {np.bincount(y_val)[0]}")
    logging.info(f" - At-risk (1): {np.bincount(y_val)[1]}")
    logging.info(f"Skipped Files: {len(skipped_files)}")

    # Create sequences
    train_seq = ECGSequence(X_train, y_train, batch_size=batch_size, is_training=True)
    val_seq = ECGSequence(X_val, y_val, batch_size=batch_size, is_training=False)

    return train_seq, val_seq, y_train, y_val

def load_test_data(music_dir: str, icare_dir: str):
    """
    Loads and processes test data from MUSIC and I-Care datasets.

    Args:
        music_dir (str): Directory containing MUSIC dataset files.
        icare_dir (str): Directory containing I-Care dataset files.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed ECG test data (X_combined) and labels (y_combined).
    """
    music_data, music_labels = [], []
    icare_data, icare_labels = [], []
    skipped_files = []

    # Load MUSIC data
    for file_name in os.listdir(music_dir):
        if file_name.endswith(".npy"):
            try:
                file_data = np.load(os.path.join(music_dir, file_name), allow_pickle=True).item()
                music_data.append(file_data["ecg_segments"])  # Shape: (4096, 3)
                music_labels.append(file_data["cardiac_arrest_risk"])
            except Exception as e:
                logging.error(f"Error loading MUSIC file {file_name}: {e}")
                skipped_files.append(file_name)

    # Load I-Care data
    for file_name in os.listdir(icare_dir):
        if file_name.endswith(".npy"):
            try:
                file_data = np.load(os.path.join(icare_dir, file_name), allow_pickle=True).item()
                ecg_segments = file_data["ecg_segments"]

                # Skip invalid signals
                if ecg_segments.shape[-1] == 2:
                    skipped_files.append(file_name)
                    continue

                # Zero-pad 1-lead signals to 3 leads
                if ecg_segments.shape[-1] == 1:
                    ecg_segments = np.concatenate([ecg_segments, np.zeros((ecg_segments.shape[0], 2))], axis=-1)

                icare_data.append(ecg_segments)
                icare_labels.append(1)  # I-Care samples are all "at-risk"
            except Exception as e:
                logging.error(f"Error loading I-Care file {file_name}: {e}")
                skipped_files.append(file_name)

    # Convert to numpy arrays
    music_data = np.array(music_data)  # Shape: (num_samples_music, 4096, 3)
    icare_data = np.array(icare_data)  # Shape: (num_samples_icare, 4096, 3)
    music_labels = np.array(music_labels)
    icare_labels = np.array(icare_labels)

    # Pad channels to match (4096, 6)
    music_data_padded = np.concatenate((music_data, np.zeros_like(music_data)), axis=-1)  # MUSIC (4096, 6)
    icare_data_padded = np.concatenate((np.zeros_like(icare_data), icare_data), axis=-1)  # I-Care (4096, 6)

    # Combine datasets
    X_combined = np.concatenate((music_data_padded, icare_data_padded), axis=0)
    y_combined = np.concatenate((music_labels, icare_labels), axis=0)

    # Log combined test dataset information
    logging.info(f"Test Data Label Distribution: {np.bincount(y_combined)}")
    logging.info(f" - No-risk (0): {np.bincount(y_combined)[0]}")
    logging.info(f" - At-risk (1): {np.bincount(y_combined)[1]}")
    logging.info(f"Skipped Files in Test Data: {len(skipped_files)}")

    return X_combined, y_combined



if __name__ == "__main__":
    music_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/Music"
    icare_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/I_Care"

    train_seq, val_seq, y_train, y_val = load_and_prepare_data(music_dir, icare_dir, val_split=0.1)
    logging.info(f"Training Batches: {len(train_seq)}, Validation Batches: {len(val_seq)}")

    for i, (X_batch, y_batch) in enumerate(train_seq):
        logging.info(f"Batch {i+1} Shapes: X={X_batch.shape}, y={y_batch.shape}")
        if i == 1:  # Only log first two batches
            break


    test_music_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Test/Music"
    test_icare_dir = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Test/I_Care"

    X_test, y_test = load_test_data(test_music_dir, test_icare_dir)
    logging.info(f"Test Data Shape: {X_test.shape}, Labels Shape: {y_test.shape}")

