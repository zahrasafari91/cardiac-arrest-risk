import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
import os

# Define Functions for Preprocessing
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=1000, order=4):
    """Apply a bandpass filter to preserve ECG features."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def resample_ecg(data, original_fs, target_fs):
    """Resample data to the target sampling rate."""
    return resample_poly(data, target_fs, original_fs, axis=0)

def z_score_normalize(signal):
    """Apply z-score normalization to a signal."""
    mean = np.mean(signal, axis=0)
    std = np.std(signal, axis=0)
    return (signal - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

def pad_signal(data, target_length):
    """Pad the signal to a fixed target length."""
    padded_data = np.zeros((target_length, data.shape[1]))
    padded_data[:len(data), :] = data
    return padded_data

def load_ecg_data(dat_file, hea_file):
    """Load ECG data from .dat file using metadata from .hea file."""
    with open(hea_file, 'r') as file:
        lines = file.readlines()
        sample_rate = int(lines[0].split()[2])  # Extract sample rate
        num_channels = int(lines[0].split()[1])  # Extract number of channels
        duration = int(lines[0].split()[3]) / sample_rate  # Calculate signal duration in seconds

    raw_data = np.fromfile(dat_file, dtype=np.int16)
    ecg_data = raw_data.reshape(-1, num_channels)  # Reshape based on channels
    return ecg_data, sample_rate, duration

def preprocess_ecg(data, original_sample_rate, target_sample_rate=400, segment_length=4096):
    """Preprocess ECG data: resample, filter, normalize, and ensure fixed-length segments."""
    # Resample to target sample rate
    resampled_data = resample_ecg(data, original_sample_rate, target_sample_rate)

    # Filter each channel
    filtered_data = np.zeros_like(resampled_data)
    for channel in range(resampled_data.shape[1]):
        filtered_data[:, channel] = bandpass_filter(resampled_data[:, channel], 
                                                    lowcut=0.5, highcut=50.0, 
                                                    fs=target_sample_rate)

    # Handle varying lengths: truncate or pad to segment length
    total_samples = filtered_data.shape[0]
    if total_samples >= segment_length:
        truncated_data = filtered_data[:segment_length, :]
    else:
        truncated_data = pad_signal(filtered_data, segment_length)

    # Normalize the fixed-length data
    normalized_data = z_score_normalize(truncated_data)

    return normalized_data, target_sample_rate, segment_length

# Set Paths and Parameters
input_folder = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/dataset/High-resolution_ECG"
output_folder = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/ecg_dataset"
csv_file_path = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/ecg_patient_data.csv"
os.makedirs(output_folder, exist_ok=True)

# Load Patient Data
patient_data = pd.read_csv(csv_file_path)

# Process Each File
def process_files():
    for _, row in patient_data.iterrows():
        patient_id = row['Patient_ID']
        ecg_file = row['ECG_File']
        cardiac_arrest_risk = row['Cardiac_arrest_Risk']  # Extract Cardiac_arrest_Risk
        dat_file = os.path.join(input_folder, ecg_file)
        hea_file = dat_file.replace('.dat', '.hea')

        if os.path.exists(dat_file) and os.path.exists(hea_file):
            try:
                # Load ECG data
                ecg_data, sample_rate, duration = load_ecg_data(dat_file, hea_file)

                # Preprocess the ECG data
                preprocessed_data, processed_rate, segment_length = preprocess_ecg(
                    ecg_data, sample_rate, target_sample_rate=400, segment_length=4096)

                # Save the preprocessed data to a .npy file
                output_file = os.path.join(output_folder, f"{patient_id}.npy")
                np.save(output_file, {
                    'patient_id': patient_id,
                    'ecg_segments': preprocessed_data,
                    'processed_sample_rate': processed_rate,
                    'processed_signal_length': segment_length,
                    'cardiac_arrest_risk': cardiac_arrest_risk
                })

                print(f"Processed patient {patient_id}")

            except Exception as e:
                print(f"Error processing {ecg_file}: {e}")

    print("Processing complete. Processed files saved in", output_folder)


process_files()
