import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample_poly

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
input_folder = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/icare_dataset"
output_folder = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/icare_processed"
os.makedirs(output_folder, exist_ok=True)

# Process Each Patient
def process_icare_files():
    for root, dirs, files in os.walk(input_folder):
        for dir_name in dirs:
            patient_folder = os.path.join(root, dir_name)
            patient_ecgs = []

            for file in os.listdir(patient_folder):
                if file.endswith(".mat"):
                    mat_file = os.path.join(patient_folder, file)
                    hea_file = mat_file.replace(".mat", ".hea")

                    if os.path.exists(mat_file) and os.path.exists(hea_file):
                        try:
                            # Load ECG data
                            mat_data = loadmat(mat_file)
                            ecg_data = mat_data['val'].T  # Transpose to shape (samples, channels)
                            sample_rate = 500  # Default sample rate for I-CARE dataset

                            # Append data to the patient's list
                            patient_ecgs.append(ecg_data)

                        except Exception as e:
                            print(f"Error processing {mat_file}: {e}")

            if patient_ecgs:
                try:
                    # Combine all ECGs for the patient
                    combined_ecg = np.concatenate(patient_ecgs, axis=0)

                    # Preprocess the ECG data
                    preprocessed_data, processed_rate, segment_length = preprocess_ecg(
                        combined_ecg, original_sample_rate=sample_rate, target_sample_rate=400, segment_length=4096)

                    # Save the preprocessed data to a .npy file
                    output_file = os.path.join(output_folder, f"{dir_name}.npy")
                    np.save(output_file, {
                        'patient_id': dir_name,
                        'ecg_segments': preprocessed_data,
                        'processed_sample_rate': processed_rate,
                        'processed_signal_length': segment_length,
                        'cardiac_arrest_risk': 1  # All patients have cardiac arrest risk
                    })

                    print(f"Processed patient {dir_name}")

                except Exception as e:
                    print(f"Error processing patient {dir_name}: {e}")

process_icare_files()
