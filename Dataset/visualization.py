import numpy as np
import matplotlib.pyplot as plt

def plot_ecg_signal(file_data, output_dir):
    """
    Plot each lead of an ECG signal in separate subplots within a single file.

    Args:
        file_data (dict): Loaded .npy file data.
        output_dir (str): Directory to save the visualization.
    """
    patient_id = file_data['patient_id']
    ecg_segments = file_data['ecg_segments']  # Shape: (num_samples, num_leads)
    sample_rate = file_data['processed_sample_rate']
    signal_length = file_data['processed_signal_length']
    label = file_data['cardiac_arrest_risk']

    # Create time axis
    time_axis = np.linspace(0, signal_length, ecg_segments.shape[0])

    # Create subplots
    num_leads = ecg_segments.shape[1]
    fig, axes = plt.subplots(num_leads, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"ECG Signal for Patient {patient_id} (Label: {label})", fontsize=16)

    # Plot each lead in a separate subplot
    for lead_idx, ax in enumerate(axes if num_leads > 1 else [axes]):
        ax.plot(time_axis, ecg_segments[:, lead_idx], label=f"Lead {lead_idx + 1}", color='black')
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title(f"Lead {lead_idx + 1}", fontsize=12)
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    # Add a shared X-axis label
    plt.xlabel("Time (s)", fontsize=12)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
    output_path = f"{output_dir}/ecg_patient_{patient_id}_label_{label}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"ECG visualization saved to {output_path}")


if __name__ == "__main__":
    file_path = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/Dataset/Train/I_Care/0284.npy"
    output_dir = "ecg_visualizations"

    # Load the file
    file_data = np.load(file_path, allow_pickle=True).item()

    # Plot and save visualization
    plot_ecg_signal(file_data, output_dir)

