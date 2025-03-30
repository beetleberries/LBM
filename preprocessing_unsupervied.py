import os
import mne
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\HYUN\capstone\LBM\dataset_unsupervised"
output_path = r"C:\Users\HYUN\capstone\LBM"
os.makedirs(output_path, exist_ok=True)

all_psds = []
all_labels = []

theta_values = []
alpha_values = []
beta_values = []

beta_theta_values = []
alpha_beta_values = []
theta_beta_ratios = []

print("Preprocessing and ratio calculation...")
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".set"):
        file_path = os.path.join(dataset_path, file_name)
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            raw.set_eeg_reference('average')
            raw.filter(4., 30.)
            raw.resample(250)

            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)

            psd = raw.compute_psd(fmin=4., fmax=30., n_fft=512)
            psds, freqs = psd.get_data(return_freqs=True)
            log_psds = 10 * np.log10(psds)

            theta_band = (freqs >= 4) & (freqs < 8)
            alpha_band = (freqs >= 8) & (freqs < 13)
            beta_band  = (freqs >= 13) & (freqs < 30)

            theta_power = log_psds[:, theta_band].mean()
            alpha_power = log_psds[:, alpha_band].mean()
            beta_power  = log_psds[:, beta_band].mean()

            theta_values.append(theta_power)
            alpha_values.append(alpha_power)
            beta_values.append(beta_power)

            beta_theta_values.append(beta_power / theta_power)
            alpha_beta_values.append(alpha_power / beta_power)
            theta_beta_ratios.append(theta_power / beta_power)

        except Exception as e:
            print(f" Error reading {file_name}: {e}")

theta_beta_ratios = np.array(theta_beta_ratios)
drowsy_threshold = float(np.median(theta_beta_ratios))  
print(f"drowsy threshold (Theta/Beta): {drowsy_threshold:.3f}")

def classify_eeg_state(theta_power, alpha_power, beta_power):
    beta_theta = beta_power / theta_power
    alpha_beta = alpha_power / beta_power
    theta_beta = theta_power / beta_power

    if beta_theta < 1.0:
        return 0  # Fatigue
    elif beta_theta > 2.0:
        return 1  # Attentive
    elif alpha_beta > 1.2:
        return 2  # Relaxed
    elif theta_beta > drowsy_threshold:
        return 3  # Drowsy
    else:
        return 4  # Distracted

print("📥 Reprocessing with labeling...")
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".set"):
        file_path = os.path.join(dataset_path, file_name)
        try:
            raw = mne.io.read_raw_eeglab(file_path, preload=True)
            raw.set_eeg_reference('average')
            raw.filter(4., 30.)
            raw.resample(250)

            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)

            psd = raw.compute_psd(fmin=4., fmax=30., n_fft=512)
            psds, freqs = psd.get_data(return_freqs=True)
            log_psds = 10 * np.log10(psds)

            theta_band = (freqs >= 4) & (freqs < 8)
            alpha_band = (freqs >= 8) & (freqs < 13)
            beta_band  = (freqs >= 13) & (freqs < 30)

            theta_power = log_psds[:, theta_band].mean()
            alpha_power = log_psds[:, alpha_band].mean()
            beta_power  = log_psds[:, beta_band].mean()

            label = classify_eeg_state(theta_power, alpha_power, beta_power)

            all_psds.append(log_psds)
            all_labels.append(label)

            print(f"{file_name} → Label: {label}")

        except Exception as e:
            print(f"Error in {file_name}: {e}")

np.save(os.path.join(output_path, "X_psd_multi.npy"), np.array(all_psds))
np.save(os.path.join(output_path, "y_label_multi.npy"), np.array(all_labels))
print("Saved X_psd_multi.npy, y_label_multi.npy")

def save_hist(data, title, filename, color='skyblue'):
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=20, color=color)
    plt.title(title)
    plt.xlabel("Ratio" if "Ratio" in title else "Power (log)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

save_hist(beta_theta_values, "Beta/Theta Ratio", "beta_theta_distribution.png", 'cornflowerblue')
save_hist(alpha_beta_values, "Alpha/Beta Ratio", "alpha_beta_distribution.png", 'lightgreen')
save_hist(theta_beta_ratios, "Theta/Beta Ratio", "theta_beta_ratio_distribution.png", 'orchid')

plt.figure(figsize=(8, 4))
plt.hist(theta_values, bins=20, alpha=0.7, label='Theta')
plt.hist(beta_values, bins=20, alpha=0.7, label='Beta')
plt.title("Theta vs Beta Power")
plt.xlabel("Power (log)")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "theta_beta_power_distribution.png"))
plt.close()

print(" All plots saved.")
print(f"Theta/Beta > {drowsy_threshold:.3f}")
