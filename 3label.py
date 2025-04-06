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
beta_values = []
theta_beta_ratios = []

print("Preprocessing and calculating Theta/Beta ratios...")

for file_name in os.listdir(dataset_path):
    if file_name.endswith(".set"):
        file_path = os.path.join(dataset_path, file_name)
        print(f"Processing: {file_path}")
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
            beta_band = (freqs >= 13) & (freqs < 30)

            theta_power = log_psds[:, theta_band].mean()
            beta_power = log_psds[:, beta_band].mean()
            theta_beta = theta_power / beta_power

            theta_values.append(theta_power)
            beta_values.append(beta_power)
            theta_beta_ratios.append(theta_beta)
            all_psds.append(log_psds)

        except Exception as e:
            print(f"Error: {e}")

# === IQR 기반 threshold 계산 ===
ratios = np.array(theta_beta_ratios)
q1 = np.percentile(ratios, 25)
q3 = np.percentile(ratios, 75)
iqr = q3 - q1
drowsy_thresh = q3 + 0.5 * iqr
distracted_thresh = q1 - 0.5 * iqr

# === 레이블 할당: 0 = Drowsy, 1 = Normal, 2 = Distracted
for ratio in theta_beta_ratios:
    if ratio > drowsy_thresh:
        all_labels.append(0)
    elif ratio < distracted_thresh:
        all_labels.append(2)
    else:
        all_labels.append(1)

np.save(os.path.join(output_path, "X_psd_label3.npy"), np.array(all_psds))
np.save(os.path.join(output_path, "y_label_3class.npy"), np.array(all_labels))
print("Saved X_psd_label3.npy and y_label_3class.npy")

# === 시각화 ===
plt.figure(figsize=(8, 4))
plt.hist(ratios, bins=20, color='mediumorchid')
plt.axvline(drowsy_thresh, color='red', linestyle='--', label='Drowsy Threshold')
plt.axvline(distracted_thresh, color='blue', linestyle='--', label='Distracted Threshold')
plt.title("Theta/Beta Ratio Distribution")
plt.xlabel("Theta/Beta Ratio")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_path, "theta_beta_ratio_thresholds.png"))
plt.close()

# === 라벨 분포 시각화 ===
import pandas as pd
label_counts = pd.Series(all_labels).value_counts().sort_index()
label_names = ['Drowsy', 'Normal', 'Distracted']
label_counts.index = label_names
plt.figure(figsize=(6, 4))
bars = plt.bar(label_names, label_counts.values, color='slateblue')
for i, count in enumerate(label_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center')
plt.title("3-Class Label Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_path, "3class_label_distribution.png"))
plt.close()
print("Saved 3class_label_distribution.png")
