import os
import mne
import numpy as np

# 1
dataset_path = r"C:\Users\HYUN\capstone\LBM\dataset"

# 2
all_psds = []
all_labels = [] 

# 3
def classify_eeg_state(log_psds, freqs):
    theta_band = (freqs >= 4) & (freqs < 8)
    alpha_band = (freqs >= 8) & (freqs < 13)
    beta_band = (freqs >= 13) & (freqs < 30)

    theta_power = log_psds[:, theta_band].mean()
    alpha_power = log_psds[:, alpha_band].mean()
    beta_power  = log_psds[:, beta_band].mean()

    beta_theta = beta_power / theta_power
    alpha_beta = alpha_power / beta_power

    if beta_theta < 1.0:
        return 0  # Fatigue
    elif beta_theta > 2.0:
        return 1  # Attentive
    elif alpha_beta > 1.2:  #reasoning for this (reference) why this is used in the research, value could have worked on single so the accuracy might not work, compare the accuracy with the source, why difference, the source could lack generalization
        return 2  # Relaxed
    elif theta_power > beta_power:
        return 3  # Drowsy
    else:
        return 4  # Distracted

# 4
for subject_id in range(1, 13):
    subject_path = os.path.join(dataset_path, str(subject_id))

    for state_file in ["Normal state.cnt", "Fatigue state.cnt"]:
        file_path = os.path.join(subject_path, state_file)

        if not os.path.exists(file_path):
            print(f"no file: {file_path}")
            continue

        print(f"processing: {file_path}")

        try:
            raw = mne.io.read_raw_cnt(file_path, preload=True)
            raw.set_eeg_reference('average')
            raw.filter(4., 30.)
            raw.resample(250)

            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)

            psd = raw.compute_psd(fmin=4., fmax=30., n_fft=512)
            psds, freqs = psd.get_data(return_freqs=True)
            log_psds = 10 * np.log10(psds)

            label = classify_eeg_state(log_psds, freqs)
            all_psds.append(log_psds)
            all_labels.append(label)

        except Exception as e:
            print(f"error: {e}")

# 5
X = np.array(all_psds)
y = np.array(all_labels)
np.save("X_psd_multi.npy", X)
np.save("y_label_multi.npy", y)

print("saved: X_psd_multi.npy, y_label_multi.npy")
