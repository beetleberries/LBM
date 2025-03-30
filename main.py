import os
import mne
import numpy as np

# load dataset
dataset_path = r"C:\Users\HYUN\capstone\LBM\dataset"

# save list
all_psds = []     # PSD data
all_labels = []   # 0 = Normal, 1 = Fatigue

# read dataset
for subject_id in range(1, 13):
    subject_path = os.path.join(dataset_path, str(subject_id))

    for state, label in [("Normal state.cnt", 0), ("Fatigue state.cnt", 1)]:
        file_path = os.path.join(subject_path, state)

        if not os.path.exists(file_path):
            print(f"no file: {file_path}")
            continue

        print(f"loading: {file_path}")

        try:
            # Raw EEG loading
            raw = mne.io.read_raw_cnt(file_path, preload=True)

            # referencing, filtering, resampling
            raw.set_eeg_reference('average')
            raw.filter(4., 30.)
            raw.resample(250)

            # ICA 
            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)

            # PSD + log 
            psd = raw.compute_psd(fmin=4., fmax=30., n_fft=512)
            psds, freqs = psd.get_data(return_freqs=True)
            log_psds = 10 * np.log10(psds)  # shape: (channels, freqs)

            # save result
            all_psds.append(log_psds)
            all_labels.append(label)

        except Exception as e:
            print(f" error: {e}")

# save as array
X = np.array(all_psds)       # shape: (samples, channels, freqs)
y = np.array(all_labels)     # shape: (samples,)

print("\n finished")
print("X shape:", X.shape)
print("y shape:", y.shape)

# save as file
np.save("X_psd.npy", X)
np.save("y_label.npy", y)
print("file saved: X_psd.npy, y_label.npy")
