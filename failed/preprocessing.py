import os
import mne
import numpy as np

dataset_path = r"C:\Users\HYUN\capstone\LBM\dataset"
all_psds = []
all_labels = []

for subject_id in range(1, 13):
    subject_path = os.path.join(dataset_path, str(subject_id))

    for state, label in [("Normal state.cnt", 0), ("Fatigue state.cnt", 1)]:
        file_path = os.path.join(subject_path, state)

        if not os.path.exists(file_path):
            print(f"❌ no file: {file_path}")
            continue

        try:
            print(f"📥 loading: {file_path}")
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

            all_psds.append(log_psds)
            all_labels.append(label)

        except Exception as e:
            print(f"❗ error: {e}")

X = np.array(all_psds)  # shape: (samples, channels, freqs)
y = np.array(all_labels)

np.save("X_psd_binary.npy", X)
np.save("y_label_binary.npy", y)

print(f"✅ saved: X_psd_binary.npy, y_label_binary.npy")
