import os
import mne
import numpy as np

all_psds = []
all_labels = []

for i in range(1, 13):
    subject_dir = f"data/dataset/{i}"
    inner_dirs = os.listdir(subject_dir)
    if len(inner_dirs) == 1:
        subject_dir = os.path.join(subject_dir, inner_dirs[0])

    for state, label in [("Normal state.cnt", 0), ("Fatigue state.cnt", 1)]:
        file_path = os.path.join(subject_dir, state)
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        print(f"Processing: {file_path}")
        try:
            raw = mne.io.read_raw_ant(file_path, preload=True)
            raw.set_eeg_reference('average')
            raw.filter(4., 30.)
            raw.resample(250)

            ica = mne.preprocessing.ICA(n_components=20, random_state=42)
            ica.fit(raw)
            raw = ica.apply(raw)

            psd = raw.compute_psd(fmin=4., fmax=30., n_fft=512)
            psds, _ = psd.get_data(return_freqs=True)
            log_psds = 10 * np.log10(psds)

            all_psds.append(log_psds)
            all_labels.append(label)
        except Exception as e:
            print(f"Error: {e}")

X = np.array(all_psds)
y = np.array(all_labels)

np.save("data/X_psd.npy", X)
np.save("data/y_label.npy", y)
print(f"Saved: data/X_psd.npy ({X.shape}), data/y_label.npy ({y.shape})")
