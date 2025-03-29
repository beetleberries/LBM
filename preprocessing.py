import os
import mne
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

nested_folder = os.path.join("dataset", "1")
condition_files = ["Fatigue state.cnt"]

for condition in condition_files:
    file_path = os.path.join(nested_folder, condition)

    print(f"\nLooking for: {file_path}")
    if not os.path.isfile(file_path):
        print("File missing!")
        continue

    print("Trying to read EEG file...")
    try:
        raw = mne.io.read_raw_ant(file_path, preload=True)
        print("ANT Neuro file loaded.")
    except:
        print("ANT failed. Trying Neuroscan...")
        try:
            raw = mne.io.read_raw_cnt(file_path, preload=True)
            print("Neuroscan file loaded.")
        except Exception as e:
            print(f"Failed to load: {e}")
            continue

    print("Printing EEG info...")
    print(raw.info)

    duration = raw.n_times / raw.info['sfreq']
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Channels: {len(raw.ch_names)}")

    raw.plot(start=0, duration=5, n_channels=20, title=f"{condition} - EEG Preview", block=True)

