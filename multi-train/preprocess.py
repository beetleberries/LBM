import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from mne.preprocessing import ICA # Import ICA
import glob # To find files
import gc # For garbage collection

# --- Configuration ---
# <<< SET THESE >>>
DATA_DIR = 'D:/EEG/'      # Directory containing all your .set files
OUTPUT_DIR = 'D:/EEG/processed' # Directory to SAVE processed data
OUTPUT_FILENAME_BASE = 'combined_s02' # Base name for saved combined files

# --- Preprocessing Parameters ---
FILTER_LFREQ = 1.0
FILTER_HFREQ = 40.0
NOTCH_FREQ = 60.0 # Use 50.0 for Europe/other regions
ICA_N_COMPONENTS = None
ICA_RANDOM_STATE = 97
ICA_METHOD = 'fastica'
AUTO_DETECT_EOG = True

# --- Event & Epoching Parameters ---
DEVIATION_CODES = [251, 252]
RESPONSE_CODE = 253
TMIN = -0.5
TMAX = 1.5
BASELINE = (None, 0)
RT_FILTER_THRESHOLD = 0.1

# --- Data Saving ---
# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
COMBINED_EPOCHS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_epochs_ica_std.npy')
COMBINED_LABELS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_labels.npy')

# --- Find Input Files ---
set_files = glob.glob(os.path.join(DATA_DIR, '*.set'))
if not set_files:
    print(f"Error: No .set files found in directory: {DATA_DIR}")
    exit()
print(f"Found {len(set_files)} .set files to process in {DATA_DIR}")

# --- Lists to Collect Data from All Files ---
all_epochs_data_list = []
all_labels_list = []
total_valid_epochs_processed = 0

# --- Main Loop Over Files ---
for i, file_path in enumerate(set_files):
    print(f"\n--- Processing File {i+1}/{len(set_files)}: {os.path.basename(file_path)} ---")
    try:
        # --- 1. Load Data ---
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

        # --- 2. Handle Channel Info & Montage ---
        print("   Setting Channel Types & Montage...")
        non_eeg_channels = [ch for ch in raw.ch_names if 'position' in ch.lower() or 'vehicle' in ch.lower()]
        if non_eeg_channels: raw.set_channel_types({ch: 'misc' for ch in non_eeg_channels})
        possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
        found_possible_refs = [ch for ch in raw.ch_names if any(ref_name.lower() == ch.lower() for ref_name in possible_ref_names)]
        if found_possible_refs: raw.set_channel_types({ch: 'misc' for ch in found_possible_refs})
        try:
            raw.set_montage('standard_1020', on_missing='warn', match_case=False, verbose=False)
        except ValueError: pass # Ignore montage setting error if channels don't match

        # --- 3. Filtering ---
        print(f"   Filtering ({FILTER_LFREQ}-{FILTER_HFREQ} Hz, Notch: {NOTCH_FREQ} Hz)...")
        raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, picks=['eeg'], fir_design='firwin', verbose=False)
        if NOTCH_FREQ: raw.notch_filter(freqs=NOTCH_FREQ, picks=['eeg'], fir_design='firwin', verbose=False)

        # --- 4. ICA Artifact Removal ---
        print("   Performing ICA...")
        raw_filtered = raw.copy() # Use a copy for ICA fitting
        ica = ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE, max_iter='auto')
        try:
            ica.fit(raw_filtered, picks='eeg', verbose=False) # Fit only on EEG channels
        except ValueError as e_ica_fit:
             print(f"   WARNING: ICA fit failed for this file ({e_ica_fit}). Skipping ICA application.")
             raw_cleaned = raw # Use unfiltered data if ICA fit fails
        else:
            print("     Identifying artifact components...")
            ica.exclude = []
            if AUTO_DETECT_EOG:
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw_filtered, ch_name=None, threshold=3.0, verbose=False)
                    if eog_indices:
                        print(f"       Auto-detected EOG components: {eog_indices}")
                        ica.exclude.extend(eog_indices)
                except Exception: pass # Ignore if detection fails

            # --- Add Manual Exclusions Here If Needed ---
            # manual_exclude_this_file = [] # Determine manually if needed
            # ica.exclude.extend(manual_exclude_this_file)
            # ica.exclude = sorted(list(set(ica.exclude)))
            # ------------------------------------------

            if ica.exclude: print(f"     Excluding components: {ica.exclude}")
            else: print("     No components marked for exclusion.")
            print("     Applying ICA...")
            # Apply ICA modifying the original 'raw' object for this file
            ica.apply(raw, exclude=ica.exclude)
            raw_cleaned = raw # Reference the modified raw

        del raw_filtered # Free memory from the copy used for fitting
        gc.collect()

        # --- 5. Extract Relevant Event Information ---
        print("   Extracting events...")
        if not raw_cleaned.annotations:
             print("   WARNING: No annotations found. Skipping file.")
             continue # Go to next file

        event_list = []
        for ann in raw_cleaned.annotations:
            # (Event extraction logic remains the same)
            event_time_sec = ann['onset']
            event_desc = ann['description']
            try:
                event_code = int(float(event_desc))
                if event_code in DEVIATION_CODES or event_code == RESPONSE_CODE:
                    event_list.append({'time_sec': event_time_sec, 'code': event_code})
            except (ValueError, TypeError): pass

        if not event_list:
            print(f"   WARNING: No relevant events ({DEVIATION_CODES} or {RESPONSE_CODE}) extracted. Skipping file.")
            continue

        # --- 6. Organize Events & Calculate RT ---
        events_df = pd.DataFrame(event_list)
        events_df = events_df.sort_values(by='time_sec').reset_index(drop=True)
        events_df['sample'] = (events_df['time_sec'] * raw_cleaned.info['sfreq']).round().astype(int)

        # (RT calculation logic remains the same)
        trials = []
        last_deviation_event = None
        for index, event in events_df.iterrows():
            if event['code'] in DEVIATION_CODES: last_deviation_event = event
            elif event['code'] == RESPONSE_CODE:
                if last_deviation_event is not None:
                    local_rt_sec = event['time_sec'] - last_deviation_event['time_sec']
                    if local_rt_sec >= 0: trials.append({ # ... (rest of dict as before)
                        'dev_code': last_deviation_event['code'], 'dev_time': last_deviation_event['time_sec'], 'dev_sample': last_deviation_event['sample'],
                        'response_code': event['code'], 'response_time': event['time_sec'], 'response_sample': event['sample'],
                        'local_rt_sec': local_rt_sec})
                    last_deviation_event = None
        trials_df = pd.DataFrame(trials)

        # --- 7. Filter Trials by Reaction Time ---
        initial_trial_count = len(trials_df)
        trials_df = trials_df[trials_df['local_rt_sec'] >= RT_FILTER_THRESHOLD].copy()
        if initial_trial_count > 0:
            removed_count = initial_trial_count - len(trials_df)
            print(f"   RT Filtering: Kept {len(trials_df)}/{initial_trial_count} trials (removed {removed_count}).")

        if trials_df.empty:
            print("   WARNING: No valid trials remaining after RT filtering. Skipping file.")
            continue

        # --- 8. Calculate Vigilance Labels ---
        # (Labeling logic remains the same)
        local_rts_filtered = trials_df['local_rt_sec'].values
        if len(local_rts_filtered) < 5: alert_rt_baseline = np.median(local_rts_filtered) if len(local_rts_filtered)>0 else 0.5 # Handle empty case
        else: alert_rt_baseline = np.percentile(local_rts_filtered, 5)
        alert_threshold = 1.5 * alert_rt_baseline
        drowsy_threshold = 2.5 * alert_rt_baseline
        labels = []
        for rt in trials_df['local_rt_sec']:
            if rt < alert_threshold: labels.append('alert')
            elif rt > drowsy_threshold: labels.append('drowsy')
            else: labels.append('transition')
        trials_df['vigilance_label'] = labels
        print(f"   Vigilance counts: {dict(trials_df['vigilance_label'].value_counts())}")


        # --- 9. Create MNE Event Array & Epochs ---
        print("   Creating epochs...")
        assigned_labels = trials_df['vigilance_label'].unique()
        vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
        epoch_event_id_dict = {label: vigilance_event_ids[label] for label in assigned_labels if label in vigilance_event_ids}
        if not epoch_event_id_dict:
             print("   WARNING: No trials classified into known states. Skipping file.")
             continue
        mne_events_list = []
        for index, trial in trials_df.iterrows():
             if trial['vigilance_label'] in epoch_event_id_dict:
                 mne_events_list.append([trial['response_sample'], 0, epoch_event_id_dict[trial['vigilance_label']]])
        if not mne_events_list:
             print("   WARNING: Failed to create MNE events. Skipping file.")
             continue
        mne_events = np.array(mne_events_list, dtype=int)

        # Create Epochs using cleaned data
        epochs = mne.Epochs(raw_cleaned, mne_events, event_id=epoch_event_id_dict,
                           tmin=TMIN, tmax=TMAX, baseline=BASELINE, picks=['eeg'],
                           preload=True, reject=None, verbose=False)

        # --- 10. Standardize Epoch Data ---
        print("   Standardizing epoch data...")
        epochs_data_array = epochs.get_data()
        mean = epochs_data_array.mean(axis=2, keepdims=True)
        std = epochs_data_array.std(axis=2, keepdims=True)
        epsilon = 1e-6
        standardized_data = (epochs_data_array - mean) / (std + epsilon)

        # --- 11. Collect Data and Labels for this File ---
        labels_array = trials_df['vigilance_label'].values
        if len(standardized_data) == len(labels_array):
            print(f"   Adding {len(standardized_data)} epochs from this file.")
            all_epochs_data_list.append(standardized_data)
            all_labels_list.append(labels_array)
            total_valid_epochs_processed += len(standardized_data)
        else:
            print(f"   WARNING: Mismatch epochs ({len(standardized_data)}) vs labels ({len(labels_array)}). Skipping file.")

        # --- Clean up memory for this file ---
        del raw, raw_cleaned, epochs, standardized_data, labels_array, trials_df, events_df, mne_events
        gc.collect()

    except Exception as e_file:
        print(f"\n   ERROR processing file {os.path.basename(file_path)}: {e_file}")
        print(f"   Skipping this file.")
        # Optional: Clean up any partially processed data for this file if necessary
        gc.collect()
        continue # Go to the next file

# --- End of Loop ---

# --- 12. Concatenate and Save Combined Data ---
print("\n--- Concatenating data from all processed files ---")
if not all_epochs_data_list:
    print("Error: No valid epochs processed from any file. Cannot save combined data.")
else:
    try:
        print(f"Concatenating {len(all_epochs_data_list)} data arrays...")
        final_epochs_data = np.concatenate(all_epochs_data_list, axis=0)
        print(f"Concatenating {len(all_labels_list)} label arrays...")
        final_labels = np.concatenate(all_labels_list, axis=0)

        print(f"\nFinal combined data shape: {final_epochs_data.shape}")
        print(f"Final combined labels shape: {final_labels.shape}")
        print(f"Total valid epochs from all files: {total_valid_epochs_processed}")

        print("\n--- Saving Combined Data ---")
        np.save(COMBINED_EPOCHS_SAVE_PATH, final_epochs_data)
        np.save(COMBINED_LABELS_SAVE_PATH, final_labels)
        print(f"Combined Epoch data saved to: {COMBINED_EPOCHS_SAVE_PATH}")
        print(f"Combined Labels saved to: {COMBINED_LABELS_SAVE_PATH}")

    except ValueError as e_concat:
        print(f"\nError during concatenation: {e_concat}")
        print("This might happen if data shapes (e.g., number of channels or time points) are inconsistent across files after processing.")
    except Exception as e_save_final:
        print(f"\nError saving combined data: {e_save_final}")

print("\nPreprocessing script finished.")