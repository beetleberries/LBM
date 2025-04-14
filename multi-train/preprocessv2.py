import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from mne.preprocessing import ICA # Import ICA
from mne.decoding import Scaler # Import Scaler for standardization
import glob # To find files
import gc # For garbage collection

# --- Configuration ---
# <<< SET THESE >>>
DATA_DIR = 'D:/EEG/'      # Directory containing all your .set files
OUTPUT_DIR = 'D:/EEG/processed' # Directory to SAVE processed data
OUTPUT_FILENAME_BASE = 'combined_s02_stimlocked_std_rej' # Updated base name

# --- Channel Identification Parameters ---
# <<< ADJUST THESE based on your typical channel names >>>
# List common EOG/ECG channel names used in your datasets
EOG_CHANNEL_NAMES = ['VEOG', 'HEOG', 'EOG', 'IO'] # Add other variations if needed
ECG_CHANNEL_NAMES = ['ECG', 'EKG']             # Add other variations if needed

# --- Preprocessing Parameters ---
FILTER_LFREQ = 1.0
FILTER_HFREQ = 30.0 # Consider 30-35 Hz if high-freq muscle artifacts are an issue
NOTCH_FREQ = 60.0 # Use 50.0 for Europe/other regions
ICA_N_COMPONENTS = None # Or set to fixed number/variance (e.g., 0.99)
ICA_RANDOM_STATE = 97
ICA_METHOD = 'fastica'
AUTO_DETECT_EOG = True # Script will prioritize explicitly named EOG channels if found
AUTO_DETECT_ECG = True # Script will attempt ECG detection if ECG channels are found

# --- Event & Epoching Parameters ---
DEVIATION_CODES = [251, 252] # Codes indicating stimulus presentation
RESPONSE_CODE = 253       # Code indicating participant response
TMIN = -0.5               # Start time before stimulus event (seconds)
TMAX = 1.5                # End time after stimulus event (seconds) -> TMAX - TMIN = 2.0 seconds
BASELINE = (None, 0)      # Baseline correction using pre-stimulus period (-0.5s to 0s)
RT_FILTER_THRESHOLD = 0.1 # Minimum reaction time to keep a trial

# --- Epoch Rejection Parameters ---
# <<< TUNE THESE thresholds based on visual inspection or typical values >>>
REJECT_CRITERIA = dict(
    eeg=150e-6    # Reject EEG epochs with peak-to-peak amplitude > 150 µV
    # eog=250e-6 # Optionally add EOG rejection threshold if EOG channels are present
)

# --- Data Saving ---
# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
COMBINED_EPOCHS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_epochs.npy')
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
total_epochs_before_rejection = 0
total_epochs_after_rejection = 0

# --- Main Loop Over Files ---
for i, file_path in enumerate(set_files):
    print(f"\n--- Processing File {i+1}/{len(set_files)}: {os.path.basename(file_path)} ---")
    if(i > 10): break 
    try:
        # --- 1. Load Data ---
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

        # --- 2. Handle Channel Info & Montage ---
        print("   Setting Channel Types & Montage...")
        # Identify potential EOG/ECG channels based on names FIRST
        found_eog_chs = [ch for ch in raw.ch_names if ch.upper() in [name.upper() for name in EOG_CHANNEL_NAMES]]
        found_ecg_chs = [ch for ch in raw.ch_names if ch.upper() in [name.upper() for name in ECG_CHANNEL_NAMES]]

        ch_types = {}
        if found_eog_chs:
            ch_types.update({ch: 'eog' for ch in found_eog_chs})
            print(f"     Identified EOG channels: {found_eog_chs}")
        else:
            print("     No explicitly named EOG channels found based on EOG_CHANNEL_NAMES.")

        if found_ecg_chs:
            # Typically only one ECG channel is needed, pick the first found one
            ch_types.update({ch: 'ecg' for ch in found_ecg_chs}) # Set all found as ECG type
            print(f"     Identified ECG channels: {found_ecg_chs}")
        else:
            print("     No explicitly named ECG channels found based on ECG_CHANNEL_NAMES.")

        # Set types for MISC channels (non-EEG/EOG/ECG)
        misc_ch_prefixes = ['position', 'vehicle', 'serial', 'event'] # Add other non-signal prefixes
        possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
        misc_chs = [ch for ch in raw.ch_names if any(prefix in ch.lower() for prefix in misc_ch_prefixes) or \
                    any(ref_name.lower() == ch.lower() for ref_name in possible_ref_names)]
        ch_types.update({ch: 'misc' for ch in misc_chs if ch not in ch_types}) # Avoid overwriting EOG/ECG

        if ch_types:
            raw.set_channel_types(ch_types)

        # Apply standard montage (will only affect channels identified as EEG)
        try:
            raw.set_montage('standard_1020', on_missing='warn', match_case=False, verbose=False)
        except ValueError as e:
            print(f"     Warning: Could not set standard_1020 montage: {e}. Ensure channel names are standard or mapping is provided if needed.")

        # --- 3. Filtering ---
        print(f"   Filtering ({FILTER_LFREQ}-{FILTER_HFREQ} Hz, Notch: {NOTCH_FREQ} Hz)...")
        raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, picks=['eeg', 'eog', 'ecg'], fir_design='firwin', verbose=False) # Filter bio signals
        if NOTCH_FREQ:
            raw.notch_filter(freqs=NOTCH_FREQ, picks=['eeg', 'eog', 'ecg'], fir_design='firwin', verbose=False)

        # --- 4. ICA Artifact Removal ---
        print("   Performing ICA...")
        # Fit ICA only on EEG data, but use a copy that includes EOG/ECG for artifact detection later
        raw_filtered_for_ica = raw.copy() # Use a copy that retains EOG/ECG if present
        picks_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(picks_eeg) == 0:
             print("   WARNING: No EEG channels found after picking. Skipping ICA and file.")
             continue

        ica = ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE, max_iter='auto')
        try:
            ica.fit(raw_filtered_for_ica, picks=picks_eeg, verbose=False) # Fit only on EEG channels
        except ValueError as e_ica_fit:
             print(f"   WARNING: ICA fit failed for this file ({e_ica_fit}). Skipping ICA application.")
             raw_cleaned = raw # Use data before ICA attempt
        else:
            print("     Identifying artifact components...")
            ica.exclude = []

            # Auto-detect EOG artifacts
            if AUTO_DETECT_EOG:
                eog_indices = []
                # Prioritize explicitly set EOG channels
                if found_eog_chs:
                    try:
                        eog_indices, eog_scores = ica.find_bads_eog(raw_filtered_for_ica, ch_name=found_eog_chs, verbose=False)
                        if eog_indices: print(f"       Auto-detected EOG components (using specified channels {found_eog_chs}): {eog_indices}")
                    except Exception as e_eog: print(f"       WARNING: EOG detection using specified channels failed: {e_eog}")
                # Fallback to MNE's automated EOG channel finding if none were specified or detection failed
                if not eog_indices:
                     try:
                         eog_indices, eog_scores = ica.find_bads_eog(raw_filtered_for_ica, ch_name=None, threshold=3.0, verbose=False) # Let MNE try to find EOG chs
                         if eog_indices: print(f"       Auto-detected EOG components (MNE guess): {eog_indices}")
                     except Exception as e_eog_auto: print(f"       WARNING: EOG detection using MNE guess failed: {e_eog_auto}")

                if eog_indices: ica.exclude.extend(eog_indices)

            # Auto-detect ECG artifacts if ECG channels were found
            if AUTO_DETECT_ECG and found_ecg_chs:
                try:
                    # Use the first identified ECG channel, or let MNE choose if multiple
                    ecg_ch_for_detection = found_ecg_chs[0] if len(found_ecg_chs) == 1 else None
                    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_filtered_for_ica, ch_name=ecg_ch_for_detection, method='correlation', threshold='auto', verbose=False)
                    if ecg_indices:
                        print(f"       Auto-detected ECG components (using specified channels {found_ecg_chs}): {ecg_indices}")
                        ica.exclude.extend(ecg_indices)
                except Exception as e_ecg:
                    print(f"       WARNING: ECG artifact detection failed: {e_ecg}")

            # Clean up duplicate indices
            ica.exclude = sorted(list(set(ica.exclude)))

            if ica.exclude: print(f"     Excluding components: {ica.exclude}")
            else: print("     No components marked for exclusion.")

            print("     Applying ICA...")
            # Apply ICA modifying the original 'raw' object for this file
            # We apply to the original 'raw' which contains all channels
            raw_cleaned = raw.copy() # Create copy before applying ICA
            ica.apply(raw_cleaned, exclude=ica.exclude)

        del raw_filtered_for_ica # Free memory
        gc.collect()

        # --- 5. Extract Relevant Event Information ---
        print("   Extracting events...")
        # Check if annotations exist
        if not raw_cleaned.annotations or len(raw_cleaned.annotations) == 0:
             print("   WARNING: No annotations found. Skipping file.")
             del raw, raw_cleaned
             gc.collect()
             continue

        event_list = []
        for ann in raw_cleaned.annotations:
            event_time_sec = ann['onset']
            event_desc = ann['description']
            # Try to convert description to event code, handle potential non-numeric descriptions
            try:
                # Handle cases where descriptions might be floats (e.g., '251.0')
                event_code = int(float(str(event_desc).strip()))
                if event_code in DEVIATION_CODES or event_code == RESPONSE_CODE:
                    event_list.append({'time_sec': event_time_sec, 'code': event_code})
            except (ValueError, TypeError):
                # print(f"   Skipping annotation with non-numeric description: '{event_desc}'")
                pass # Ignore annotations that cannot be converted to int

        if not event_list:
            print(f"   WARNING: No relevant events ({DEVIATION_CODES} or {RESPONSE_CODE}) extracted from annotations. Skipping file.")
            del raw, raw_cleaned
            gc.collect()
            continue

        # --- 6. Organize Events & Calculate RT ---
        print("   Calculating Reaction Times (RT)...")
        events_df = pd.DataFrame(event_list)
        events_df = events_df.sort_values(by='time_sec').reset_index(drop=True)
        events_df['sample'] = (events_df['time_sec'] * raw_cleaned.info['sfreq']).round().astype(int)

        # Find pairs of Deviation (Stimulus) -> Response
        trials = []
        last_deviation_event = None
        for index, event in events_df.iterrows():
            if event['code'] in DEVIATION_CODES:
                # If there was a previous deviation without response, store it only if it's different
                # Or handle cases where multiple stimuli might occur before a response
                last_deviation_event = event # Keep the most recent stimulus
            elif event['code'] == RESPONSE_CODE:
                if last_deviation_event is not None:
                    local_rt_sec = event['time_sec'] - last_deviation_event['time_sec']
                    # Ensure response happens after stimulus
                    if local_rt_sec >= 0:
                        trials.append({
                            'dev_code': last_deviation_event['code'],
                            'dev_time': last_deviation_event['time_sec'],
                            'dev_sample': last_deviation_event['sample'],
                            'response_code': event['code'],
                            'response_time': event['time_sec'],
                            'response_sample': event['sample'],
                            'local_rt_sec': local_rt_sec
                        })
                    # Reset last deviation event after a response is paired
                    last_deviation_event = None
                # else: Response without preceding stimulus (ignore or handle as needed)

        if not trials:
             print("   WARNING: No stimulus-response pairs found. Skipping file.")
             del raw, raw_cleaned, events_df
             gc.collect()
             continue

        trials_df = pd.DataFrame(trials)

        # --- 7. Filter Trials by Reaction Time ---
        initial_trial_count = len(trials_df)
        trials_df = trials_df[trials_df['local_rt_sec'] >= RT_FILTER_THRESHOLD].copy()
        if initial_trial_count > 0:
            removed_count = initial_trial_count - len(trials_df)
            print(f"   RT Filtering: Kept {len(trials_df)}/{initial_trial_count} trials (Removed {removed_count} with RT < {RT_FILTER_THRESHOLD}s).")

        if trials_df.empty:
            print("   WARNING: No valid trials remaining after RT filtering. Skipping file.")
            del raw, raw_cleaned, events_df, trials_df
            gc.collect()
            continue

        # --- 8. Calculate Vigilance Labels ---
        print("   Calculating Vigilance Labels...")
        local_rts_filtered = trials_df['local_rt_sec'].values
        # Calculate baseline RT (e.g., 10th percentile of valid RTs)
        # Handle cases with very few trials
        if len(local_rts_filtered) < 5:
             alert_rt_baseline = np.median(local_rts_filtered) if len(local_rts_filtered) > 0 else 0.5 # Use median or a default if very few
        else:
            # Using 10th percentile as baseline for 'alert' seems more robust than 5th
            alert_rt_baseline = np.percentile(local_rts_filtered, 10)

        # Define thresholds based on the baseline RT
        alert_threshold = 1.5 * alert_rt_baseline  # Faster than this = alert
        drowsy_threshold = 2.5 * alert_rt_baseline # Slower than this = drowsy
        # Adjust multiplier factors (1.5, 2.5) based on literature or data exploration

        labels = []
        for rt in trials_df['local_rt_sec']:
            if rt < alert_threshold:
                labels.append('alert')
            elif rt > drowsy_threshold:
                labels.append('drowsy')
            else:
                labels.append('transition')
        trials_df['vigilance_label'] = labels
        print(f"   Vigilance counts: {dict(trials_df['vigilance_label'].value_counts())}")

        # --- 9. Create MNE Event Array & Epochs ---
        print(f"   Creating stimulus-locked epochs ({TMIN}s to {TMAX}s)...")
        assigned_labels = trials_df['vigilance_label'].unique()
        # Map string labels to integer event IDs for MNE
        vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
        # Create the dict needed for MNE Epochs constructor
        epoch_event_id_dict = {label: vigilance_event_ids[label] for label in assigned_labels if label in vigilance_event_ids}

        if not epoch_event_id_dict:
             print("   WARNING: No trials classified into known states ('alert', 'transition', 'drowsy'). Skipping file.")
             del raw, raw_cleaned, events_df, trials_df
             gc.collect()
             continue

        # Create the events array [[sample, previous_event_id, event_id]]
        # We use the STIMULUS sample ('dev_sample') as the time-locking event
        mne_events_list = []
        for index, trial in trials_df.iterrows():
             label = trial['vigilance_label']
             if label in epoch_event_id_dict:
                 event_id = epoch_event_id_dict[label]
                 # Use the deviation/stimulus sample number
                 stimulus_sample = trial['dev_sample']
                 mne_events_list.append([stimulus_sample, 0, event_id])

        if not mne_events_list:
             print("   WARNING: Failed to create MNE events list from trials. Skipping file.")
             del raw, raw_cleaned, events_df, trials_df
             gc.collect()
             continue

        mne_events = np.array(mne_events_list, dtype=int)

        # --- Create Epochs (with rejection) using ICA cleaned data ---
        # Picks only EEG channels for epoching
        picks_eeg_only = mne.pick_types(raw_cleaned.info, eeg=True, eog=False, ecg=False, misc=False, exclude='bads')
        if len(picks_eeg_only) == 0:
            print("   WARNING: No EEG channels available for epoching after cleaning. Skipping file.")
            del raw, raw_cleaned, events_df, trials_df, mne_events
            gc.collect()
            continue

        try:
            epochs = mne.Epochs(raw_cleaned, mne_events, event_id=epoch_event_id_dict,
                               tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                               picks=picks_eeg_only, # Use only EEG channels for the final data
                               preload=True, reject=REJECT_CRITERIA, # Apply rejection
                               verbose=False) # Set verbose=True to see rejection details

            # Explicitly drop epochs exceeding the rejection threshold
            n_epochs_before_reject = len(epochs)
            total_epochs_before_rejection += n_epochs_before_reject
            epochs.drop_bad(verbose=False) # Drop based on REJECT_CRITERIA
            n_epochs_after_reject = len(epochs)
            total_epochs_after_rejection += n_epochs_after_reject
            print(f"   Epoch rejection: {n_epochs_before_reject - n_epochs_after_reject} epochs dropped ({n_epochs_after_reject} remaining).")

        except Exception as e_epoch:
             print(f"   ERROR creating or rejecting epochs: {e_epoch}. Skipping file.")
             del raw, raw_cleaned, events_df, trials_df, mne_events
             gc.collect()
             continue

        if len(epochs) == 0:
            print("   WARNING: No epochs remaining after rejection. Skipping file.")
            del raw, raw_cleaned, events_df, trials_df, mne_events, epochs
            gc.collect()
            continue

        # --- 10. Standardize Epoch Data (Per Channel, Across Time) ---
        print("   Standardizing epoch data using mne.decoding.Scaler...")
        # Get data AFTER rejection
        epochs_data_array = epochs.get_data(picks='eeg') # Ensure shape is (n_epochs, n_channels, n_times)

        # Use Scaler for robust standardization (mean 0, std 1 per channel)
        # Fit the scaler on the current file's valid epochs
        scaler = Scaler(scalings='mean', with_mean=True, with_std=True)
        # Scaler expects (n_epochs, n_features) or (n_epochs * n_channels, n_times)
        # We need to reshape, scale, then reshape back, OR apply per channel
        # Easiest: fit and transform directly if Scaler handles 3D: (Check MNE docs - yes it does)
        scaler.fit(epochs_data_array)
        standardized_data = scaler.transform(epochs_data_array)

        # --- 11. Collect Data and Labels for this File ---
        # Get labels corresponding to the *remaining* epochs AFTER rejection
        # Create reverse mapping from event ID back to label string
        label_map_rev = {v: k for k, v in vigilance_event_ids.items()}
        # Extract event IDs from the final epochs object
        final_event_ids = epochs.events[:, -1]
        # Convert back to string labels
        labels_array = np.array([label_map_rev[eid] for eid in final_event_ids])

        # Sanity check: Number of epochs should match number of labels
        if len(standardized_data) == len(labels_array):
            print(f"   Adding {len(standardized_data)} standardized epochs from this file.")
            all_epochs_data_list.append(standardized_data)
            all_labels_list.append(labels_array)
            total_valid_epochs_processed += len(standardized_data)
        else:
            # This should ideally not happen if label extraction is correct
            print(f"   CRITICAL WARNING: Mismatch between epochs ({len(standardized_data)}) and labels ({len(labels_array)}) after rejection/standardization. Skipping file data.")

        # --- Clean up memory for this file ---
        del raw, raw_cleaned, epochs, standardized_data, labels_array, trials_df, events_df, mne_events
        gc.collect()

    except Exception as e_file:
        print(f"\n   ERROR processing file {os.path.basename(file_path)}: {e_file}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        print(f"   Skipping this file.")
        # Optional: Clean up any partially processed data for this file if necessary
        gc.collect()
        continue # Go to the next file

# --- End of Loop ---

# --- 12. Concatenate and Save Combined Data ---
print("\n--- Concatenating data from all processed files ---")
if not all_epochs_data_list:
    print("Error: No valid epochs collected from any file. Cannot save combined data.")
else:
    try:
        print(f"Concatenating {len(all_epochs_data_list)} data arrays...")
        final_epochs_data = np.concatenate(all_epochs_data_list, axis=0)
        print(f"Concatenating {len(all_labels_list)} label arrays...")
        final_labels = np.concatenate(all_labels_list, axis=0)

        print(f"\nTotal epochs processed across all files (before rejection): {total_epochs_before_rejection}")
        print(f"Total epochs kept across all files (after rejection): {total_epochs_after_rejection}")
        print(f"Total valid epochs collected for saving: {total_valid_epochs_processed}") # Should match total_epochs_after_rejection if no errors

        print(f"\nFinal combined data shape: {final_epochs_data.shape}") # (n_total_epochs, n_eeg_channels, n_times)
        print(f"Final combined labels shape: {final_labels.shape}")   # (n_total_epochs,)
        print(f"Label distribution: {dict(Counter(final_labels))}")

        print("\n--- Saving Combined Data ---")
        np.save(COMBINED_EPOCHS_SAVE_PATH, final_epochs_data)
        np.save(COMBINED_LABELS_SAVE_PATH, final_labels)
        print(f"Combined Epoch data saved to: {COMBINED_EPOCHS_SAVE_PATH}")
        print(f"Combined Labels saved to: {COMBINED_LABELS_SAVE_PATH}")

    except ValueError as e_concat:
        print(f"\nError during concatenation: {e_concat}")
        print("This might happen if data shapes (e.g., number of channels or time points) are inconsistent across files after processing.")
        print("Check if all files resulted in the same number of EEG channels being epoched.")
    except Exception as e_save_final:
        print(f"\nError saving combined data: {e_save_final}")

print("\nPreprocessing script finished.")