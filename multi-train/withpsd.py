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
import warnings # To manage warnings

# --- Configuration ---
# <<< SET THESE >>>
DATA_DIR = 'D:/EEG/'      # Directory containing all your .set files
OUTPUT_DIR = 'D:/EEG/processed' # Directory to SAVE processed data
# Updated base name to reflect inclusion of PSD
OUTPUT_FILENAME_BASE = 'combined_s02_stimlocked_std_rej_psd'

# --- Channel Identification Parameters ---
EOG_CHANNEL_NAMES = ['VEOG', 'HEOG', 'EOG', 'IO']
ECG_CHANNEL_NAMES = ['ECG', 'EKG']

# --- Preprocessing Parameters ---
FILTER_LFREQ = 4.0
FILTER_HFREQ = 30.0
NOTCH_FREQ = 60.0 # Use 50.0 for Europe/other regions
ICA_N_COMPONENTS = None
ICA_RANDOM_STATE = 97
ICA_METHOD = 'fastica'
AUTO_DETECT_EOG = True
AUTO_DETECT_ECG = True

# --- Event & Epoching Parameters ---
DEVIATION_CODES = [251, 252]
RESPONSE_CODE = 253
TMIN = -0.5
TMAX = 1.5 # Total epoch duration: TMAX - TMIN = 2.0 seconds
BASELINE = (None, 0)
RT_FILTER_THRESHOLD = 0.1

# --- Epoch Rejection Parameters ---
# <<< TUNE THESE >>>
REJECT_CRITERIA = dict(
    eeg=150e-6    # Reject EEG epochs with peak-to-peak amplitude > 150 µV
)

# --- PSD Calculation Parameters ---
# <<< ADJUST THESE as needed >>>
PSD_METHOD = 'welch'      # Method for PSD estimation ('welch' or 'multitaper')
PSD_FMIN = FILTER_LFREQ   # Minimum frequency for PSD (e.g., 1.0)
PSD_FMAX = FILTER_HFREQ   # Maximum frequency for PSD (e.g., 40.0)
PSD_WINDOW = 'hann'       # Window function for Welch method
# n_fft: Length of FFT segments. Rule of thumb: same as sampling freq for 1Hz resolution
# Let's calculate it based on sfreq later, ensures adaptation if sfreq varies slightly
# PSD_N_FFT = 256 # Example: Or calculate based on sfreq * desired_window_length_sec

# --- Data Saving ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Paths for Time-Domain Data
COMBINED_EPOCHS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_time_epochs.npy')
# Paths for PSD Data
COMBINED_PSD_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_psd_epochs.npy')
COMBINED_FREQS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_psd_freqs.npy')
# Path for Labels (applies to both time and PSD epochs)
COMBINED_LABELS_SAVE_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_labels.npy')


# --- Find Input Files ---
set_files = glob.glob(os.path.join(DATA_DIR, '*.set'))
if not set_files:
    print(f"Error: No .set files found in directory: {DATA_DIR}")
    exit()
print(f"Found {len(set_files)} .set files to process in {DATA_DIR}")

# --- Lists to Collect Data from All Files ---
all_epochs_data_list = [] # For time-domain data
all_psd_data_list = []    # For PSD data
all_labels_list = []      # For labels (common to both)
psd_freqs = None          # To store the frequency bins from PSD calculation (should be consistent)

total_valid_epochs_processed = 0
total_epochs_before_rejection = 0
total_epochs_after_rejection = 0

# --- Main Loop Over Files ---
for i, file_path in enumerate(set_files):
    print(f"\n--- Processing File {i+1}/{len(set_files)}: {os.path.basename(file_path)} ---")
    current_file_processed_epochs = 0 # Track epochs added from this file
    #if (i > 10): break
    try:
        # --- 1. Load Data ---
        # Suppress specific EEGLAB reading warnings if they are noisy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

        # --- 2. Handle Channel Info & Montage ---
        # (Code is identical to the previous version - finding EOG/ECG/MISC, setting types/montage)
        print("   Setting Channel Types & Montage...")
        found_eog_chs = [ch for ch in raw.ch_names if ch.upper() in [name.upper() for name in EOG_CHANNEL_NAMES]]
        found_ecg_chs = [ch for ch in raw.ch_names if ch.upper() in [name.upper() for name in ECG_CHANNEL_NAMES]]
        ch_types = {}
        if found_eog_chs: ch_types.update({ch: 'eog' for ch in found_eog_chs}); print(f"     Identified EOG channels: {found_eog_chs}")
        if found_ecg_chs: ch_types.update({ch: 'ecg' for ch in found_ecg_chs}); print(f"     Identified ECG channels: {found_ecg_chs}")
        misc_ch_prefixes = ['position', 'vehicle', 'serial', 'event']
        possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
        misc_chs = [ch for ch in raw.ch_names if any(prefix in ch.lower() for prefix in misc_ch_prefixes) or \
                    any(ref_name.lower() == ch.lower() for ref_name in possible_ref_names)]
        ch_types.update({ch: 'misc' for ch in misc_chs if ch not in ch_types})
        if ch_types: raw.set_channel_types(ch_types)
        try: raw.set_montage('standard_1020', on_missing='warn', match_case=False, verbose=False)
        except ValueError as e: print(f"     Warning: Could not set standard_1020 montage: {e}.")

        # --- 3. Filtering ---
        # (Code is identical to the previous version)
        print(f"   Filtering ({FILTER_LFREQ}-{FILTER_HFREQ} Hz, Notch: {NOTCH_FREQ} Hz)...")
        raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, picks=['eeg', 'eog', 'ecg'], fir_design='firwin', verbose=False)
        if NOTCH_FREQ: raw.notch_filter(freqs=NOTCH_FREQ, picks=['eeg', 'eog', 'ecg'], fir_design='firwin', verbose=False)

        # --- 4. ICA Artifact Removal ---
        # (Code is identical to the previous version - fitting ICA, finding bads, applying)
        print("   Performing ICA...")
        raw_filtered_for_ica = raw.copy()
        picks_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(picks_eeg) == 0: print("   WARNING: No EEG channels found after picking. Skipping ICA and file."); continue
        ica = ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD, random_state=ICA_RANDOM_STATE, max_iter='auto')
        try: ica.fit(raw_filtered_for_ica, picks=picks_eeg, verbose=False)
        except ValueError as e_ica_fit: print(f"   WARNING: ICA fit failed ({e_ica_fit}). Skipping ICA application."); raw_cleaned = raw
        else:
            print("     Identifying artifact components...")
            ica.exclude = []
            if AUTO_DETECT_EOG:
                eog_indices = []
                if found_eog_chs:
                    try: eog_indices, _ = ica.find_bads_eog(raw_filtered_for_ica, ch_name=found_eog_chs, verbose=False); print(f"       EOG components (specified channels): {eog_indices}")
                    except Exception: pass # Ignore if detection fails with specified channels
                if not eog_indices: # Fallback
                    try: eog_indices, _ = ica.find_bads_eog(raw_filtered_for_ica, ch_name=None, verbose=False); print(f"       EOG components (MNE guess): {eog_indices}")
                    except Exception: pass
                if eog_indices: ica.exclude.extend(eog_indices)
            if AUTO_DETECT_ECG and found_ecg_chs:
                try:
                    ecg_ch_for_detection = found_ecg_chs[0] if len(found_ecg_chs) == 1 else None
                    ecg_indices, _ = ica.find_bads_ecg(raw_filtered_for_ica, ch_name=ecg_ch_for_detection, method='correlation', threshold='auto', verbose=False); print(f"       ECG components: {ecg_indices}")
                    if ecg_indices: ica.exclude.extend(ecg_indices)
                except Exception as e_ecg: print(f"       WARNING: ECG artifact detection failed: {e_ecg}")
            ica.exclude = sorted(list(set(ica.exclude)))
            if ica.exclude: print(f"     Excluding components: {ica.exclude}")
            else: print("     No components marked for exclusion.")
            print("     Applying ICA...")
            raw_cleaned = raw.copy(); ica.apply(raw_cleaned, exclude=ica.exclude)
        del raw_filtered_for_ica; gc.collect()

        # --- 5. Extract Relevant Event Information ---
        # (Code is identical to the previous version - finding events from annotations)
        print("   Extracting events...")
        if not raw_cleaned.annotations or len(raw_cleaned.annotations) == 0: print("   WARNING: No annotations found. Skipping file."); del raw, raw_cleaned; gc.collect(); continue
        event_list = []
        for ann in raw_cleaned.annotations:
            try: event_code = int(float(str(ann['description']).strip())); event_time_sec = ann['onset']
            except (ValueError, TypeError): continue
            if event_code in DEVIATION_CODES or event_code == RESPONSE_CODE: event_list.append({'time_sec': event_time_sec, 'code': event_code})
        if not event_list: print(f"   WARNING: No relevant events found. Skipping file."); del raw, raw_cleaned; gc.collect(); continue

        # --- 6. Organize Events & Calculate RT ---
        # (Code is identical to the previous version - pairing stim/resp, calculating RT)
        print("   Calculating Reaction Times (RT)...")
        events_df = pd.DataFrame(event_list).sort_values(by='time_sec').reset_index(drop=True)
        events_df['sample'] = (events_df['time_sec'] * raw_cleaned.info['sfreq']).round().astype(int)
        trials = []; last_deviation_event = None
        for _, event in events_df.iterrows():
            if event['code'] in DEVIATION_CODES: last_deviation_event = event
            elif event['code'] == RESPONSE_CODE and last_deviation_event is not None:
                local_rt_sec = event['time_sec'] - last_deviation_event['time_sec']
                if local_rt_sec >= 0: trials.append({'dev_code': last_deviation_event['code'], 'dev_time': last_deviation_event['time_sec'], 'dev_sample': last_deviation_event['sample'], 'response_code': event['code'], 'response_time': event['time_sec'], 'response_sample': event['sample'], 'local_rt_sec': local_rt_sec})
                last_deviation_event = None
        if not trials: print("   WARNING: No stimulus-response pairs found. Skipping file."); del raw, raw_cleaned, events_df; gc.collect(); continue
        trials_df = pd.DataFrame(trials)

        # --- 7. Filter Trials by Reaction Time ---
        # (Code is identical to the previous version)
        initial_trial_count = len(trials_df)
        trials_df = trials_df[trials_df['local_rt_sec'] >= RT_FILTER_THRESHOLD].copy()
        if initial_trial_count > 0: print(f"   RT Filtering: Kept {len(trials_df)}/{initial_trial_count} trials.")
        if trials_df.empty: print("   WARNING: No valid trials after RT filtering. Skipping file."); del raw, raw_cleaned, events_df, trials_df; gc.collect(); continue

        # --- 8. Calculate Vigilance Labels ---
        # (Code is identical to the previous version - labeling based on RT thresholds)
        print("   Calculating Vigilance Labels...")
        local_rts_filtered = trials_df['local_rt_sec'].values
        alert_rt_baseline = np.percentile(local_rts_filtered, 10) if len(local_rts_filtered) >= 5 else (np.median(local_rts_filtered) if len(local_rts_filtered) > 0 else 0.5)
        alert_threshold = 1.5 * alert_rt_baseline; drowsy_threshold = 2.5 * alert_rt_baseline
        labels = ['alert' if rt < alert_threshold else ('drowsy' if rt > drowsy_threshold else 'transition') for rt in trials_df['local_rt_sec']]
        trials_df['vigilance_label'] = labels
        print(f"   Vigilance counts: {dict(trials_df['vigilance_label'].value_counts())}")

        # --- 9. Create MNE Event Array & Epochs ---
        # (Code is identical to the previous version - creating events for MNE)
        print(f"   Creating stimulus-locked epochs ({TMIN}s to {TMAX}s)...")
        vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
        epoch_event_id_dict = {label: vigilance_event_ids[label] for label in trials_df['vigilance_label'].unique() if label in vigilance_event_ids}
        if not epoch_event_id_dict: print("   WARNING: No known vigilance states found. Skipping file."); del raw, raw_cleaned, events_df, trials_df; gc.collect(); continue
        mne_events_list = [[trial['dev_sample'], 0, epoch_event_id_dict[trial['vigilance_label']]] for _, trial in trials_df.iterrows() if trial['vigilance_label'] in epoch_event_id_dict]
        if not mne_events_list: print("   WARNING: Failed to create MNE events list. Skipping file."); del raw, raw_cleaned, events_df, trials_df; gc.collect(); continue
        mne_events = np.array(mne_events_list, dtype=int)

        # --- Create Epochs (with rejection) using ICA cleaned data ---
        picks_eeg_only = mne.pick_types(raw_cleaned.info, eeg=True, exclude='bads')
        if len(picks_eeg_only) == 0: print("   WARNING: No EEG channels for epoching. Skipping file."); del raw, raw_cleaned, events_df, trials_df, mne_events; gc.collect(); continue

        try:
            epochs = mne.Epochs(raw_cleaned, mne_events, event_id=epoch_event_id_dict,
                               tmin=TMIN, tmax=TMAX, baseline=BASELINE,
                               picks=picks_eeg_only, preload=True, reject=REJECT_CRITERIA,
                               verbose=False)
            n_epochs_before_reject = len(epochs)
            total_epochs_before_rejection += n_epochs_before_reject
            epochs.drop_bad(verbose=False) # Apply rejection specified in constructor
            n_epochs_after_reject = len(epochs)
            total_epochs_after_rejection += n_epochs_after_reject
            print(f"   Epoch rejection: {n_epochs_before_reject - n_epochs_after_reject} epochs dropped ({n_epochs_after_reject} remaining).")
        except Exception as e_epoch: print(f"   ERROR creating/rejecting epochs: {e_epoch}. Skipping file."); del raw, raw_cleaned, events_df, trials_df, mne_events; gc.collect(); continue

        if len(epochs) == 0: print("   WARNING: No epochs remaining after rejection. Skipping file."); del raw, raw_cleaned, events_df, trials_df, mne_events, epochs; gc.collect(); continue

        # --- 10. Calculate Power Spectral Density (PSD) ---
        print(f"   Calculating PSD ({PSD_METHOD}, {PSD_FMIN}-{PSD_FMAX} Hz)...")
        sfreq = epochs.info['sfreq']
        # Calculate n_fft based on desired window length (e.g., 1 second)
        n_fft_calculated = int(sfreq * 1.0) # Use 1-second FFT windows
        # Alternative: Set fixed n_fft like PSD_N_FFT = 256 or 512 if desired

        try:
            # Use compute_psd method on the cleaned Epochs object
            spectrum = epochs.compute_psd(
                method=PSD_METHOD,
                fmin=PSD_FMIN,
                fmax=PSD_FMAX,
                picks='eeg',
                n_fft=n_fft_calculated, # Use calculated n_fft
                window=PSD_WINDOW,
                n_overlap = n_fft_calculated // 2, # Common overlap for Welch
                average=False, # Important: Get PSD for each epoch
                verbose=False
            )
            # Get PSD data: shape (n_epochs, n_channels, n_freqs)
            psd_data = spectrum.get_data()
            # Get frequency bins
            current_freqs = spectrum.freqs

            # Store freqs from the first file, check consistency later
            if psd_freqs is None:
                psd_freqs = current_freqs
            elif not np.array_equal(psd_freqs, current_freqs):
                print(f"   WARNING: PSD frequency bins mismatch between files! Previous: {len(psd_freqs)} bins, Current: {len(current_freqs)} bins. This might indicate inconsistent sfreq or PSD settings.")
                # Handle this: skip file, re-calculate previous PSDs, or ensure settings are identical.
                # For now, we'll continue but saving might fail if shapes mismatch.

        except Exception as e_psd:
            print(f"   ERROR calculating PSD: {e_psd}. Skipping PSD calculation for this file.")
            psd_data = None # Indicate PSD calculation failed for this file

        # --- 11. Standardize Time-Domain Epoch Data ---
        print("   Standardizing time-domain epoch data using mne.decoding.Scaler...")
        try:
            epochs_data_array = epochs.get_data(picks='eeg') # Get time data AFTER rejection
            scaler = Scaler(scalings='mean', with_mean=True, with_std=True)
            scaler.fit(epochs_data_array)
            standardized_data = scaler.transform(epochs_data_array)
        except Exception as e_std:
            print(f"   ERROR standardizing time-domain data: {e_std}. Skipping file.")
            # If standardization fails, we can't proceed with this file's data
            del raw, raw_cleaned, events_df, trials_df, mne_events, epochs
            if 'psd_data' in locals(): del psd_data
            gc.collect()
            continue

        # --- 12. Collect Data and Labels for this File ---
        # Get labels corresponding to the *remaining* epochs AFTER rejection
        label_map_rev = {v: k for k, v in vigilance_event_ids.items()}
        final_event_ids = epochs.events[:, -1]
        labels_array = np.array([label_map_rev[eid] for eid in final_event_ids])

        # Ensure number of epochs matches labels before collecting
        if len(standardized_data) == len(labels_array) and (psd_data is None or len(psd_data) == len(labels_array)):
            print(f"   Adding {len(standardized_data)} epochs (time-domain and PSD) from this file.")
            all_epochs_data_list.append(standardized_data)
            all_labels_list.append(labels_array)
            if psd_data is not None:
                all_psd_data_list.append(psd_data)
                current_file_processed_epochs = len(standardized_data) # Track epochs added
            else:
                # Handle case where PSD failed but time-domain succeeded
                # Need to decide: skip file entirely, or add placeholder/skip PSD for this file?
                # For simplicity now, if PSD failed, we won't add *either* to maintain alignment easily.
                # Let's refine this: if PSD failed, we can still add time-domain and labels, but PSD concat will fail later.
                # Better: Skip adding data from this file if PSD failed and we intend to save aligned PSD.
                 print(f"   Skipping data collection for this file due to PSD calculation failure.")
                 # Remove potentially added time-domain data / labels if needed (shouldn't be added yet)

            total_valid_epochs_processed += current_file_processed_epochs

        else:
            print(f"   CRITICAL WARNING: Mismatch epochs/labels/PSD. Time: {len(standardized_data)}, Labels: {len(labels_array)}, PSD: {len(psd_data) if psd_data is not None else 'N/A'}. Skipping file data.")


        # --- Clean up memory for this file ---
        del raw, raw_cleaned, epochs, standardized_data, labels_array, trials_df, events_df, mne_events
        if 'psd_data' in locals(): del psd_data
        if 'spectrum' in locals(): del spectrum
        gc.collect()

    except Exception as e_file:
        print(f"\n   ERROR processing file {os.path.basename(file_path)}: {e_file}")
        import traceback
        traceback.print_exc()
        print(f"   Skipping this file.")
        gc.collect()
        continue

# --- End of Loop ---

# --- 13. Concatenate and Save Combined Data ---
print("\n--- Concatenating data from all processed files ---")

save_successful = True

# Concatenate Time-Domain Data
if not all_epochs_data_list:
    print("Error: No valid time-domain epochs collected. Cannot save time-domain data.")
    save_successful = False
else:
    try:
        print(f"Concatenating {len(all_epochs_data_list)} time-domain data arrays...")
        final_epochs_data = np.concatenate(all_epochs_data_list, axis=0)
        print(f"Final combined time-domain data shape: {final_epochs_data.shape}")
    except ValueError as e_concat_time:
        print(f"\nError during time-domain data concatenation: {e_concat_time}")
        print("Check for consistent channel counts or time points across files.")
        save_successful = False
    except Exception as e_concat_time_other:
        print(f"\nUnexpected error during time-domain data concatenation: {e_concat_time_other}")
        save_successful = False

# Concatenate PSD Data
if not all_psd_data_list:
    print("Warning: No valid PSD epochs collected. Cannot save PSD data.")
    # Decide if this is an error or just a warning depending on requirements
    # If PSD is essential, set save_successful = False
else:
    # Check if we attempted to collect PSD data but failed for some files
    if len(all_psd_data_list) != len(all_epochs_data_list):
         print("ERROR: Mismatch between number of files contributing time-domain and PSD data. Cannot reliably concatenate PSD.")
         print(f"Time data arrays: {len(all_epochs_data_list)}, PSD data arrays: {len(all_psd_data_list)}")
         save_successful = False
    else:
        try:
            print(f"Concatenating {len(all_psd_data_list)} PSD data arrays...")
            final_psd_data = np.concatenate(all_psd_data_list, axis=0)
            print(f"Final combined PSD data shape: {final_psd_data.shape}")
        except ValueError as e_concat_psd:
            print(f"\nError during PSD data concatenation: {e_concat_psd}")
            print("Check for consistent channel counts or frequency bins across files.")
            save_successful = False
        except Exception as e_concat_psd_other:
            print(f"\nUnexpected error during PSD data concatenation: {e_concat_psd_other}")
            save_successful = False

# Concatenate Labels
if not all_labels_list:
    print("Error: No labels collected. Cannot save labels.")
    save_successful = False
# Check alignment again before saving labels
elif len(all_labels_list) != len(all_epochs_data_list) or (all_psd_data_list and len(all_labels_list) != len(all_psd_data_list)):
     print("ERROR: Mismatch between number of files contributing labels and data. Cannot reliably save labels.")
     save_successful = False
else:
    try:
        print(f"Concatenating {len(all_labels_list)} label arrays...")
        final_labels = np.concatenate(all_labels_list, axis=0)
        print(f"Final combined labels shape: {final_labels.shape}")
        print(f"Label distribution: {dict(Counter(final_labels))}")
    except Exception as e_concat_labels:
        print(f"\nUnexpected error during label concatenation: {e_concat_labels}")
        save_successful = False


# --- Final Saving ---
if save_successful:
    print("\n--- Saving Combined Data ---")
    try:
        # Save Time-Domain Data
        if 'final_epochs_data' in locals():
            np.save(COMBINED_EPOCHS_SAVE_PATH, final_epochs_data)
            print(f"Combined Time-Domain Epoch data saved to: {COMBINED_EPOCHS_SAVE_PATH}")
        else:
            print("Skipping time-domain data saving due to earlier errors.")

        # Save PSD Data
        if 'final_psd_data' in locals() and psd_freqs is not None:
            np.save(COMBINED_PSD_SAVE_PATH, final_psd_data)
            np.save(COMBINED_FREQS_SAVE_PATH, psd_freqs)
            print(f"Combined PSD Epoch data saved to: {COMBINED_PSD_SAVE_PATH}")
            print(f"PSD Frequency bins saved to: {COMBINED_FREQS_SAVE_PATH}")
        elif not all_psd_data_list:
             print("No PSD data was generated or collected, skipping PSD saving.")
        else:
            print("Skipping PSD data saving due to earlier errors or missing frequency bins.")

        # Save Labels
        if 'final_labels' in locals():
            np.save(COMBINED_LABELS_SAVE_PATH, final_labels)
            print(f"Combined Labels saved to: {COMBINED_LABELS_SAVE_PATH}")
        else:
            print("Skipping labels saving due to earlier errors.")

        print(f"\nTotal epochs processed across all files (before rejection): {total_epochs_before_rejection}")
        print(f"Total epochs kept across all files (after rejection): {total_epochs_after_rejection}")
        print(f"Total valid epochs collected for saving (time-domain): {len(final_epochs_data) if 'final_epochs_data' in locals() else 0}")
        print(f"Total valid epochs collected for saving (PSD): {len(final_psd_data) if 'final_psd_data' in locals() else 0}")

    except Exception as e_save_final:
        print(f"\nError saving combined data: {e_save_final}")
else:
    print("\nCombined data saving skipped due to errors during processing or concatenation.")


print("\nPreprocessing script finished.")