import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from mne.preprocessing import ICA # Import ICA

# --- Configuration ---
# <<< SET THESE >>>
SET_FOLDER = "C:/Users/HYUN/capstone/LBM/dataset_unsupervised"  
set_file_paths = [os.path.join(SET_FOLDER, f) for f in os.listdir(SET_FOLDER) if f.endswith(".set")]


# --- Preprocessing Parameters ---
FILTER_LFREQ = 1.0      # High-pass filter frequency (Hz)
FILTER_HFREQ = 40.0     # Low-pass filter frequency (Hz)
NOTCH_FREQ = 60.0       # Power line frequency (Hz) - Use 50.0 for Europe/other regions
ICA_N_COMPONENTS = None # Number of ICA components (e.g., 15, 20, or None to use data rank)
ICA_RANDOM_STATE = 97   # For reproducibility
ICA_METHOD = 'fastica'
# Set to True to attempt automatic EOG detection (requires clear blink artifacts or EOG channel)
AUTO_DETECT_EOG = True

# --- Event & Epoching Parameters ---
DEVIATION_CODES = [251, 252] # Stimulus events
RESPONSE_CODE = 253         # The single response event we are interested in
TMIN = -0.5                 # Start time before response onset (seconds)
TMAX = 1.5                  # End time after response onset (seconds)
BASELINE = (None, 0)        # Baseline correction (relative to response onset)
RT_FILTER_THRESHOLD = 0.1   # Minimum reaction time in seconds (e.g., 0.1 = 100ms)

# --- Data Saving ---
SAVE_DIR = './processed_data_ica'
os.makedirs(SAVE_DIR, exist_ok=True)

all_epochs = []
all_labels = []

for set_file_path in set_file_paths:
    print(f"\n--- Processing file: {set_file_path} ---\n")

    base_name = os.path.basename(set_file_path).split(".")[0]
    EPOCHS_SAVE_PATH = os.path.join(SAVE_DIR, f'{base_name}_epochs_ica_std.npy')
    LABELS_SAVE_PATH = os.path.join(SAVE_DIR, f'{base_name}_labels.npy')

    # --- 1. Load Data ---
    print(f"Loading EEG data from: {set_file_path}")
    try:
        raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
        print("File loaded successfully.")

        # --- 2. Handle Channel Info & Montage ---
        print("\n--- Setting Channel Types & Montage ---")
        # Set known non-EEG channels (adjust if needed)
        non_eeg_channels = [ch for ch in raw.ch_names if 'position' in ch.lower() or 'vehicle' in ch.lower()]
        if non_eeg_channels:
            print(f"Setting non-EEG channels to 'misc': {non_eeg_channels}")
            raw.set_channel_types({ch: 'misc' for ch in non_eeg_channels})

        # Identify and set potential reference channels (adjust if needed)
        possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
        found_possible_refs = [ch for ch in raw.ch_names if any(ref_name.lower() == ch.lower() for ref_name in possible_ref_names)]
        if found_possible_refs:
            print(f"Setting potential reference channels to 'misc': {found_possible_refs}")
            raw.set_channel_types({ch: 'misc' for ch in found_possible_refs})

        # Apply Montage
        try:
            raw.set_montage('standard_1020', on_missing='warn', match_case=False)
            print("Standard 10-20 montage applied (warnings for non-matching channels are ok).")
        except ValueError as e:
            print(f"Could not set standard montage: {e}.")

        # --- 3. Filtering ---
        print(f"\n--- Applying Filters ---")
        print(f"Filtering: High-pass={FILTER_LFREQ} Hz, Low-pass={FILTER_HFREQ} Hz")
        # Apply band-pass filter (modifies raw in-place)
        raw.filter(l_freq=FILTER_LFREQ, h_freq=FILTER_HFREQ, picks=['eeg'], fir_design='firwin', verbose=False)

        # Apply notch filter for power line noise (modifies raw in-place)
        if NOTCH_FREQ:
            print(f"Applying Notch Filter at {NOTCH_FREQ} Hz")
            raw.notch_filter(freqs=NOTCH_FREQ, picks=['eeg'], fir_design='firwin', verbose=False)

        # Copy raw data before ICA for potential comparison or re-fitting
        raw_filtered = raw.copy()
        print("Filtering complete.")

        # --- 4. ICA Artifact Removal ---
        print("\n--- Performing ICA ---")
        # Define ICA parameters
        ica = ICA(n_components=ICA_N_COMPONENTS,
                method=ICA_METHOD,
                random_state=ICA_RANDOM_STATE,
                max_iter='auto')

        print(f"Fitting ICA ({ICA_METHOD}, n_components={ICA_N_COMPONENTS if ICA_N_COMPONENTS else 'rank'})...")
        # Fit ICA on the filtered EEG data
        ica.fit(raw_filtered, picks='eeg') # Fit only on EEG channels

        print("ICA fitting complete. Identifying artifact components...")
        ica.exclude = [] # Initialize list of components to exclude

        # Attempt automatic EOG detection (if enabled)
        if AUTO_DETECT_EOG:
            try:
                # Try finding EOG components based on EEG channels (less reliable)
                # You might need to specify channel names if you have dedicated EOGs: eog_ch='YourEOGChannel'
                eog_indices, eog_scores = ica.find_bads_eog(raw_filtered, ch_name=None, threshold=3.0, verbose=False) # ch_name=None tries to find from EEG
                if eog_indices:
                    print(f"  Automatically identified potential EOG components: {eog_indices}")
                    ica.exclude.extend(eog_indices)
                else:
                    print("  No EOG components automatically detected based on EEG signals.")
                    print("  INFO: Consider manual inspection or providing EOG channel names if available.")
            except Exception as e_ica_eog:
                print(f"  Warning: Automatic EOG detection failed: {e_ica_eog}")
                print("  INFO: Consider manual inspection.")

        # --- !!! CRITICAL STEP: Manual Inspection Recommended !!! ---
        print("\n--- ICA Manual Inspection Recommended ---")
        print("-> Plot components: ica.plot_components()")
        print("-> Plot sources: ica.plot_sources(raw_filtered)")
        print("-> Plot specific component properties: ica.plot_properties(raw_filtered, picks=[index])")
        print("Identify indices of artifactual components (blinks, muscle, etc.).")
        print("Then, MANUALLY add them to the exclude list before applying ICA:")
        print("Example: ica.exclude.extend([index1, index2])")
        # -------------------------------------------------------
        # --- Example Placeholder for Manual Exclusion (Commented Out) ---
        # manual_exclude_indices = [0, 5] # Replace with indices identified manually
        # print(f"  Adding manually identified components to exclude: {manual_exclude_indices}")
        # ica.exclude.extend(manual_exclude_indices)
        # ica.exclude = sorted(list(set(ica.exclude))) # Remove duplicates and sort
        # ---------------------------------------------------------------

        if not ica.exclude:
            print("\nWARNING: No artifact components identified (automatically or manually). Applying ICA without excluding any components.")
        else:
            print(f"\nExcluding ICA components: {ica.exclude}")

        print("Applying ICA to data...")
        # Apply ICA to a *copy* of the filtered raw data
        # raw_cleaned = ica.apply(raw_filtered.copy(), exclude=ica.exclude)
        # Or apply in-place (be careful):
        ica.apply(raw, exclude=ica.exclude) # Modifies 'raw' directly now
        raw_cleaned = raw # Reference the modified raw object
        print("ICA application complete.")


        # --- 5. Extract Relevant Event Information (from original annotations) ---
        print("\nExtracting relevant events from annotations...")
        # Extract from raw_cleaned.annotations (ICA doesn't change annotations)
        if not raw_cleaned.annotations:
            print("Error: No annotations found in the cleaned file.")
            exit()

        event_list = []
        for ann in raw_cleaned.annotations:
            event_time_sec = ann['onset']
            event_desc = ann['description']
            try:
                event_code = int(float(event_desc))
                if event_code in DEVIATION_CODES or event_code == RESPONSE_CODE:
                    event_list.append({'time_sec': event_time_sec, 'code': event_code})
            except (ValueError, TypeError):
                pass

        if not event_list:
            print(f"Error: No relevant events ({DEVIATION_CODES} or {RESPONSE_CODE}) extracted.")
            exit()

        # --- 6. Organize Events & Calculate RT ---
        events_df = pd.DataFrame(event_list)
        events_df = events_df.sort_values(by='time_sec').reset_index(drop=True)
        # Use sfreq from the cleaned data
        events_df['sample'] = (events_df['time_sec'] * raw_cleaned.info['sfreq']).round().astype(int)

        print(f"\nPairing Deviation ({DEVIATION_CODES}) -> Response ({RESPONSE_CODE}) events...")
        trials = []
        last_deviation_event = None
        for index, event in events_df.iterrows():
            if event['code'] in DEVIATION_CODES:
                last_deviation_event = event
            elif event['code'] == RESPONSE_CODE:
                if last_deviation_event is not None:
                    local_rt_sec = event['time_sec'] - last_deviation_event['time_sec']
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
                    last_deviation_event = None

        trials_df = pd.DataFrame(trials)
        print(f"Found {len(trials_df)} initial Deviation -> Response pairs.")

        # --- 7. Filter Trials by Reaction Time ---
        initial_trial_count = len(trials_df)
        print(f"Filtering trials where reaction time < {RT_FILTER_THRESHOLD*1000:.0f}ms")
        trials_df = trials_df[trials_df['local_rt_sec'] >= RT_FILTER_THRESHOLD].copy()
        filtered_trial_count = len(trials_df)
        removed_count = initial_trial_count - filtered_trial_count
        print(f"Removed {removed_count} trials with reaction time < {RT_FILTER_THRESHOLD*1000:.0f}ms.")

        if trials_df.empty:
            print("Error: No valid trials remaining after filtering RT. Cannot proceed.")
            exit()

        # --- 8. Calculate Vigilance Labels ---
        print("\nCalculating Alert RT baseline & Vigilance Labels...")
        local_rts_filtered = trials_df['local_rt_sec'].values
        if len(local_rts_filtered) < 5: # Need at least a few for percentile
            print(f"Warning: Only {len(local_rts_filtered)} valid trials found. Percentile calculation unreliable.")
            # Handle this case - maybe skip labeling or use fixed thresholds?
            # For now, we proceed but the baseline will be unstable.
            alert_rt_baseline = np.median(local_rts_filtered) # Fallback to median
        elif len(local_rts_filtered) < 20:
            print(f"Warning: Only {len(local_rts_filtered)} valid trials found. 5th percentile might be unstable.")
            alert_rt_baseline = np.percentile(local_rts_filtered, 5)
        else:
            alert_rt_baseline = np.percentile(local_rts_filtered, 5)

        print(f"Alert RT Baseline (5th percentile of valid RTs): {alert_rt_baseline:.4f} s")
        alert_threshold = 1.5 * alert_rt_baseline
        drowsy_threshold = 2.5 * alert_rt_baseline
        print(f"Thresholds: Alert < {alert_threshold:.4f}s | Drowsy > {drowsy_threshold:.4f}s")

        labels = []
        for rt in trials_df['local_rt_sec']:
            if rt < alert_threshold: labels.append('alert')
            elif rt > drowsy_threshold: labels.append('drowsy')
            else: labels.append('transition')
        trials_df['vigilance_label'] = labels
        print("\n--- Vigilance State Counts (Valid Trials) ---")
        print(trials_df['vigilance_label'].value_counts())

        # --- 9. Create MNE Event Array & Epochs ---
        print("\nCreating MNE event array and Epochs...")
        assigned_labels = trials_df['vigilance_label'].unique()
        vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
        epoch_event_id_dict = {label: vigilance_event_ids[label] for label in assigned_labels if label in vigilance_event_ids}

        if not epoch_event_id_dict:
            print("Error: No trials classified into recognized vigilance states. Cannot create epochs.")
            exit()

        mne_events_list = []
        for index, trial in trials_df.iterrows():
            if trial['vigilance_label'] in epoch_event_id_dict:
                mne_events_list.append([trial['response_sample'], 0, epoch_event_id_dict[trial['vigilance_label']]])

        if not mne_events_list:
            print("Error: Failed to create MNE event entries.")
            exit()
        mne_events = np.array(mne_events_list, dtype=int)

        # Create Epochs using the *cleaned* raw data
        epochs = mne.Epochs(raw_cleaned, # Use ICA cleaned data
                        mne_events,
                        event_id=epoch_event_id_dict,
                        tmin=TMIN,
                        tmax=TMAX,
                        baseline=BASELINE,
                        picks=['eeg'], # Explicitly pick only EEG channels for epochs
                        preload=True,  # Preload data for standardization
                        reject=None,   # No rejection here, apply earlier if needed
                        verbose=False) # Less verbose epoch creation

        print(f"\nEpochs created successfully: {len(epochs)} epochs found.")
        print(epochs)

        # --- 10. Standardize Epoch Data ---
        print("\n--- Standardizing Epoch Data (per epoch, per channel) ---")
        # Get data array: (n_epochs, n_channels, n_times)
        epochs_data_array = epochs.get_data() # Use copy=False if memory is tight

        # Calculate mean and std deviation across the time dimension for each epoch and channel
        mean = epochs_data_array.mean(axis=2, keepdims=True)
        std = epochs_data_array.std(axis=2, keepdims=True)
        epsilon = 1e-6 # To avoid division by zero

        # Apply standardization: (data - mean) / (std + epsilon)
        standardized_data = (epochs_data_array - mean) / (std + epsilon)
        print(f"Data standardized. Shape: {standardized_data.shape}")

        # --- 11. Save Processed Data and Labels ---
        print("\n--- Saving Standardized Epoch Data and Labels ---")
        try:
            # Get labels corresponding to the final epochs
            # The trials_df should already be filtered and match the epochs created
            if 'trials_df' in locals() and len(trials_df) == len(epochs):
                labels_array = trials_df['vigilance_label'].values
            else:
                # Fallback or error if lengths don't match
                print(f"Error: Mismatch between number of epochs ({len(epochs)}) and trials in DataFrame ({len(trials_df)}). Cannot reliably save labels.")
                raise ValueError("Epoch and trial count mismatch.")

            # Save the *standardized* data
            np.save(EPOCHS_SAVE_PATH, standardized_data)
            np.save(LABELS_SAVE_PATH, labels_array)

            print(f"Successfully saved Standardized Epoch data to: {EPOCHS_SAVE_PATH}")
            print(f"  Data shape: {standardized_data.shape}")
            print(f"Successfully saved Labels to: {LABELS_SAVE_PATH}")
            print(f"  Labels shape: {labels_array.shape}")
            print(f"  Example Labels: {labels_array[:5]}")

            all_epochs.append(standardized_data)
            all_labels.append(labels_array)

        except Exception as e_save:
            print(f"\nError saving processed data: {e_save}")

        # --- 12. Optional: Visualize ERPs (using original, non-standardized epochs for plotting) ---
        print("\nGenerating ERP plots for each existing vigilance state...")
        existing_conditions = list(epochs.event_id.keys())
        print(f"Plotting conditions: {existing_conditions}")
        figs = []
        for condition in existing_conditions:
            try:
                evoked = epochs[condition].average()
                fig = evoked.plot(window_title=f'Evoked Response - {condition} (Locked to Resp {RESPONSE_CODE})',
                                spatial_colors=True, gfp=True, show=False)

                
                fig_path = os.path.join(SAVE_DIR, f"{base_name}_erp_{condition}.png")
                fig.savefig(fig_path)
                print(f"Saved ERP plot for '{condition}' to {fig_path}")

                plt.close(fig) 

            except Exception as e_plot:
                print(f"Could not plot condition '{condition}': {e_plot}")
        if figs:
            print("Displaying ERP plots...")
            plt.show()
        else:
            print("No ERP plots generated.")


    except ValueError as e_epoch:
        print(f"\nValueError during processing: {e_epoch}")
    except Exception as e_main:
        print(f"\nAn unexpected error occurred during processing: {e_main}")

    print("\nScript finished.")


print("\n--- Saving concatenated results from all files ---")
try:
    all_epochs_np = np.concatenate(all_epochs, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(SAVE_DIR, "all_epochs_ica_std.npy"), all_epochs_np)
    np.save(os.path.join(SAVE_DIR, "all_labels_ica.npy"), all_labels_np)

    print(f"Combined Epochs shape: {all_epochs_np.shape}")
    print(f"Combined Labels shape: {all_labels_np.shape}")
except Exception as e:
    print(f"Error saving concatenated data: {e}")