import mne
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter

# --- Configuration ---
# <<< SET THIS >>> Path to your EEG file
set_file_path = 'D:/EEG/s02_051115m.set' # Replace with your actual path

# Event code definitions
DEVIATION_CODES = [251, 252] # Stimulus events
RESPONSE_CODE = 253         # The single response event we are interested in

# Epoching parameters (relative to RESPONSE_CODE onset)
TMIN = -0.5  # Start time before response onset (seconds)
TMAX = 1.5   # End time after response onset (seconds)
BASELINE = (None, 0) # Baseline correction from tmin to 0 (relative to response onset)

# Histogram reaction time limit for better visualization (in ms)
HIST_RT_LIMIT_MS = 1500 # Adjust as needed, e.g., 1000 or 2000

# --- Check if file exists ---
if not os.path.exists(set_file_path):
    print(f"Error: File not found at {set_file_path}")
    exit()

print(f"Loading EEG data from: {set_file_path}")
try:
    # Load with verbose=True initially to potentially see reference info during loading
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=True)
    print("\nFile loaded successfully.")

    # --- Identify Channel Types and Potential References ---
    print("\n--- Channel Information ---")
    all_ch_names = raw.ch_names
    print(f"Total channels found: {len(all_ch_names)}")
    print(f"Channel names: {all_ch_names}")

    # Set known non-EEG channels
    non_eeg_channels = [ch for ch in raw.ch_names if 'position' in ch.lower() or 'vehicle' in ch.lower()]
    if non_eeg_channels:
        print(f"Identified potential non-EEG channels by name: {non_eeg_channels}. Setting type to 'misc'.")
        mapping = {ch: 'misc' for ch in non_eeg_channels}
        raw.set_channel_types(mapping)
    else:
        print("No channels with 'position' or 'vehicle' in name found.")

    # Get current channel types after setting misc channels
    ch_types = raw.get_channel_types(unique=False)
    eeg_channels = [name for name, type in zip(all_ch_names, ch_types) if type == 'eeg']
    misc_channels = [name for name, type in zip(all_ch_names, ch_types) if type == 'misc']
    print(f"Number of channels currently typed as EEG: {len(eeg_channels)}")
    print(f"Number of channels currently typed as MISC: {len(misc_channels)}")

    # Attempt to identify potential reference channels
    # 1. Check standard names
    possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
    found_possible_refs = [ch for ch in all_ch_names if any(ref_name.lower() == ch.lower() for ref_name in possible_ref_names)]

    print("\n--- Reference Channel Identification ---")
    if found_possible_refs:
        print(f"Potential reference channels found by common names: {found_possible_refs}, setting to misc")
        # You might want to set their type to 'misc' or handle them specifically
        raw.set_channel_types({ch: 'misc' for ch in found_possible_refs})
    else:
        print("No channels found matching common reference names (e.g., M1, M2, A1, A2, TP9, TP10, REF).")

    # 2. Check if MNE detected a reference during loading or if one is set
    ref_info = raw.info.get('custom_ref_applied', 'N/A')
    print(f"MNE custom reference applied flag: {ref_info}")
    if raw.info['projs']:
         print("Active projectors found (could indicate average reference):")
         for i, proj in enumerate(raw.info['projs']):
              print(f"  Proj {i+1}: {proj['desc']} ({'active' if proj['active'] else 'inactive'})")
    else:
         print("No projectors found in the data.")

    # 3. Guidance based on counts
    num_expected_eeg = 30
    num_eeg_found = len(eeg_channels)
    num_misc_found = len(misc_channels) # Includes vehicle position + potentially references
    num_unaccounted = len(all_ch_names) - num_eeg_found - num_misc_found

    print(f"\nAnalysis based on counts (Expected: 30 EEG, 1 Vehicle, 2 Ref):")
    print(f" - Found {num_eeg_found} EEG channels.")
    print(f" - Found {num_misc_found} MISC channels (Vehicle Pos + others?).")
    print(f" - Found {num_unaccounted} channels with other types (if any).")

    if num_eeg_found == num_expected_eeg + 2 and not found_possible_refs:
         print("INFO: Found 32 channels typed as EEG. The 2 extra EEG channels *might* be the references (e.g., M1/M2 were not renamed). Check channel list carefully.")
    elif num_eeg_found == num_expected_eeg and num_misc_found >= 1 + 2:
         print("INFO: Found 30 EEG channels and >= 3 MISC channels. The references might be among the MISC channels (if not identified by name).")
    elif num_eeg_found == num_expected_eeg:
         print("INFO: Found exactly 30 EEG channels. References might be already applied (average ref?) or are among MISC channels.")
    else:
         print(f"INFO: The count of EEG channels ({num_eeg_found}) doesn't immediately suggest which are references. Manual inspection or dataset documentation needed.")

    print("\nRECOMMENDATION: Please review the channel list and types above. If reference channels are still typed as 'eeg', consider setting their type to 'misc' or excluding them manually before analysis (e.g., using raw.pick() or epochs.pick_types()). Check your dataset's documentation for the exact reference scheme used.")
    # Example: If you identify 'M1', 'M2' are refs: raw.set_channel_types({'M1': 'misc', 'M2': 'misc'})
    # Example: Keep only EEG channels for analysis: raw.pick_types(eeg=True, misc=False, exclude='bads') # Do this *before* epoching if desired


    # --- Apply Montage ---
    try:
        raw.set_montage('standard_1020', on_missing='warn', match_case=False)
        print("\nStandard 10-20 montage applied (with warnings for non-matching channels if any).")
    except ValueError as e:
        print(f"\nCould not set standard montage: {e}. Proceeding without it.")


except Exception as e:
    print(f"Error loading file or processing channel info: {e}")
    exit()

# --- Extract Relevant Event Information ---
print("\nExtracting relevant events from annotations...")
try:
    if not raw.annotations:
         print("Error: No annotations found in the file.")
         exit()

    event_list = []
    for ann in raw.annotations:
        event_time_sec = ann['onset']
        event_desc = ann['description']
        try:
            event_code = int(float(event_desc))
            if event_code in DEVIATION_CODES or event_code == RESPONSE_CODE:
                event_list.append({'time_sec': event_time_sec, 'code': event_code})
        except (ValueError, TypeError):
            pass

except AttributeError:
     print("Error: Could not access raw.annotations.")
     exit()
except Exception as e_annot:
     print(f"An unexpected error occurred while processing annotations: {e_annot}")
     exit()

if not event_list:
    print(f"Error: No relevant events ({DEVIATION_CODES} or {RESPONSE_CODE}) could be extracted.")
    exit()

# --- Organize and Analyze Events ---
events_df = pd.DataFrame(event_list)
events_df = events_df.sort_values(by='time_sec').reset_index(drop=True)
events_df['sample'] = (events_df['time_sec'] * raw.info['sfreq']).round().astype(int)

print(f"\nFound {len(events_df)} relevant events ({DEVIATION_CODES} or {RESPONSE_CODE}).")
print("\n--- Relevant Event Code Summary ---")
code_counts = events_df['code'].value_counts().sort_index()
print(code_counts)

# --- Pair Deviation (Stimulus) and Response Events & Calculate Local RT ---
print(f"\nPairing Deviation ({DEVIATION_CODES}) -> Response ({RESPONSE_CODE}) events...")
trials = []
last_deviation_event = None

for index, event in events_df.iterrows():
    if event['code'] in DEVIATION_CODES:
        last_deviation_event = event
    elif event['code'] == RESPONSE_CODE:
        if last_deviation_event is not None:
            local_rt_sec = event['time_sec'] - last_deviation_event['time_sec']
            if local_rt_sec >= 0: # Ensure RT is not negative
                trials.append({
                    'dev_code': last_deviation_event['code'],
                    'dev_time': last_deviation_event['time_sec'],
                    'dev_sample': last_deviation_event['sample'],
                    'response_code': event['code'],
                    'response_time': event['time_sec'],
                    'response_sample': event['sample'],
                    'local_rt_sec': local_rt_sec
                })
            # else: # Silently skip negative RT trials
            #    pass
            last_deviation_event = None # Reset for next pair

trials_df = pd.DataFrame(trials)
num_expected_trials = min(sum(code_counts.get(code, 0) for code in DEVIATION_CODES),
                          code_counts.get(RESPONSE_CODE, 0))
print(f"Found {len(trials_df)} initial Deviation -> Response pairs (Expected up to {num_expected_trials}).")

# --- Filter out 0ms Reaction Time Trials ---
initial_trial_count = len(trials_df)
# Filter trials where local_rt_sec is greater than 0
print("filtering trials where reaction is less than 100ms")
trials_df = trials_df[trials_df['local_rt_sec'] > 0.1].copy() # Use small epsilon instead of 0 for float safety
filtered_trial_count = len(trials_df)
removed_count = initial_trial_count - filtered_trial_count
if removed_count > 0:
    print(f"Removed {removed_count} trials with zero or negative reaction time.")
else:
    print("No trials with zero or negative reaction time found.")

if trials_df.empty:
    print("Error: No valid trials remaining after filtering 0ms RT. Cannot proceed.")
    exit()

# --- Analyze and Plot Local Reaction Time Histogram (Using Filtered Data) ---
print("\n--- Local Reaction Time Statistics (seconds, RT > 0) ---")
print(f"Interval: Deviation ({DEVIATION_CODES}) -> Response ({RESPONSE_CODE})")
print(trials_df['local_rt_sec'].describe())

print("\nGenerating Local Reaction Time Histogram (RT > 0)...")
plt.figure(figsize=(8, 6))
ax = plt.gca()
local_rt_ms = trials_df['local_rt_sec'] * 1000
n_bins = 50

counts, bin_edges, patches = plt.hist(local_rt_ms, bins=n_bins, color='dodgerblue', edgecolor='black',
         range=(1e-6, HIST_RT_LIMIT_MS) if HIST_RT_LIMIT_MS else None) # Start range slightly above 0

plt.title(f'Local Reaction Time Distribution (RT > 0)\n(Stim {DEVIATION_CODES} -> Resp {RESPONSE_CODE})')
plt.xlabel('Local RT (ms)')
plt.ylabel('Frequency (Trial Count)')
plt.grid(axis='y', alpha=0.7)

if HIST_RT_LIMIT_MS is not None:
    plt.xlim(0, HIST_RT_LIMIT_MS) # Keep xlim starting at 0 for visual clarity
    outliers = local_rt_ms[local_rt_ms > HIST_RT_LIMIT_MS]
    if not outliers.empty:
        ax.text(0.98, 0.95, f'{len(outliers)} outliers > {HIST_RT_LIMIT_MS}ms',
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, fontsize=9, color='red')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- Calculate Alert RT Baseline (Using Filtered Data) ---
print("\nCalculating Alert RT baseline (Individualized Baseline, using RT > 0)...")
local_rts_filtered = trials_df['local_rt_sec'].values # Already filtered

if len(local_rts_filtered) < 20:
     print(f"Warning: Only {len(local_rts_filtered)} valid trials (RT > 0) found. 5th percentile calculation might be unstable.")

if len(local_rts_filtered) > 0:
    alert_rt_baseline = np.percentile(local_rts_filtered, 5) # Use filtered RTs
    print(f"Alert RT Baseline (5th percentile of local RTs > 0): {alert_rt_baseline:.4f} seconds ({alert_rt_baseline*1000:.1f} ms)")

    # --- Classify Trials by Vigilance State based on Local RT (Using Filtered Data) ---
    print("Classifying valid trials (RT > 0) into vigilance states...")
    alert_threshold = 1.5 * alert_rt_baseline
    drowsy_threshold = 2.5 * alert_rt_baseline
    print(f"Alert threshold (< 1.5 * Alert RT Baseline): < {alert_threshold:.4f} s ({alert_threshold*1000:.1f} ms)")
    print(f"Drowsy threshold (> 2.5 * Alert RT Baseline): > {drowsy_threshold:.4f} s ({drowsy_threshold*1000:.1f} ms)")
    print(f"Transition threshold: Between {alert_threshold:.4f}s and {drowsy_threshold:.4f}s")

    labels = []
    # Iterate through the filtered trials_df
    for rt in trials_df['local_rt_sec']:
        if rt < alert_threshold:
            labels.append('alert')
        elif rt > drowsy_threshold:
            labels.append('drowsy')
        else:
            labels.append('transition')

    trials_df['vigilance_label'] = labels # Add labels to the filtered DataFrame
    assigned_labels = trials_df['vigilance_label'].unique()

    print("\n--- Vigilance State Counts (Trials with RT > 0) ---")
    print(trials_df['vigilance_label'].value_counts())

    expected_labels = {'alert', 'transition', 'drowsy'}
    missing_labels = expected_labels - set(assigned_labels)
    if missing_labels:
        print(f"\n*** NOTE: No trials were classified as: {', '.join(missing_labels)} based on the RT thresholds. ***")


    # --- Create MNE Event Array for Epoching (Using Filtered Data) ---
    print("\nCreating MNE event array based on vigilance states (RT > 0), time-locked to Response onset...")

    vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
    epoch_event_id_dict = {label: vigilance_event_ids[label] for label in assigned_labels if label in vigilance_event_ids}

    if not epoch_event_id_dict:
        print("Error: No trials were successfully labeled. Cannot create MNE events.")
        exit()

    print(f"Event IDs being used for epoching: {epoch_event_id_dict}")

    mne_events_list = []
    # Iterate through the filtered trials_df
    for index, trial in trials_df.iterrows():
        if trial['vigilance_label'] in epoch_event_id_dict:
            event_sample = trial['response_sample']
            previous_event_id = 0
            vigilance_id = epoch_event_id_dict[trial['vigilance_label']]
            mne_events_list.append([event_sample, previous_event_id, vigilance_id])

    if not mne_events_list:
        print("Error: Failed to create any MNE event entries. Check labeling and dictionary creation.")
        exit()

    mne_events = np.array(mne_events_list, dtype=int)
    print(f"Generated MNE event array with shape: {mne_events.shape} and dtype: {mne_events.dtype}")

    # --- Create Epochs ---
    print(f"\nCreating epochs around RESPONSE ({RESPONSE_CODE}) ONSET ({TMIN}s to {TMAX}s)...")

    try:
        # Optional: Select only EEG channels before epoching if references identified and unwanted
        # raw_eeg_only = raw.copy().pick_types(eeg=True, exclude='bads')
        # epochs = mne.Epochs(raw_eeg_only, ...) # Use raw_eeg_only here
        # --- or ---
        epochs = mne.Epochs(raw, # Use the original raw object for now
                           mne_events,
                           event_id=epoch_event_id_dict,
                           tmin=TMIN,
                           tmax=TMAX,
                           baseline=BASELINE,
                           preload=True,
                           reject=None,
                           proj=True,
                           verbose=True)
        # Optional: If you want to exclude non-EEG channels *after* epoching
        # epochs.pick_types(eeg=True, exclude='bads')

        print("\nEpochs created successfully!")
        print(epochs)

        print("\n--- Saving Processed Epoch Data and Labels ---")

        # Define file paths for saving (adjust as needed)
        SAVE_DIR = './processed_data' # Create this directory if it doesn't exist
        EPOCHS_SAVE_PATH = os.path.join(SAVE_DIR, 'processed_epochs.npy')
        LABELS_SAVE_PATH = os.path.join(SAVE_DIR, 'processed_labels.npy')

        # Create the directory if it doesn't exist
        os.makedirs(SAVE_DIR, exist_ok=True)

        try:
            epochs_data_array = epochs.get_data()

            if 'trials_df' in locals() and not trials_df.empty:
                 labels_array = trials_df['vigilance_label'].values # Shape: (n_epochs,)
            else:
                 print("Error: Cannot find or access 'trials_df' to retrieve labels.")
                 raise ValueError("Could not reliably get string labels for saving.")


            # 3. Save the data and labels
            np.save(EPOCHS_SAVE_PATH, epochs_data_array)
            np.save(LABELS_SAVE_PATH, labels_array)

            print(f"Successfully saved Epoch data to: {EPOCHS_SAVE_PATH}")
            print(f"  Data shape: {epochs_data_array.shape}")
            print(f"Successfully saved Labels to: {LABELS_SAVE_PATH}")
            print(f"  Labels shape: {labels_array.shape}")
            print(f"  Example Labels: {labels_array[:5]}") # Print first 5 labels

        except Exception as e_save:
            print(f"\nError saving processed data: {e_save}")

        # --- Optional: Visualize Epochs ---
        print("\nGenerating ERP plots for each existing vigilance state...")
        existing_conditions = list(epochs.event_id.keys())
        print(f"Plotting conditions: {existing_conditions}")

        figs = []
        for condition in existing_conditions:
             try:
                 # Select only EEG channels for plotting average
                 evoked = epochs[condition].average().pick_types(eeg=True)
                 fig = evoked.plot(window_title=f'Evoked Response - {condition} (Locked to Resp {RESPONSE_CODE})',
                                                      spatial_colors=True, gfp=True, show=False)
                 figs.append(fig)
             except Exception as e_plot:
                 print(f"Could not plot condition '{condition}': {e_plot}")

        if figs:
             print("Displaying ERP plots...")
             plt.show()
        else:
             print("No ERP plots generated.")


    except ValueError as e_epoch:
        print(f"\nValueError creating epochs: {e_epoch}")
        print(f"Unique event IDs in mne_events (3rd column): {np.unique(mne_events[:, 2])}")
        print(f"Event IDs expected by mne.Epochs: {epoch_event_id_dict}")
    except Exception as e_epoch:
        print(f"\nAn unexpected error occurred during epoch creation: {e_epoch}")

else:
     print("\nNo valid Local Reaction Times (RT > 0) found, cannot classify trials or create epochs.")





print("\nScript finished.")