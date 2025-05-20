import os
import mne
import numpy as np
import glob
import pandas as pd # Needed for easier RT calculation
from collections import Counter
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler
import gc

# --- Configuration ---
DATA_DIR = 'D:/EEG/'      # Directory containing all your .set files
OUTPUT_DIR = 'D:/EEG/processed'
OUTPUT_FILENAME_BASE = 'stimlocked_psd_features_labeled' # Base name for saved files

CHANNELS_TO_DROP = ['A1', 'A2', 'vehicle position']
ICA_EOG_CHANNELS = ['FP1', 'FP2']
STIMULUS_EVENT_CODES = [251, 252] # Codes indicating stimulus presentation
RESPONSE_CODE = 253               # Code indicating participant response

FILTER_LFREQ = 1.0
FILTER_HFREQ = 40.0
ICA_N_COMPONENTS = 0.95
ICA_RANDOM_STATE = 97

EPOCH_TMIN = -2.0                 # Start time before stimulus event (seconds)
EPOCH_TMAX = 0                  # End time after stimulus event (seconds)
BASELINE_CORRECTION = (None, 0)   # Use pre-stimulus period for baseline
RT_FILTER_THRESHOLD = 0.1         # Minimum reaction time to keep a trial (seconds)
REJECT_CRITERIA = dict(eeg=150e-6) # Reject epochs with p-p > 150 µV

PSD_FMIN = 1.0
PSD_FMAX = 40.0
PSD_N_FFT = 512

# --- Initialization ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
set_files = glob.glob(os.path.join(DATA_DIR, '*.set'))
all_features_list = []
all_labels_list = []
total_epochs_processed = 0

print(f"Found {len(set_files)} files in {DATA_DIR}")
print(f"Output will be saved to: {OUTPUT_DIR}")
print(f"Epochs: {EPOCH_TMIN}s to {EPOCH_TMAX}s relative to stimulus codes {STIMULUS_EVENT_CODES}")
print(f"Labels based on RT between stimulus and response code {RESPONSE_CODE}")

# --- Main Processing Loop ---
for i, file_path in enumerate(set_files):
    print(f"\nProcessing file {i+1}/{len(set_files)}: {os.path.basename(file_path)}")
    try:

        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)

        existing_drop = [ch for ch in CHANNELS_TO_DROP if ch in raw.info['ch_names']]
        if existing_drop:
            raw.drop_channels(existing_drop)

        raw.filter(FILTER_LFREQ, FILTER_HFREQ, fir_design='firwin', picks='eeg', verbose=False)

        ica = ICA(n_components=ICA_N_COMPONENTS, max_iter="auto", random_state=ICA_RANDOM_STATE)
        ica.fit(raw, picks='eeg', verbose=False)
        eog_inds = []
        for ch_name in ICA_EOG_CHANNELS:
            if ch_name in raw.info['ch_names']:
                try:
                    inds, _ = ica.find_bads_eog(raw, ch_name=ch_name, threshold=1.5, verbose=False)
                    eog_inds.extend(inds)
                except Exception: pass # Ignore if detection fails
        ica.exclude = sorted(list(set(eog_inds)))
        if ica.exclude:
            ica.apply(raw, exclude=ica.exclude, verbose=False)

        print("  Extracting events and calculating RTs...")
        try:
            events_from_annot, event_dict = mne.events_from_annotations(raw, verbose=False)
        except ValueError:
             print("  No annotations found. Skipping file.")
             del raw, ica; gc.collect(); continue

        ## LABELING DATA
        event_list = []
        str_stim_codes = [str(code).strip() for code in STIMULUS_EVENT_CODES]
        str_resp_code = str(RESPONSE_CODE).strip()
        for sample, _, code_int in events_from_annot:
            code_str = str(list(event_dict.keys())[list(event_dict.values()).index(code_int)]).strip()
            time_sec = sample / raw.info['sfreq']
            if code_str in str_stim_codes:
                event_list.append({'time_sec': time_sec, 'sample': sample, 'code': int(float(code_str)), 'type': 'stimulus'})
            elif code_str == str_resp_code:
                event_list.append({'time_sec': time_sec, 'sample': sample, 'code': int(float(code_str)), 'type': 'response'})

        if not event_list:
             print("  No relevant stimulus or response events found. Skipping file.")
             del raw, ica, events_from_annot; gc.collect(); continue

        events_df = pd.DataFrame(event_list).sort_values(by='time_sec').reset_index(drop=True)

        trials = []
        last_stim_event = None
        for _, event in events_df.iterrows():
            if event['type'] == 'stimulus':
                last_stim_event = event
            elif event['type'] == 'response' and last_stim_event is not None:
                rt_sec = event['time_sec'] - last_stim_event['time_sec']
                if rt_sec >= 0: # Ensure response is after stimulus
                     trials.append({
                        'stim_code': last_stim_event['code'],
                        'stim_time': last_stim_event['time_sec'],
                        'stim_sample': last_stim_event['sample'],
                        'resp_time': event['time_sec'],
                        'rt_sec': rt_sec
                    })
                last_stim_event = None # Reset after pairing

        if not trials:
             print("  Could not form stimulus-response pairs. Skipping file.")
             del raw, ica, events_from_annot, events_df; gc.collect(); continue

        trials_df = pd.DataFrame(trials)
        initial_trial_count = len(trials_df)
        trials_df = trials_df[trials_df['rt_sec'] >= RT_FILTER_THRESHOLD].copy()
        print(f"  RT Filtering: Kept {len(trials_df)}/{initial_trial_count} trials (RT >= {RT_FILTER_THRESHOLD}s).")

        if trials_df.empty:
            print("  No trials remaining after RT filtering. Skipping file.")
            del raw, ica, events_from_annot, events_df, trials_df; gc.collect(); continue

        print("  Calculating vigilance labels...")
        rts = trials_df['rt_sec'].values
        alert_rt_baseline = np.percentile(rts, 10) if len(rts) >= 5 else np.median(rts)
        alert_threshold = 1.5 * alert_rt_baseline
        drowsy_threshold = 2.5 * alert_rt_baseline
        labels = ['alert' if rt < alert_threshold else 'drowsy' if rt > drowsy_threshold else 'transition' for rt in rts]
        trials_df['vigilance_label'] = labels
        print(f"  Vigilance counts: {dict(Counter(labels))}")

        vigilance_event_ids = {'alert': 1, 'transition': 2, 'drowsy': 3}
        mne_events_list = []
        final_labels_for_epochs = []
        for _, trial in trials_df.iterrows():
            label = trial['vigilance_label']
            event_id = vigilance_event_ids[label]
            mne_events_list.append([trial['stim_sample'], 0, event_id])
            final_labels_for_epochs.append(label)

        if not mne_events_list:
             print("  Failed to create MNE events for epoching. Skipping file.")
             del raw, ica, events_from_annot, events_df, trials_df; gc.collect(); continue

        mne_events = np.array(mne_events_list, dtype=int)
        epoch_event_id_dict = {label: vigilance_event_ids[label] for label in trials_df['vigilance_label'].unique()}

        print(f"  Creating stimulus-locked epochs ({EPOCH_TMIN}s to {EPOCH_TMAX}s)...")
        epochs = mne.Epochs(raw, mne_events, event_id=epoch_event_id_dict,
                            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=BASELINE_CORRECTION,
                            picks='eeg', preload=True, reject=REJECT_CRITERIA, verbose=False)

        n_epochs_before_reject = len(final_labels_for_epochs) # Use labels list length before drop_bad
        epochs.drop_bad(verbose=False)
        n_epochs_after_reject = len(epochs)
        print(f"  Epoch rejection: {n_epochs_before_reject - n_epochs_after_reject} epochs dropped ({n_epochs_after_reject} remaining).")

        if n_epochs_after_reject == 0:
            print("  No epochs remaining after rejection. Skipping file.")
            del raw, ica, events_from_annot, events_df, trials_df, mne_events, epochs; gc.collect(); continue

        # Get labels corresponding to *kept* epochs
        kept_indices = epochs.selection # Indices of epochs that were NOT dropped
        final_labels_kept = [final_labels_for_epochs[i] for i in kept_indices]

        print("  Calculating PSD features for kept epochs...")
        
        #need to get 39 out #FIX THIS why is it 20???
        psd = epochs.compute_psd(method='welch', fmin=PSD_FMIN, fmax=PSD_FMAX, n_fft=PSD_N_FFT, picks='eeg', verbose=False)
        psds_data = psd.get_data()


        features = psds_data

        if features.shape[0] == len(final_labels_kept):
            print(f"  Extracted features shape: {features.shape}, Labels count: {len(final_labels_kept)}")
            all_features_list.append(features)
            all_labels_list.extend(final_labels_kept) # Use extend for list of strings
            total_epochs_processed += len(final_labels_kept)
        else:
             print(f"  WARNING: Mismatch between features ({features.shape[0]}) and labels ({len(final_labels_kept)}). Skipping data from this file.")

        del raw, ica, events_from_annot, events_df, trials_df, mne_events, epochs, psd, features, final_labels_kept
        gc.collect()

    except Exception as e:
        print(f"  !!! ERROR processing file {os.path.basename(file_path)}: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()

# --- Final Aggregation, Scaling, and Saving ---
print("\n--- Post-processing ---")
if not all_features_list:
    print("No features were extracted from any file. Nothing to save.")
else:
    final_features = np.vstack(all_features_list)
    final_labels = np.array(all_labels_list) # Convert list of strings to numpy array

    print(f"Total valid epochs processed and collected: {total_epochs_processed}")
    print(f"Combined features shape: {final_features.shape}")
    print(f"Combined labels shape: {final_labels.shape}")
    print(f"Label distribution: {dict(Counter(final_labels))}")

    scaler = StandardScaler()
    n_epochs, n_channels, n_freqs = final_features.shape

    # 1. Reshape to (n_epochs, n_channels * n_freqs)
    features_reshaped = final_features.reshape(n_epochs, n_channels * n_freqs)
    print(f"Reshaped features for scaling: {features_reshaped.shape}")

    # 2. Apply StandardScaler
    scaler = StandardScaler()
    features_scaled_reshaped = scaler.fit_transform(features_reshaped)
    print("Applied StandardScaler.")

    # 3. Reshape back to (n_epochs, n_channels, n_freqs)
    features_scaled = features_scaled_reshaped.reshape(n_epochs, n_channels, n_freqs)
    print(f"Scaled features reshaped back to: {features_scaled.shape}")

    # Optional: Reshape for specific ML models (e.g., CNN/LSTM)
    # features_scaled_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], features_scaled.shape[1], 1))
    # print(f"Reshaped features shape: {features_scaled_reshaped.shape}")

    features_save_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_features.npy')
    labels_save_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_FILENAME_BASE}_labels.npy')

    np.save(features_save_path, features_scaled) # Save scaled features
    np.save(labels_save_path, final_labels)     # Save corresponding string labels

    print(f"\nScaled features saved to: {features_save_path}")
    print(f"Labels saved to: {labels_save_path}")

print("\nProcessing complete.")