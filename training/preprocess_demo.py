import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA

def process_single_set_file(
    set_file_path,
    deviation_codes=[251, 252],
    response_code=253,
    tmin=-0.5,
    tmax=1.5,
    baseline=(None, 0),
    filter_lfreq=1.0,
    filter_hfreq=40.0,
    notch_freq=60.0
):
    print(f"\n--- Processing file: {set_file_path} ---\n")
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)

    # Set non-EEG channels
    non_eeg_channels = [ch for ch in raw.ch_names if 'position' in ch.lower() or 'vehicle' in ch.lower()]
    if non_eeg_channels:
        raw.set_channel_types({ch: 'misc' for ch in non_eeg_channels})

    # Set reference channels to misc
    possible_ref_names = ['M1', 'M2', 'A1', 'A2', 'TP9', 'TP10', 'REF', 'Reference', 'Mastoid']
    found_refs = [ch for ch in raw.ch_names if ch in possible_ref_names]
    if found_refs:
        raw.set_channel_types({ch: 'misc' for ch in found_refs})

    # Set montage
    raw.set_montage('standard_1020', on_missing='warn', match_case=False)

    # Filtering
    raw.filter(l_freq=filter_lfreq, h_freq=filter_hfreq, picks='eeg', verbose=False)
    raw.notch_filter(freqs=notch_freq, picks='eeg', verbose=False)

    # ICA
    raw_filtered = raw.copy()
    ica = ICA(n_components=None, method='fastica', random_state=97, max_iter='auto')
    ica.fit(raw_filtered, picks='eeg')
    try:
        eog_indices, _ = ica.find_bads_eog(raw_filtered)
        ica.exclude.extend(eog_indices)
    except:
        pass
    ica.apply(raw)

    # Extract events
    event_list = []
    for ann in raw.annotations:
        try:
            code = int(float(ann['description']))
            if code in deviation_codes or code == response_code:
                event_list.append({'onset': ann['onset'], 'code': code})
        except:
            continue

    df = pd.DataFrame(event_list)
    df = df.sort_values(by='onset').reset_index(drop=True)
    df['sample'] = (df['onset'] * raw.info['sfreq']).round().astype(int)

    # Match Deviation -> Response
    trials = []
    last_dev = None
    for _, row in df.iterrows():
        if row['code'] in deviation_codes:
            last_dev = row
        elif row['code'] == response_code and last_dev is not None:
            rt = row['onset'] - last_dev['onset']
            if rt >= 0.1:
                trials.append({'response_sample': row['sample'], 'rt': rt})
            last_dev = None

    if len(trials) == 0:
        print("No valid trials.")
        return None, None

    trial_df = pd.DataFrame(trials)
    rt5 = np.percentile(trial_df['rt'], 5)
    alert_thr = 1.5 * rt5
    drowsy_thr = 2.5 * rt5

    labels = []
    for rt in trial_df['rt']:
        if rt < alert_thr:
            labels.append("alert")
        elif rt > drowsy_thr:
            labels.append("drowsy")
        else:
            labels.append("transition")
    trial_df['label'] = labels

    event_ids = {"alert": 1, "transition": 2, "drowsy": 3}
    events = np.array([
        [samp, 0, event_ids[label]]
        for samp, label in zip(trial_df['response_sample'], trial_df['label'])
    ], dtype=int) 

    epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=baseline, picks="eeg", preload=True, verbose=False)

    data = epochs.get_data()
    mean = data.mean(axis=2, keepdims=True)
    std = data.std(axis=2, keepdims=True)
    std_data = (data - mean) / (std + 1e-6)

    return std_data, trial_df['label'].values

# Wrapper function for external use (e.g., in demo.py)
def run_pipeline(set_file_path):
    return process_single_set_file(set_file_path)