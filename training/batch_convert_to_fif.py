import mne
import sys
import os
import re # Still useful for channel name safety/truncation

# --- Configuration ---
# The directory containing the .set files you want to convert.
from dotenv import load_dotenv

load_dotenv()
INPUT_EEG_DIR  = os.getenv("DATA_PATH")
assert(INPUT_EEG_DIR != None)

# The suffix to add before the new .fif extension (indicates processing done)
OUTPUT_SUFFIX = '_noannots_raw'
# ---

# --- IMPORTANT ---
# Ensure you run this script with an older version on mne (1.5.1) is what i used
# ---

print(f"--- Starting Batch Conversion ---")
print(f"Input Directory: {INPUT_EEG_DIR}")
print(f"Using MNE version: {mne.__version__}")
print(f"Output suffix: {OUTPUT_SUFFIX}.fif")
print("-" * 30)

conversion_count = 0
error_count = 0

# Iterate through all files in the specified directory
for filename in os.listdir(INPUT_EEG_DIR):
    # Check if the file is an EEGLAB .set file (case-insensitive)
    if filename.lower().endswith('.set'):
        input_filepath = os.path.join(INPUT_EEG_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}{OUTPUT_SUFFIX}.fif"
        output_filepath = os.path.join(INPUT_EEG_DIR, output_filename)

        print(f"Processing: {filename} -> {output_filename}")

        try:
            # 1. Load the .set file using the older MNE version
            raw = mne.io.read_raw_eeglab(input_filepath, preload=True, verbose=False)

            # 2. Sanitize Channel Names (optional but good practice)
            #    Mainly handles potential length issues (like 'vehicle position')
            #    and basic problematic characters.
            original_names = list(raw.ch_names)
            sanitized_names = []
            name_map = {}
            for i, name in enumerate(original_names):
                sanitized_name = str(name)
                # Basic sanitization: remove potentially problematic chars
                sanitized_name = re.sub(r'[^A-Za-z0-9 ._-]', '_', sanitized_name)
                sanitized_name = sanitized_name.strip()
                if not sanitized_name:
                    sanitized_name = f"Ch{i+1}" # Fallback
                # Truncate to FIF standard 15 characters
                if len(sanitized_name) > 15:
                    sanitized_name = sanitized_name[:15]
                sanitized_names.append(sanitized_name)
                if sanitized_name != name:
                    name_map[name] = sanitized_name

            if name_map: # Only rename if changes were made
                rename_dict = {orig: new for orig, new in zip(original_names, sanitized_names) if orig != new}
                raw.rename_channels(rename_dict, verbose=False)
                # print(f"  Renamed channels: {name_map}")


            # 3. Remove Annotations (The key workaround for the save bug)
            if raw.annotations is not None and len(raw.annotations) > 0:
                # print(f"  Removing {len(raw.annotations)} annotations.")
                raw.set_annotations(None)

            # 4. Save as FIF file
            #    Using overwrite=True in case you re-run the script
            raw.save(output_filepath, overwrite=True, verbose=False)
            print(f"  Success: Saved {output_filename}")
            conversion_count += 1

        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            # Optionally print full traceback for debugging specific file errors:
            # import traceback
            # traceback.print_exc()
            error_count += 1
        print("-" * 10) # Separator between files

print("\n--- Batch Conversion Complete ---")
print(f"Successfully converted: {conversion_count} files")
print(f"Failed conversions:     {error_count} files")
print("-" * 30)