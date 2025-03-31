import os
import zipfile
import requests
from io import BytesIO

main_url = "https://figshare.com/ndownloader/articles/5202739/versions/1"
main_extract_dir = "data/dataset"

os.makedirs("data", exist_ok=True)

if not os.path.exists(main_extract_dir):
    print("Downloading main dataset...")
    response = requests.get(main_url)
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(main_extract_dir)
    print("Main dataset extracted.")

for i in range(1, 13):
    zip_name = f"{i}.zip"
    zip_path = os.path.join(main_extract_dir, zip_name)
    subject_dir = os.path.join(main_extract_dir, str(i))

    if not os.path.exists(subject_dir):
        print(f"Extracting subject {i}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(subject_dir)

    inner_dirs = os.listdir(subject_dir)
    if len(inner_dirs) == 1:
        nested = os.path.join(subject_dir, inner_dirs[0])
        if os.path.isdir(nested):
            subject_dir = nested
    globals()[f"subject_dir_{i}"] = subject_dir

    if os.path.exists(zip_path):
        os.remove(zip_path)
