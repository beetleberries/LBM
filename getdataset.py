import requests
import zipfile
import os


dataset_url = "https://figshare.com/ndownloader/articles/5202739/versions/1"
output_file = "dataset.zip"
extract_folder = "dataset"

if os.path.exists(extract_folder):
    print("Dataset already exists.")
    exit()


print(f"downloading from {dataset_url}")
response = requests.get(dataset_url, stream=True)
if response.status_code != 200:
    print("Failed to download the dataset.")
    exit()

with open(output_file, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
print(f"Dataset downloaded successfully as {output_file}")

with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
print(f"and extracted to {extract_folder}")
os.remove(output_file) 

for root, _, files in os.walk(extract_folder):
    for file in files:
        if file.endswith(".zip"):
            nested_zip_path = os.path.join(root, file)
            nested_extract_folder = os.path.join(root, file[:-4])
            with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip_ref:
                nested_zip_ref.extractall(root)  
            os.remove(nested_zip_path) 
