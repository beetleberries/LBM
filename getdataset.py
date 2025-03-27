import requests

# URL of the dataset
dataset_url = "https://figshare.com/ndownloader/articles/5202739/versions/1"

# Output file name
output_file = "dataset.zip"

# Download the file
response = requests.get(dataset_url, stream=True)

if response.status_code == 200:
    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Dataset downloaded successfully as {output_file}")
else:
    print("Failed to download the dataset.")