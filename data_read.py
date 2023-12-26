import os
import wget
import json
import gzip
import pandas as pd


def download_file(url: str, file_name: str):
    data_path = 'data/'
    save_path = data_path + file_name

    # Check if the file already exists
    if os.path.exists(save_path):
        print("File already exists. Not downloading")
    # Else create a new file
    else:
        print(f"Downloading file from {url} to {save_path}...")
        wget.download(url=url, out=save_path)
        print("Download complete.")


def load_data():
    # Extract the gzip format file
    extracted_dataset = gzip.open('data/Luxury_Beauty_5.json.gz', 'rb')

    # Each observation is in a json string format. Use json.loads() to parse it into a dictionary.

    # Note: After iterating through each and every observation, it will get removed from the extracted_dataset variable
    df = pd.DataFrame([json.loads(each_observation) for each_observation in extracted_dataset])

    return df