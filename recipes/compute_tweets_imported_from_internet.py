# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import os
import kaggle

project = dataiku.api_client().get_default_project()
client = dataiku.api_client()

# Retrieve Kaggle username and api key
auth_info = client.get_auth_info(with_secrets=True)
secret_value = None
for secret in auth_info["secrets"]:
    if secret["key"] == "KAGGLE_KEY":
        os.environ["KAGGLE_KEY"] = secret["value"]
        break
    if secret["key"] == "KAGGLE_USERNAME":
        os.environ["KAGGLE_USERNAME"] = secret["value"]
        break
if not secret_value:
    raise Exception("secret not found")

# Retrieve the folder id where the dataset will be stored
folder_id = next((folder["id"] for folder in project.list_managed_folders() if fodler["name"]=="data"), None)
if folder_id is None:
    print("Folder 'data' not found!")

# Set the folder path where the dataset will be stored
folder = dataiku.Folder(folder_id)
folder_path = folder.get_path()
print(folder_path)

# Download the Kaggle dataset from internet
dataset_slug = "kazanova/sentiment140"
kaggle.api.dataset_download_files(dataset_slug, path=folder_path, unzip=True)

# Create the dataframe from the csv file
dataset_info = kaggle.api.dataset_metadata(dataset_slug)
dataset_title = dataset_info["title"]
dataset_name = dataset_title + ".csv"
annotated_tweets_df = pd.read_csv(os.path.join(folder_path, dataset_name))

# Write recipe outputs
annotated_tweets = dataiku.Dataset("annotated_tweets")
annotated_tweets.write_with_schema(annotated_tweets_df)
