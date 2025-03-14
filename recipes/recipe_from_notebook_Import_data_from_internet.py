# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_ASSISTANT_MAGIC_CELL
# %load_ext ai_code_assistant

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu
import os
import tempfile

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
project = dataiku.api_client().get_default_project()
client = dataiku.api_client()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Retrieve Kaggle username and api key
auth_info = client.get_auth_info(with_secrets=True)
secret_value = None
for secret in auth_info["secrets"]:
    if secret["key"] == "KAGGLE_API_KEY":
        os.environ["KAGGLE_KEY"] = secret["value"]
        
    elif secret["key"] == "KAGGLE_USERNAME":
        os.environ["KAGGLE_USERNAME"] = secret["value"]
        
from kaggle import api # import the already authenticated API client

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Retrieve the folder id where the dataset will be stored
folder_id = next((folder["id"] for folder in project.list_managed_folders() if folder["name"]=="data"), None)
if folder_id is None:
    print("Folder 'data' not found!")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import chardet
folder = dataiku.Folder(folder_id)
dataset_slug = "kazanova/sentiment140"
with tempfile.TemporaryDirectory() as tmpdirname:
    
    api.dataset_download_files(dataset_slug, path=tmpdirname, unzip=True)

    for file in os.listdir(tmpdirname):
        local_file = os.path.join(tmpdirname, file)
        folder.upload_file(file, local_file)
        imported_tweets_df = pd.read_csv(local_file, encoding="latin-1")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
imported_tweets = dataiku.Dataset("imported_tweets")
imported_tweets.write_with_schema(imported_tweets_df)
