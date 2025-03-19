# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Load the required libraries
import os
import torch
import dataiku
import pandas as pd
from transformers import pipeline
from dataiku import pandasutils as pdu
from transformers import AutoTokenizer, AutoModelForTokenClassification
pd.set_option('display.max_colwidth', None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
labelled_tweets = dataiku.Dataset("labelled_tweets")
df = labelled_tweets.get_dataframe()

# 6. Vérification de la présence de données sensibles
## 6.1. Named Entities Recognition
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
hf_transformers_home_dir = os.getenv("HF_HOME")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Compute recipe outputs from inputs
# Define the model to use
model_name = "dslim/bert-base-NER"

def perform_ner_inference(model_name, input_df):
    """
    perform_ner_inference performs NER inference on a dataframe using a specified Hugging Face model.
    
    :param model_name: The name of the Hugging Face model to use for NER.
    :param input_df: The input dataframe with at least two columns, document_id and text.
    :return: pd.DataFrame. A dataframe containing the NER results, with at least columns "document_id", "text", and "predicted_labels".
    """
    # Load the pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_transformers_home_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=hf_transformers_home_dir)

    # Load the token classification pipeline
    token_classification_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first") # pass device=0 if using gpu

    # Perform token classification on each row of the dataframe
    predicted_labels = []
    for index, row in input_df.iterrows():
        document_id = row["id"]
        text = row["text"]
        results = token_classification_pipeline(text)
        predicted_labels.append(results)
        
    input_df['predicted_labels'] = predicted_labels

    return input_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df = perform_ner_inference(model_name, df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import re

# Define a function to extract email addresses
def extract_email(text):
    # Regular expression pattern for matching email addresses
    email_pattern = r'[\w\-\.]+@([\w-]+\.)+[\w-]{2,}'
    # Search for the pattern in the text
    match = re.search(email_pattern, text)
    if match:
        return match.group(0)
    else:
        return None

# Apply the function to the 'text' column and create a new column 'email_present'
df['email_present'] = df['text'].apply(extract_email)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

# Define a function to check for IP addresses and URLs containing IPs
def check_ip_or_url_presence(text):
    # Regular expression pattern for matching IP addresses
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    # Regular expression pattern for matching URLs containing IP addresses
    url_pattern = r'\b(?:http|https)://(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?:/[^\s]*)?\b'
    
    # Search for the URL pattern first
    url_match = re.search(url_pattern, text)
    if url_match:
        return url_match.group(0)
    
    # If no URL is found, search for the IP pattern
    ip_match = re.search(ip_pattern, text)
    if ip_match:
        return ip_match.group(0)
    
    return None

# Apply the function to the 'text' column and create a new column 'ip_present'
df['ip_present'] = df['text'].apply(check_ip_or_url_presence)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
sensible_data_identified = dataiku.Dataset("sensible_data_identified")
sensible_data_identified.write_with_schema(df)
