# Load the required libraries
import os
import torch
import dataiku
import pandas as pd
from transformers import pipeline
from dataiku import pandasutils as pdu
from transformers import AutoTokenizer, AutoModelForTokenClassification

hf_transformers_home_dir = os.getenv("HF_HOME")

# Read recipe inputs
labelled_tweets = dataiku.Dataset("labelled_tweets")
df = labelled_tweets.get_dataframe()

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

document_scored_df = perform_ner_inference(model_name, df)

tweets_NER_with_Python_code_df = df # For this sample code, simply copy input to output

# Write recipe outputs
tweets_NER_with_Python_code = dataiku.Dataset("tweets_NER_with_Python_code")
tweets_NER_with_Python_code.write_with_schema(tweets_NER_with_Python_code_df)
