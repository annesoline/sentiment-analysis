# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
enhanced_tweets_informations = dataiku.Dataset("enhanced_tweets_informations")
enhanced_tweets_informations_df = enhanced_tweets_informations.get_dataframe()


# Compute recipe outputs from inputs

filtered_tweets_df = enhanced_tweets_informations_df 
filtered_tweets_df = filtered_tweets_df.sample(n=500, random_state=42)

# Write recipe outputs
filtered_tweets = dataiku.Dataset("filtered_tweets")
filtered_tweets.write_with_schema(filtered_tweets_df)
