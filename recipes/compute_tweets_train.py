# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
prepared_tweets = dataiku.Dataset("prepared_tweets")
prepared_tweets_df = prepared_tweets.get_dataframe()


# Compute recipe outputs from inputs
# TODO: Replace this part by your actual code that computes the output, as a Pandas dataframe
# NB: DSS also supports other kinds of APIs for reading and writing data. Please see doc.

tweets_train_df = prepared_tweets_df # For this sample code, simply copy input to output


# Write recipe outputs
tweets_train = dataiku.Dataset("tweets_train")
tweets_train.write_with_schema(tweets_train_df)
