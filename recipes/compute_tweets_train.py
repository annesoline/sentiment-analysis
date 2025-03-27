# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

# Read recipe inputs
prepared_tweets = dataiku.Dataset("prepared_tweets")
prepared_tweets_df = prepared_tweets.get_dataframe()


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Filter to balance the data on label column

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
prepared_tweets_df['label'].value_counts()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
min_count = prepared_tweets_df['label'].value_counts().min()
balanced_df = prepared_tweets_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), min_count))).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
balanced_df['label'].value_counts()


tweets_train_df = prepared_tweets_df # For this sample code, simply copy input to output


# Write recipe outputs
tweets_train = dataiku.Dataset("tweets_train")
tweets_train.write_with_schema(tweets_train_df)
