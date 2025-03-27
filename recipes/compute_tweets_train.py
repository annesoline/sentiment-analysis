# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
import dataiku

# Read recipe inputs
prepared_tweets = dataiku.Dataset("prepared_tweets")
prepared_tweets_df = prepared_tweets.get_dataframe()

# Split the data into 80% training and 20% eval
train_df = prepared_tweets_df.sample(frac=0.8, random_state=42)
eval_df = prepared_tweets_df.drop(train_df.index)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# Filter to balance the data on label column
# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("VALUE COUNT")
print(train_df['label'].value_counts())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
min_count = train_df['label'].value_counts().min()
balanced_df = train_df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), min_count))).reset_index(drop=True)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print("VALUE COUNT")
print(balanced_df['label'].value_counts())

# Write recipe outputs
tweets_train = dataiku.Dataset("tweets_train")
tweets_train.write_with_schema(balanced_df)

tweets_eval = dataiku.Dataset("tweets_eval")
tweets_eval.write_with_schema(eval_df)
