# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import numpy as np
from textblob import TextBlob

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
enhanced_tweets_informations = dataiku.Dataset("enhanced_tweets_informations")
whole_df = enhanced_tweets_informations.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

def get_sentiment_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity <= -0.6:
        return "very negative"
    elif -0.6 < polarity <= -0.2:
        return "negative"
    elif -0.2 < polarity <= 0.2:
        return "neutral"
    elif 0.2 < polarity <= 0.6:
        return "positive"
    else:
        return "very positive"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
whole_df['label'] = whole_df["text"].apply(get_sentiment_label)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
whole_df = whole_df[['label', 'target', 'id', 'date', 'flag', 'user', 'text', 'is_duplicated', 'tweet_length_chars', 'tweet_length_words', 'repetitive_letters', 'mention_only', 'unreadable', 'too_many_numbers', 'language']]
whole_df.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

filtered_tweets = dataiku.Dataset("filtered_tweets")
df = filtered_tweets.get_dataframe()

df['label'] = df['text'].apply(get_sentiment_label)

df.head(10)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
labelled_tweets = dataiku.Dataset("labelled_tweets")
labelled_tweets.write_with_schema(pandas_dataframe)