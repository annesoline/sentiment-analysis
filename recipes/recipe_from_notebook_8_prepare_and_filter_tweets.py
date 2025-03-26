# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cleaned_tweets_encryption = dataiku.Dataset("cleaned_tweets_encryption")
cleaned_tweets_encryption_df = cleaned_tweets_encryption.get_dataframe()

cleaned_tweets_removal = dataiku.Dataset("cleaned_tweets_removal")
cleaned_tweets_removal_df = cleaned_tweets_removal.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 8.1. Retrait des duplicatas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Remove duplicates based on the 'is_duplicated' column
cleaned_tweets_encryption_df = cleaned_tweets_encryption_df[cleaned_tweets_encryption_df['is_duplicated'] == 0]
cleaned_tweets_removal_df = cleaned_tweets_removal_df[cleaned_tweets_removal_df['is_duplicated'] == 0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 8.2. Stemming

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply stemming to the 'text' column
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def apply_stemming(text):
    words = str(text).split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

cleaned_tweets_encryption_df['text'] = cleaned_tweets_encryption_df['text'].apply(apply_stemming)
cleaned_tweets_removal_df['text'] = cleaned_tweets_removal_df['text'].apply(apply_stemming)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
prepared_tweets_encryption = dataiku.Dataset("prepared_tweets_encryption")
prepared_tweets_encryption.write_with_schema(cleaned_tweets_encryption_df)
prepared_tweets_removal = dataiku.Dataset("prepared_tweets_removal")
prepared_tweets_removal.write_with_schema(cleaned_tweets_removal_df)
