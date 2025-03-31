# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

from nltk.stem import PorterStemmer

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Input

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cleaned_tweets_encryption = dataiku.Dataset("cleaned_tweets")
cleaned_tweets_encryption_df = cleaned_tweets_encryption.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 8.1. Retrait des duplicatas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
cleaned_tweets_encryption_df = cleaned_tweets_encryption_df[cleaned_tweets_encryption_df['is_duplicated'] == 0]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 8.2. Stemming

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply stemming to the 'text' column

stemmer = PorterStemmer()

def apply_stemming(text):
    words = str(text).split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

cleaned_tweets_encryption_df['text'] = cleaned_tweets_encryption_df['text'].apply(apply_stemming)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
prepared_tweets_encryption = dataiku.Dataset("prepared_tweets")
prepared_tweets_encryption.write_with_schema(cleaned_tweets_encryption_df)