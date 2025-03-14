# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_colwidth', None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 1. Load tweets dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Read recipe inputs
imported_tweets = dataiku.Dataset("imported_tweets")
df = imported_tweets.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 2. Textual data quality analysis
# ## 2.1. Basic information

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.info()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(f"Values taken by the column flag: {df['flag'].unique()[0]}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
print(f"There are {df['user'].nunique()} different users.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for empty tweets
empty_tweets = len(df[df['text'].str.len() == 0])
print(f"\nNumber of empty tweets: {empty_tweets}")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.2. Tweet length

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df['tweet_length_chars'] = df['text'].str.len()
df['tweet_length_words'] = df['text'].str.split().apply(len)
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Tweet column analysis
print("\nTweet length statistics:")
print(df['tweet_length_chars'].describe())

# Plot distribution of tweet lengths
plt.figure(figsize=(12,6))
plt.hist(df['tweet_length_chars'], bins=50, edgecolor='black')
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

# Most common tweet lengths in words
print("\nMost common tweet lengths (in words):")
print(df['tweet_length_words'].value_counts().head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.3. Tweet specificities (characters, URL, and mentions)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Unique characters analysis
all_chars = ''.join(df['text'].values)
unique_chars = set(all_chars)
print(f"\nNumber of unique characters used: {len(unique_chars)}")
print(f"Unique characters used: {''.join(sorted(unique_chars))}")

# Check for repetitive characters (like 'aaaaaa' or '!!!!!!!')
repetitive_chars = df[df['text'].str.match(r'.*(.)\1{4,}.*')].shape[0]
print(f"\nTweets with repetitive characters: {repetitive_chars} ({(repetitive_chars/len(df)*100):.2f}%)")
print("\nExamples of tweets with repetitive characters:")
print(df[df['text'].str.match(r'.*(.)\1{4,}.*')]['text'].head(10))


# URL and mention analysis
tweets_with_urls = len(df[df['text'].str.contains('http|www', regex=True)])
tweets_with_mentions = len(df[df['text'].str.contains('@')])

# Print examples of tweets with URLs
print("\nExample tweets containing URLs:")
urls = df['text'].str.extract(r'\b(http|www\S+)', expand=False)
print(df[urls.notna()]['text'].head())

# Print examples of tweets with mentions
print("\nExample tweets containing @mentions:")
print(df[df['text'].str.contains('@')]['text'].head())

print(f"\nTweets containing URLs: {tweets_with_urls} ({tweets_with_urls/len(df)*100:.2f}%)")
print(f"Tweets containing @mentions: {tweets_with_mentions} ({tweets_with_mentions/len(df)*100:.2f}%)")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# Tweets with repetitive characters often contain repeated dots or letters. For the latter, we can easily eliminate the repetition in the letters, helping the model better understand the words.
# 
# It will be useful to remove all the URL from the text to ease the detection of sentiment in the text.
# 
# Tweets containing only a mention in the text are tagged, allowing them to be removed from the dataset later, as they provide no relevant information for sentiment analysis.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create mention_only column
# Pattern matches tweets that only contain @ followed by word characters
df['mention_only'] = df['text'].str.match(r'^\s*@\w+\s*$').astype(int)

# Print summary
print(f"\nTweets that are only mentions: {df['mention_only'].sum()} ({df['mention_only'].sum()/len(df)*100:.2f}%)")
print("\nExample tweets that are only mentions:")
print(df[df['mention_only'] == 1]['text'].head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.4. Tweets with special characters and unreadable tweets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import string

special_chars = [c for c in unique_chars 
                if c not in string.ascii_letters 
                and c not in string.digits
                and c not in string.punctuation
                and not c.isalpha()
                and c not in ['¸', '·', ' ', '´', '»', '«']]  # Excludes accented letters, euro symbol, and specific characters
print('Special characters:', (sorted(special_chars)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create unreadable column based on special character count
df['unreadable'] = df['text'].apply(lambda x: 1 if sum(1 for c in x if c in special_chars) > 5 else 0)

# Print summary statistics
print("\nTweets with more than 5 special characters:")
print("-" * 50)
print(f"\nNumber of unreadable tweets: {df['unreadable'].sum()}")
print("\nExample unreadable tweets:")
print(df[df['unreadable'] == 1][['text']].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Show tweets containing special characters
print("\nTweets containing special characters:")
print("-" * 50)
for char in special_chars:
    tweets_with_char = df[df['text'].str.contains(char, regex=False)]
    if len(tweets_with_char) > 0:
        print(f"\nTweets containing '{char}':")
        print(tweets_with_char[['text']].head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for tweets with high percentage of numbers
number_ratio = df['text'].str.count(r'[0-9]') / df['tweet_length_chars']
df['too_many_numbers'] = (number_ratio > 0.3).astype(int)
high_numbers = df['too_many_numbers'].sum()
print(f"\nTweets with high number ratio (>30%): {high_numbers} ({(high_numbers/len(df)*100):.2f}%)")
print("\nExamples of tweets with many numbers:")
print(df[df['too_many_numbers'] == 1]['text'].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# In this section, we reviewed tweets containing excessive special characters that render them unreadable, and created a column to tag these tweets so they can be removed from the dataset later.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.5. Short tweets, repetitive characters, all caps tweets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for very short tweets that might be low quality
very_short_tweets = df[df['tweet_length_chars'] < 10].shape[0]
print(f"\nVery short tweets (<10 chars): {very_short_tweets} ({(very_short_tweets/len(df)*100):.2f}%)")
print("\nExamples of very short tweets:")
print(df[df['tweet_length_chars'] < 10]['text'].head(10))

# Check for all caps tweets (possible spam/low quality)
all_caps_tweets = df[df['text'].str.match(r'^[A-Z0-9\s\W]+$')].shape[0]
print(f"\nAll caps tweets: {all_caps_tweets} ({(all_caps_tweets/len(df)*100):.2f}%)")
print("\nExamples of all caps tweets:")
print(df[df['text'].str.match(r'^[A-Z0-9\s\W]+$')]['text'].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# As shown in the example above, very short tweets can still be used for sentiment analysis.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.6. Average punctuation marks per tweet, word/character ratio

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Calculate average punctuation per tweet
punct_counts = df['text'].str.count(f'[{string.punctuation}]')
avg_punct = punct_counts.mean()
print(f"\nAverage punctuation marks per tweet: {avg_punct:.2f}")

# Check for tweets with excessive punctuation
excessive_punct = df[punct_counts > punct_counts.mean() + 2*punct_counts.std()].shape[0]
print(f"Tweets with excessive punctuation: {excessive_punct} ({(excessive_punct/len(df)*100):.2f}%)")

# Analyze word/character ratio (very low ratio might indicate spam or low quality)
char_word_ratio = df['tweet_length_chars'] / df['tweet_length_words']
suspicious_ratio = df[char_word_ratio > char_word_ratio.mean() + 2*char_word_ratio.std()].shape[0]
print(f"\nTweets with suspicious character-to-word ratio: {suspicious_ratio} ({(suspicious_ratio/len(df)*100):.2f}%)")
print("\nExamples of tweets with suspicious character-to-word ratio:")
print(df[char_word_ratio > char_word_ratio.mean() + 2*char_word_ratio.std()]['text'].head(10))

# Distribution of character-to-word ratios
plt.figure(figsize=(12, 6))
plt.hist(char_word_ratio, bins=50, edgecolor='black')
plt.title('Distribution of Character-to-Word Ratios')
plt.xlabel('Characters per Word')
plt.ylabel('Frequency')
plt.show()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# Tweets with excessive punctuations are often containing dots or a URL. Therefore, it is not necessary to remove them from the dataset.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Replace &quot; with " in text column
df['text'] = df['text'].str.replace('&quot;', '"')

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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
        imported_tweets_df = pd.read_csv(local_file, encoding="latin-1", 
                                       names=['target', 'id', 'date', 'flag', 'user', 'text'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Write recipe outputs
imported_tweets = dataiku.Dataset("imported_tweets")
imported_tweets.write_with_schema(imported_tweets_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 3. Textual data visual exploration

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Word Cloud Visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# Combine all tweets into one text
all_text = ' '.join(df['text'].astype(str))

# Clean text - remove URLs, mentions, special chars
all_text = re.sub(r'http\S+|@\S+|[^\w\s]', '', all_text.lower())

# Create word cloud
wordcloud = WordCloud(width=1200, height=600, 
                     background_color='white',
                     max_words=100).generate(all_text)

# Display the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Tweets')
plt.show()