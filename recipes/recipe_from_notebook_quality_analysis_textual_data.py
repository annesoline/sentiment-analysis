# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import matplotlib.pyplot as plt
import pandas as pd
import string

import dataiku
pd.set_option('display.max_colwidth', None)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 1. Load tweets dataset

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for duplicate tweets
duplicate_tweets = df.duplicated(subset='text').sum()
print(f"Number of duplicate tweets: {duplicate_tweets}")
# Display some duplicated tweets
duplicated_tweets_df = df[df.duplicated(subset='text', keep=False)]
print("\nSample of duplicated tweets:")
print(duplicated_tweets_df[['text']].head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to preprocess text by converting to lowercase and removing punctuation
def preprocess_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Check for duplicate tweets (case insensitive and ignoring punctuation)
duplicate_tweets_case_insensitive_no_punct = df['text'].apply(preprocess_text).duplicated().sum()
print(f"Number of duplicate tweets (case insensitive and ignoring punctuation): {duplicate_tweets_case_insensitive_no_punct}")

# Create a new column 'is_duplicated' to tag duplicated tweets, except the first one
df['is_duplicated'] = df['text'].apply(preprocess_text).duplicated(keep='first').astype(int)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.2. Tweet length

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create a new dataframe with the original columns plus the new length columns
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
# ## 2.3. Very short tweets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for very short tweets that might be low quality
# Create a new DataFrame with only very short tweets
df_very_short_tweets = df[df['tweet_length_chars'] < 10]

# Count and display the number of very short tweets
very_short_tweets_count = df_very_short_tweets.shape[0]
print(f"\nVery short tweets (<10 chars): {very_short_tweets_count} ({(very_short_tweets_count/len(df)*100):.2f}%)")

# Display examples of very short tweets
print("\nExamples of very short tweets:")
print(df_very_short_tweets['text'].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# As shown in the example above, very short tweets can still be used for sentiment analysis.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.4. Tweet specificities (characters, URL, and mentions)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Unique characters analysis
all_chars = ''.join(df['text'].values)
unique_chars = set(all_chars)
print(f"\nNumber of unique characters used: {len(unique_chars)}")
print(f"Unique characters used: {''.join(sorted(unique_chars))}")

# Create a new column 'repetitive_chars' with 1 if there are repetitive characters and 0 otherwise
df['repetitive_chars'] = df['text'].str.contains(r'(.)\1{4,}').astype(int)

# Count tweets with repetitive characters
repetitive_chars = df['repetitive_chars'].sum()
print(f"\nTweets with repetitive characters: {repetitive_chars} ({(repetitive_chars/len(df)*100):.2f}%)")

# Display examples of tweets with repetitive characters
print("\nExamples of tweets with repetitive characters:")
print(df[df['repetitive_chars'] == 1]['text'].head(10))

# URL and mention analysis
tweets_with_urls = len(df[df['text'].str.contains('http|www', regex=True)])
tweets_with_mentions = len(df[df['text'].str.contains('@')])

# Print examples of tweets with URLs
print("\nExample tweets containing URLs:")
df_with_urls = df[df['text'].str.contains('http|www', regex=True)].copy()
print(df_with_urls['text'].head())

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
# Create new dataframe with mention_only column
df['mention_only'] = df['text'].str.match(r'^\s*@\w+\s*$').astype(int)

# Print summary
print(f"\nTweets that are only mentions: {df['mention_only'].sum()} ({df['mention_only'].sum()/len(df)*100:.2f}%)")
print("\nExample tweets that are only mentions:")
print(df[df['mention_only'] == 1]['text'].head())

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.5. Tweets with special characters and unreadable tweets

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
# Create new dataframe with only unreadable rows
# Create a new column in df with a tag 1/0 if the text is unreadable
df['unreadable'] = df['text'].apply(lambda x: int(sum(1 for c in x if c in special_chars) > 5))

# Create a new dataframe with only unreadable rows
df_unreadable = df[df['unreadable'] == 1].copy()

# Print summary statistics
print("\nTweets with more than 5 special characters:")
print("-" * 50)
print(f"\nNumber of unreadable tweets: {len(df_unreadable)}")
print("\nExample unreadable tweets:")
print(df_unreadable[['text']].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Check for tweets with high percentage of numbers
df_high_percentage_numbers = df.copy()
number_ratio = df_high_percentage_numbers['text'].str.count(r'[0-9]') / df_high_percentage_numbers['tweet_length_chars']
df_high_percentage_numbers['too_many_numbers'] = (number_ratio > 0.3).astype(int)
df['too_many_numbers'] = df_high_percentage_numbers['too_many_numbers']
high_numbers = df_high_percentage_numbers['too_many_numbers'].sum()
print(f"\nTweets with high number ratio (>30%): {high_numbers} ({(high_numbers/len(df_high_percentage_numbers)*100):.2f}%)")
print("\nExamples of tweets with many numbers:")
print(df_high_percentage_numbers[df_high_percentage_numbers['too_many_numbers'] == 1]['text'].head(10))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Comments
# In this section, we reviewed tweets containing excessive special characters that render them unreadable, and created a column to tag these tweets so they can be removed from the dataset later. We did the same for the tweets with too many numbers.

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.6. Word/character ratio

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 2.7. Language detection

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from ftlangdetect import detect
def detect_language(text):
    try:
        return detect(text)['lang']
    except Exception as e:
        return 'unknown'

# Create a new column with the detected language
df['language'] = df['text'].apply(detect_language)

# Display the count of tweets per detected language
language_counts = df['language'].value_counts()
print("\nLanguage distribution in tweets:")
print(language_counts)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# Recipe outputs
enhanced_tweets_informations  = dataiku.Dataset("enhanced_tweets_informations")
enhanced_tweets_informations.write_with_schema(df)

# Dataset df_repetitive renamed to repetitive_tweets by anne-soline.guilbert-ly@dataiku.com on 2025-03-17 10:33:36
repetitive_tweets = dataiku.Dataset("repetitive_tweets")
repetitive_tweets.write_with_schema(df_repetitive)
# Dataset df_unreadable renamed to unreadable_tweets by anne-soline.guilbert-ly@dataiku.com on 2025-03-17 10:33:52
unreadable_tweets = dataiku.Dataset("unreadable_tweets")
unreadable_tweets.write_with_schema(df_unreadable)
# Dataset df_high_percentage_numbers renamed to high_percentage_numbers_tweets by anne-soline.guilbert-ly@dataiku.com on 2025-03-17 10:33:58
high_percentage_numbers_tweets = dataiku.Dataset("high_percentage_numbers_tweets")
high_percentage_numbers_tweets.write_with_schema(df_high_percentage_numbers)

# Dataset df_very_short_tweets renamed to very_short_tweets by anne-soline.guilbert-ly@dataiku.com on 2025-03-17 10:34:03
very_short_tweets = dataiku.Dataset("very_short_tweets")
very_short_tweets.write_with_schema(df_very_short_tweets)
