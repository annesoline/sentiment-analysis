# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
import pandas as pd
import re

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sensitive_data_identified = dataiku.Dataset("sensitive_data_identified")
df = sensitive_data_identified.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 7. Nettoyage de la donnée
# ## 7.1. Encryption des données sensibles

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df["encrypted_text"] = df["text"]
df["text_without_sensitive_data"] = df["text"]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df.head()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Encryption des NER : création de deux datasets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Define a function to encrypt text
def encrypt_text(text):
    return cipher_suite.encrypt(text.encode()).decode()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def process_NER(row):
    ner_list = eval(row['NER'])
    if ner_list:
        for entity in ner_list:
            word = entity['word']
            # Encrypt and replace the word with its encrypted version in encrypted_text
            row['encrypted_text'] = re.sub(r'\b' + re.escape(word) + r'\b', encrypt_text(word), row['encrypted_text'])

    return row

# Apply the function to the dataframe
df = df.apply(process_NER, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ### Encryption des emails et adresses IP

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Encrypt emails and IPs if present, otherwise fill with original text
def process_emails_and_ip(row):
    if pd.notnull(row['email_present']):
        # Encrypt email in encrypted_text column
        row['encrypted_text'] = re.sub(row['email_present'], 
                             lambda match: encrypt_text(match.group()), row['encrypted_text'])

    if pd.notnull(row['ip_present']):
        # Encrypt IP in encrypted_text column
        row['encrypted_text'] = re.sub(row['ip_present'], 
                             lambda match: encrypt_text(match.group()), row['encrypted_text'])
            
    return row

# Apply the encryption function to the dataframe
df = df.apply(process_emails_and_ip, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 7.2. Retrait des stopwords et des caractères spéciaux, et normalisation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from nltk.corpus import stopwords
import nltk

# Ensure the stopwords are downloaded
nltk.download('stopwords')

lang_dic = {
    "en": "english",
    "es": "spanish",
    "de": "german",
    "pl": "polish",
    "fr": "french",
    "pt": "portuguese",
    "nl": "dutch",
    "fa": "persian",
    "ja": "japanese",
    "ms": "malay",
    "et": "estonian",
    "vi": "vietnamese",
    "ur": "urdu",
    "cy": "welsh",
    "cs": "czech",
    "it": "italian",
    "zh": "chinese",
    "id": "indonesian",
    "ru": "russian",
    "sl": "slovene",
    "ko": "korean",
    "la": "latin",
    "no": "norwegian",
    "ro": "romanian",
    "fi": "finnish",
    "tl": "tagalog",
    "uk": "ukrainian",
    "hu": "hungarian",
    "ca": "catalan",
    "sv": "swedish",
    "tr": "turkish",
    "nds": "low_saxon",
    "da": "danish",
    "ta": "tamil",
    "lb": "luxembourgish",
    "bn": "bengali",
    "mr": "marathi",
    "eo": "esperanto",
    "th": "thai",
    "hi": "hindi",
    "af": "afrikaans",
    "sk": "slovak",
    "so": "somali",
    "is": "icelandic",
    "br": "breton",
    "te": "telugu",
    "ar": "arabic",
    "sh": "serbo_croatian",
    "ceb": "cebuano",
    "eu": "basque",
    "kn": "kannada",
    "ml": "malayalam",
    "gl": "galician",
    "qu": "quechua",
    "gom": "goan_konkani",
    "bs": "bosnian",
    "war": "waray",
    "sq": "albanian",
    "el": "greek",
    "sr": "serbian",
    "az": "azerbaijani",
    "lt": "lithuanian",
    "dv": "divehi",
    "si": "sinhala",
    "kw": "cornish",
    "fy": "frisian",
    "he": "hebrew",
    "ast": "asturian",
    "kk": "kazakh",
    "mk": "macedonian",
    "rm": "romansh",
    "hr": "croatian",
    "lo": "lao",
    "km": "khmer",
    "su": "sundanese",
    "ilo": "ilocano",
    "lv": "latvian",
    "ie": "interlingue",
    "vo": "volapük",
    "pms": "piedmontese",
    "uz": "uzbek",
    "ia": "interlingua",
    "nn": "nynorsk",
    "sw": "swahili",
    "als": "alsatian",
    "jv": "javanese",
    "gu": "gujarati",
    "tt": "tatar",
    "oc": "occitan",
    "ne": "nepali",
    "jbo": "lojban",
    "sco": "scots",
    "ce": "chechen",
    "ga": "irish",
    "lmo": "lombard",
    "ka": "georgian",
    "vec": "venetian",
    "mn": "mongolian",
    "mg": "malagasy",
    "hy": "armenian",
    "bcl": "bikol",
    "an": "aragonese",
    "sd": "sindhi",
    "wa": "walloon",
    "io": "ido",
    "li": "limburgish",
    "my": "burmese",
    "hsb": "upper_sorbian",
    "bh": "bihari",
    "as": "assamese",
    "cbk": "chavacano",
    "yo": "yoruba",
    "mt": "maltese",
    "gd": "scottish_gaelic",
    "nah": "nahuatl",
    "min": "minangkabau",
    "tk": "turkmen",
    "tg": "tajik",
    "bar": "bavarian",
    "ku": "kurdish",
    "be": "belarusian",
    "pa": "punjabi",
    "new": "newari"
}

# Define a function to remove stopwords from text based on language
def remove_stopwords(text, lang):
    try:
        # Use lang_dic to get the language name
        language_name = lang_dic.get(lang, 'english')
        stop_words = set(stopwords.words(language_name))
    except OSError:
        # If the language is not supported, default to English
        stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the function to the 'text' column based on the 'language' column
df['encrypted_text'] = df.apply(lambda row: remove_stopwords(row['encrypted_text'], row['language']), axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to remove special characters from text
def remove_special_characters(text):
    # Use regex to remove special characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Apply the function to the 'text' column
df['text'] = df['text'].apply(remove_special_characters)
df['encrypted_text'] = df['encrypted_text'].apply(remove_special_characters)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Normalize the text
df['text'] = df['text'].str.lower()
df['text_without_sensitive_data'] = df['text_without_sensitive_data'].str.lower()
df['encrypted_text'] = df['encrypted_text'].str.lower()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create one dataframe with encrypted sensitive data
cleaned_tweets_encryption_df = df.drop(columns=['text'], errors='ignore')
cleaned_tweets_encryption_df = cleaned_tweets_encryption_df.rename(columns={'encrypted_text': 'text'})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
cleaned_tweets_encryption = dataiku.Dataset("cleaned_tweets_encryption")
cleaned_tweets_encryption.write_with_schema(cleaned_tweets_encryption_df)

cleaned_tweets_removal = dataiku.Dataset("cleaned_tweets_removal")
cleaned_tweets_removal.write_with_schema(cleaned_tweets_removal_df)
