import dataiku

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sensitive_data_identified = dataiku.Dataset("sensitive_data_identified")
df = sensitive_data_identified.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # 7. Nettoyage de la donnée
# ## 7.1. Encryption des données sensibles

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from cryptography.fernet import Fernet

# Generate a key for encryption
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Define a function to encrypt text
def encrypt_text(text):
    return cipher_suite.encrypt(text.encode()).decode()

# Encrypt emails and IPs if present
def encrypt_sensitive_data(row):
    if pd.notnull(row['email_present']) or pd.notnull(row['ip_present']):
        if pd.notnull(row['email_present']):
            # Encrypt email
            row['encrypted_text'] = re.sub(row['email_present'], 
                                 lambda match: encrypt_text(match.group()), row['text'])
        if pd.notnull(row['ip_present']):
            # Encrypt IP
            row['encrypted_text'] = re.sub(row['ip_present'], 
                                 lambda match: encrypt_text(match.group()), row['text'])
    return row

# Apply the encryption function to the dataframe
df = df.apply(encrypt_sensitive_data, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Print the text of rows where email_present is not None and ip_present is not None
email_and_ip_present_rows = df[df['email_present'].notnull()| df['ip_present'].notnull()]
for index, row in email_and_ip_present_rows.iterrows():
    print(row['encrypted_text'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def remove_sensitive_data(row):
    text = row['text']
    # Remove encrypted emails if email_present is not null
    text = text.replace(row['email_present'], '') if pd.notnull(row['email_present']) else text
    # Remove encrypted IPs if ip_present is not null
    text = text.replace(row['ip_present'], '') if pd.notnull(row['ip_present']) else text
    
    return text

# Create a new column 'text_without_sensitive_data'
df['text_without_sensitive_data'] = df.apply(remove_sensitive_data, axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Print text without sensitive data for a maximum of 5 rows where it's different from the original text
count = 0
for index, row in df.iterrows():
    if row['text'] != row['text_without_sensitive_data']:
        print(row['text_without_sensitive_data'])
        print()
        count += 1
        if count >= 5:
            break

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
sensitive_data_encrypted_df = df.drop(columns=['text', 'text_without_sensitive_data'], errors='ignore')

sensitive_data_removed_df = df.drop(columns=['text', 'encrypted_text'])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## 7.2. Retrait des stopwords et des caractères spéciaux, et normalisation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from nltk.corpus import stopwords
import nltk
import re

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
df['text'] = df.apply(lambda row: remove_stopwords(row['text'], row['language']), axis=1)
df['text_without_sensitive_data'] = df.apply(lambda row: remove_stopwords(row['text_without_sensitive_data'], row['language']), axis=1)
df['encrypted_text'] = df.apply(lambda row: remove_stopwords(row['encrypted_text'], row['language']), axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Function to remove special characters from text
def remove_special_characters(text):
    # Use regex to remove special characters
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Apply the function to the 'text' column
df['text'] = df['text'].apply(remove_special_characters)
df['text_without_sensitive_data'] = df['text_without_sensitive_data'].apply(remove_special_characters)
df['encrypted_text'] = df['encrypted_text'].apply(remove_special_characters)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Normalize the text
df['text'] = df['text'].str.lower()
df['text_without_sensitive_data'] = df['text_without_sensitive_data'].str.lower()
df['encrypted_text'] = df['encrypted_text'].str.lower()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Create one dataframe with encrypted sensitive data
sensitive_data_encrypted_df = df.drop(columns=['text', 'text_without_sensitive_data'], errors='ignore')
sensitive_data_encrypted_df = sensitive_data_encrypted_df.rename(columns={'encrypted_text': 'text'})

# Create one dataframe with removed sensitive data
sensitive_data_removed_df = df.drop(columns=['text', 'encrypted_text'])
sensitive_data_removed_df = sensitive_data_removed_df.rename(columns={'text_without_sensitive_data': 'text'})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
cleaned_tweets = dataiku.Dataset("cleaned_tweets")
cleaned_tweets.write_with_schema(df)

sensitive_data_removed = dataiku.Dataset("sensitive_data_removed")
sensitive_data_removed.write_with_schema(sensitive_data_removed_df)

sensitive_data_encrypted = dataiku.Dataset("sensitive_data_encrypted")
sensitive_data_encrypted.write_with_schema(sensitive_data_encrypted_df)


