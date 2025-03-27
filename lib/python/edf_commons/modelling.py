import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from datetime import datetime
import dataiku
import tempfile
import os
import pickle
import numpy as np

DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LABEL_MAPPING = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}

# Import project variables
project = dataiku.Project()
variables = project.get_variables()
MODELS_PATH = variables["standard"]["models_path"]
TFIDF_PATH = variables["standard"]["tfidf_path"]
MODEL_FOLDER_ID = variables["standard"]["model_folder_id"]

def preprocess_data_for_dl(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the data by handling categorical variables with one-hot encoding,
    numerical columns with standard scaling, and text data with tokenization and padding.

    Parameters:
    X (pd.DataFrame): The data to preprocess.
    y (pd.Series): The label to encode.

    Returns:
    pd.DataFrame: The processed data including numerical, categorical, and text data.
    pd.Series: The processed label.
    """

    # Handle numerical columns with standard scaling
    numerical_features = ['tweet_length_chars', 'tweet_length_words']
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(X[numerical_features])

    # Handle text data with tokenization and padding
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X['text'].astype(str))  # Ensure text data is string
    X_text = tokenizer.texts_to_sequences(X['text'].astype(str))  # Ensure text data is string
    X_text = pad_sequences(X_text, maxlen=100)

    # Concatenate processed categorical, numerical, and text features
    X_processed = pd.concat([pd.DataFrame(X_numerical, columns=numerical_features), pd.DataFrame(X_text)], axis=1)

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y))

    return X_processed, y


def preprocess_data(df: pd.DataFrame, tfidf: TfidfVectorizer, label_col: str = 'label') -> tuple[pd.DataFrame, pd.Series]:

    features = ['tweet_length_chars', 'tweet_length_words', 'repetitive_letters',
                      'mention_only', 'unreadable', 'too_many_numbers']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features].fillna(0))
    X = pd.DataFrame(X, columns=features, index=df.index)

    # Encode label
    y = df[label_col]
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y))

    # Step 3: Apply TF-IDF transformation on the text column
    if tfidf is None:
        tfidf = TfidfVectorizer(
            min_df=0.01,  # Adjusted min_df to a lower value
            max_df=0.9,   # Adjusted max_df to a higher value
            ngram_range=(1, 1),
            stop_words=None
        )

        try:
            # Fit transform
            X_tfidf = tfidf.fit_transform(df['text'].fillna(''))

            # Save tfidf for evaluation
            tfidf_pickle_name = f"tfidf_{DATE_TIME}.pkl" 
            path = os.path.join(TFIDF_PATH, tfidf_pickle_name)

            with tempfile.TemporaryDirectory() as temp_dir:

                local_file_path = os.path.join(temp_dir, tfidf_pickle_name)

                with open(local_file_path, 'wb') as file:
                    pickle.dump(tfidf, file)
                    
                MODELS_DATA_FOLDER = dataiku.Folder(MODEL_FOLDER_ID)
                MODELS_DATA_FOLDER.upload_file(path, local_file_path)

        except ValueError as e:
            print(f"Error during TF-IDF transformation: {e}")
            return None, None, None, None
    else:
        X_tfidf = tfidf.transform(df['text'].fillna(''))

    # Combine TF-IDF features with other features
    X_combined = np.hstack((X.values, X_tfidf.toarray()))

    return X_combined, y, tfidf