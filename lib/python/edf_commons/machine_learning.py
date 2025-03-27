import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import dataiku
import tempfile
import os
import pickle

DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TFIDF_PATH = "/tfidf"
MODELS_DATA_FOLDER = dataiku.Folder("VQ6fLov2")
MODELS_PATH = "/lr"

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

                MODELS_DATA_FOLDER.upload_file(path, local_file_path)

        except ValueError as e:
            print(f"Error during TF-IDF transformation: {e}")
            return None, None, None, None
    else:
        X_tfidf = tfidf.transform(df['text'].fillna(''))

    # Combine TF-IDF features with other features
    X_combined = np.hstack((X.values, X_tfidf.toarray()))

    return X_combined, y, tfidf