# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu

import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay  
from sklearn.metrics import classification_report  

import matplotlib.pyplot as plt

from datetime import datetime
import os
import tempfile
import pickle

from edf_commons.modelling import preprocess_data_for_dl

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
LABEL_MAPPING = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
INDEX_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Global variables
project = dataiku.Project()
variables = project.get_variables()
ARTEFACTS_FOLDER_ID = variables["standard"]["artefacts_folder_id"]
CONFUSION_MATRICES_FOLDER_ID = variables["standard"]["confusion_matrices_path"]
EPOCHS_PERF_FOLDER_ID = variables["standard"]["epochs_perf_path"]
DL_MODELS_FOLDER_ID = variables["standard"]["df_model_folder_id"]
DL_MODELS_DATA_FOLDER = dataiku.Folder(DL_MODELS_FOLDER_ID)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Input

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tweets_train = dataiku.Dataset("tweets_train")
df = tweets_train.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Deep Learning

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preprocess the data
y = df['label']
X, y = preprocess_data_for_dl(df[['tweet_length_chars', 'tweet_length_words', 'text']], y)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def save_image(img_name: str, folder_path: str)->None:
    """Save the image of a graph recently plotted.
    
    ----------
    Parameters
        img_name: str
            Name of the image given to the png file.
        folder_path: str
            Path to the sub folder within the managed folder.
    """

    # Artefacts
    fig_name = f"{img_name}_{DATE_TIME}.png"
    output_folder = dataiku.Folder(ARTEFACTS_FOLDER_ID)
    output_folder_path = os.path.join(folder_path, fig_name)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        local_file_path = os.path.join(tmp_dir_name, fig_name)
        plt.savefig(fig_name)
        output_folder.upload_file(output_folder_path, fig_name)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def apply_and_evaluate_deep_learning_model(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, tf.keras.callbacks.History, pd.DataFrame]:
    """
    Applies a deep learning model to the preprocessed data and evaluates its performance.

    Parameters:
    X (pd.DataFrame): The preprocessed data including numerical, categorical, and text data.
    y (pd.Series): The labels.

    Returns:
    pd.DataFrame: A DataFrame containing the loss and accuracy of the model on the test data.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = len(y.unique())

    # Convert processed data to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

    # Batch the datasets
    train_dataset = train_dataset.batch(32)
    test_dataset = test_dataset.batch(32)

    # Initialize the model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Sigmoid for binary classification (positive/negative sentiment)
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset, verbose=0)

    # Obtenir les prédictions
    y_pred = model.predict(X_test.values)
    y_pred_classes = y_pred.argmax(axis=1)

    # Afficher la matrice
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the image
    save_image("confusion_matrix", CONFUSION_MATRICES_FOLDER_ID)

    # Générer le rapport
    report_dict = classification_report(y_test, y_pred_classes, output_dict=True, target_names=y_test.unique())
    print(report_dict)

    # Le convertir en DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save metrics into a DataFrame
    metrics_df = pd.DataFrame({'average_loss': [loss], 'average_accuracy': [accuracy]})
    
    return metrics_df, history, report_df, model

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Evaluate model on encrypted data
metrics, history, report_df, dl_model = apply_and_evaluate_deep_learning_model(X, y)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Save pickle

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Pipeline for encrypted data
nn_artefact_name = f"nn_{DATE_TIME}"
artefact_pickle_name = f"{nn_artefact_name}.pkl"

with tempfile.TemporaryDirectory() as temp_dir:

    local_file_path = os.path.join(temp_dir, artefact_pickle_name)

    with open(local_file_path, 'wb') as file:
        pickle.dump(dl_model, file)

    DL_MODELS_DATA_FOLDER.upload_file(artefact_pickle_name, local_file_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Outputs

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# remove last metrics
report_df = report_df.iloc[:-3]
# add model name column
report_df["model"] = nn_artefact_name
# add date time column
report_df["date_time"] = DATE_TIME
# create label column by remapping the names
report_df['label'] = report_df.index.map(INDEX_MAPPING)
for col in metrics.columns:
    report_df[col] = metrics[col].iloc[0]
    
# drop the indexes
report_df.reset_index(drop=True, inplace=True)

# Recipe outputs
dl_metrics = dataiku.Dataset("dl_metrics")
dl_metrics.write_with_schema(report_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
plot = pd.DataFrame(history.history).plot()
fig = plot.get_figure()
name = f"epochs_perf_evolution_{DATE_TIME}.png"
save_image("epochs_perf_evolution_", EPOCHS_PERF_FOLDER_ID)
