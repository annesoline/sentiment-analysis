# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from edf_commons.modelling import preprocess_data

import tempfile
from datetime import datetime
import shutil
import pickle
import os

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Global variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
project = dataiku.Project()
variables = project.get_variables()
MODELS_PATH = variables["standard"]["models_path"]
MODEL_FOLDER_ID = variables["standard"]["model_folder_id"]
MODELS_DATA_FOLDER = dataiku.Folder(MODEL_FOLDER_ID)
LABEL_MAPPING = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
INDEX_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Input

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tweets_eval = dataiku.Dataset("tweets_eval")
eval_df = tweets_eval.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the TFIDF model
tfidf_path = [path for path in MODELS_DATA_FOLDER.list_paths_in_partition() if "/tfidf/" in path][-1]
tfidf_name = tfidf_path.split("/")[-1].split(".pkl")[0]

# load latest TFIDF model
with tempfile.TemporaryDirectory() as temp_directory_name:

    local_file_path = temp_directory_name + "/" + tfidf_name

    # Copy file from remote to local
    with MODELS_DATA_FOLDER.get_download_stream(tfidf_path) as f_remote, open(local_file_path,'wb') as f_local:
        shutil.copyfileobj(f_remote, f_local)

    # Load the pipeline
    with open(local_file_path, 'rb') as file:
        tfidf = pickle.load(file)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the Logistic Regression model
lr_path = [path for path in MODELS_DATA_FOLDER.list_paths_in_partition() if "/lr/" in path][-1]
lr_name = MODELS_PATH.split("/")[-1].split(".pkl")[0]

# load latest TFIDF model
with tempfile.TemporaryDirectory() as temp_directory_name:

    local_file_path = temp_directory_name + "/" + lr_name

    # Copy file from remote to local
    with MODELS_DATA_FOLDER.get_download_stream(lr_path) as f_remote, open(local_file_path,'wb') as f_local:
        shutil.copyfileobj(f_remote, f_local)

    # Load the pipeline
    with open(local_file_path, 'rb') as file:
        lr_model = pickle.load(file)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Evaluation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preprocess the data
X_eval, y_eval, _ = preprocess_data(eval_df, tfidf)

# Predict
eval_prediction_results_df = eval_df.copy()
eval_prediction_results_df['predicted'] = lr_model.predict(X_eval)
eval_prediction_results_df['probability'] = lr_model.predict_proba(X_eval)[:, 1]
eval_prediction_results_df['correct_prediction'] = eval_prediction_results_df['label'] == eval_prediction_results_df['predicted']

# Calculate evaluation metrics
accuracy = accuracy_score(y_eval, eval_prediction_results_df['predicted'])
precision = precision_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
recall = recall_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
f1 = f1_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
roc_auc = roc_auc_score(y_eval, lr_model.predict_proba(X_eval), multi_class='ovr', average='weighted')

# Remap the predicted labels from int to string
eval_prediction_results_df['predicted'] = eval_prediction_results_df['predicted'].map(INDEX_MAPPING)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

eval_metrics_df = pd.DataFrame({
    'Mean Accuracy': [accuracy],
    'Mean Precision': [precision],
    'Mean Recall': [recall],
    'Mean F1 Score': [f1],
    'Mean ROC AUC': [roc_auc],
    'Model Name': ['Logistic Regression'],
    'Date and Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
eval_prediction_results = dataiku.Dataset("eval_prediction_results")
eval_prediction_results.write_with_schema(eval_prediction_results_df)

eval_metrics = dataiku.Dataset("eval_metrics")
eval_metrics.write_with_schema(eval_metrics_df)