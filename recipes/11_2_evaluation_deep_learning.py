# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku

import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from edf_commons.modelling import preprocess_data_for_dl

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
DL_MODEL_FOLDER_ID = variables["standard"]["dl_model_folder_id"]
DL_MODELS_DATA_FOLDER = dataiku.Folder(DL_MODEL_FOLDER_ID)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Input

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tweets_eval = dataiku.Dataset("tweets_eval")
eval_df = tweets_eval.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Get the Logistic Regression model
dl_model_path = DL_MODELS_DATA_FOLDER.list_paths_in_partition()[-1]
nn_name = dl_model_path.split("/")[-1].split(".pkl")[0]

# load latest Deep Learning model
with tempfile.TemporaryDirectory() as temp_directory_name:

    local_file_path = temp_directory_name + "/" + nn_name

    # Copy file from remote to local
    with DL_MODELS_DATA_FOLDER.get_download_stream(dl_model_path) as f_remote, open(local_file_path,'wb') as f_local:
        shutil.copyfileobj(f_remote, f_local)

    # Load the pipeline
    with open(local_file_path, 'rb') as file:
        nn_model = pickle.load(file)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Evaluation

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Preprocess the data
X_eval, y_eval = preprocess_data_for_dl(eval_df, y_eval)

# Predict
eval_prediction_results_df = eval_df.copy()
eval_prediction_results_df['predicted'] = nn_model.predict(X_eval)
eval_prediction_results_df['probability'] = nn_model.predict_proba(X_eval)[:, 1]
eval_prediction_results_df['correct_prediction'] = eval_prediction_results_df['label'] == eval_prediction_results_df['predicted']

# Calculate evaluation metrics
accuracy = accuracy_score(y_eval, eval_prediction_results_df['predicted'])
precision = precision_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
recall = recall_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
f1 = f1_score(y_eval, eval_prediction_results_df['predicted'], average='weighted')
roc_auc = roc_auc_score(y_eval, nn_model.predict_proba(X_eval), multi_class='ovr', average='weighted')

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
    'Model Name': ['Neural Network'],
    'Date and Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
})

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
eval_prediction_results = dataiku.Dataset("dl_eval_prediction_results")
eval_prediction_results.write_with_schema(eval_prediction_results_df)

eval_metrics = dataiku.Dataset("dl_eval_metrics")
eval_metrics.write_with_schema(eval_metrics_df)
