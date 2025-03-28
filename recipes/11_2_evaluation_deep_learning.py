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
LABEL_MAPPING = 
INDEX_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

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
y = eval_df['label']
X_eval, y_eval = preprocess_data_for_dl(eval_df[['tweet_length_chars', 'tweet_length_words', 'text']], y)

# Predict
eval_prediction_results_df = eval_df.copy()

# Create class probabilities columns
class_probabilities = nn_model.predict(X_eval)
for i in range(class_probabilities.shape[1]):
    class_name = INDEX_MAPPING[i]
    eval_prediction_results_df[f'{class_name}_probability'] = class_probabilities[:, i]
    
# Create the prediction column
eval_prediction_results_df['max_probability'] = class_probabilities.max(axis=1)
eval_prediction_results_df['prediction'] = eval_prediction_results_df.filter(like='_probability').idxmax(axis=1).str.replace('_probability', '')
eval_prediction_results_df['prediction_label_id'] = eval_prediction_results_df['prediction'].map(LABEL_MAPPING)

# Move label column next to prediction
cols = eval_prediction_results_df.columns.tolist()
cols.remove('label')
cols.insert(cols.index('prediction'), 'label')
eval_prediction_results_df = eval_prediction_results_df[cols]

eval_prediction_results_df['correct_prediction'] = eval_prediction_results_df['label'] == eval_prediction_results_df['prediction']

# Calculate evaluation metrics
accuracy = accuracy_score(y_eval, eval_prediction_results_df['prediction_label_id'])
precision = precision_score(y_eval, eval_prediction_results_df['prediction_label_id'], average='weighted')
recall = recall_score(y_eval, eval_prediction_results_df['prediction_label_id'], average='weighted')
f1 = f1_score(y_eval, eval_prediction_results_df['prediction_label_id'], average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_eval), pd.get_dummies(eval_prediction_results_df['prediction_label_id']), multi_class='ovr', average='weighted')

# Drop max_probability
eval_prediction_results_df = eval_prediction_results_df.drop(columns=['max_probability', 'prediction_label_id'], errors='ignore')

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