# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Evaluate

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pandas as pd

data_path = 'data/tweets_encryption_test.csv'
data = pd.read_csv(data_path)

# Convert the data to a DataFrame
eval_df = pd.DataFrame(data)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import tempfile

# Convert the data to a DataFrame
# tweets_eval = dataiku.Dataset("tweets_eval")
# eval_df = tweets_eval.get_dataframe()

# Get latest model in 
# folder = dataiku.Folder("20Z7bzGW")
PICKLE_MODELS_PATH = 'pickle_models/'

# Get the list of all files in the directory
files = os.listdir(PICKLE_MODELS_PATH)

# Filter out non-pickle files and sort by modification time
pickle_files = [f for f in files if f.endswith('.pkl')]
pickle_files.sort(key=lambda f: os.path.getmtime(os.path.join(PICKLE_MODELS_PATH, f)))

# Get the latest model file
latest_model_file = pickle_files[-1] if pickle_files else None

# Construct the full path to the latest model
model_path = os.path.join(PICKLE_MODELS_PATH, latest_model_file) if latest_model_file else None

# Load the latest model
with tempfile.TemporaryDirectory() as temp_directory_name:

        # local_file_path = temp_directory_name + "/" + model_name
        local_file_path = model_path

        # Copy file from remote to local
        with folder.get_download_stream(model_path) as f_remote, open(local_file_path,'wb') as f_local:
            shutil.copyfileobj(f_remote, f_local)

        # Load the pipeline
        with open(local_file_path, 'rb') as file:
            pipeline = pickle.load(file)

model_name = model_path.split("/")[-1].split(".pkl")[0]


X_eval, y_eval, _ = preprocess_data(eval_df, tfidf)

eval_prediction_results_df = eval_df.copy()

eval_prediction_results_df['predicted'] = lr_model.predict(X_eval)
eval_prediction_results_df['probability'] = lr_model.predict_proba(X_eval)[:, 1]
eval_prediction_results_df['correct_prediction'] = results_df['label'] == results_df['predicted']

# Calculate evaluation metrics
accuracy = accuracy_score(y_eval, results_df['predicted'])
precision = precision_score(y_eval, results_df['predicted'], average='weighted')
recall = recall_score(y_eval, results_df['predicted'], average='weighted')
f1 = f1_score(y_eval, results_df['predicted'], average='weighted')
roc_auc = roc_auc_score(y_eval, lr_model.predict_proba(X_eval), multi_class='ovr', average='weighted')

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
# # Create output datasets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
eval_prediction_results = dataiku.Dataset("eval_prediction_results")
eval_prediction_results.write_with_schema(eval_prediction_results_df)

eval_metrics = dataiku.Dataset("eval_metrics")
eval_metrics.write_with_schema(eval_metrics_df)