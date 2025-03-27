# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Packages

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import tempfile
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from edf_commons.modelling import preprocess_data

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Global variables

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
project = dataiku.Project()
variables = project.get_variables()
MODELS_PATH = variables["standard"]["models_path"]
MODEL_FOLDER_ID = variables["standard"]["model_folder_id"]
MODELS_DATA_FOLDER = dataiku.Folder(MODEL_FOLDER_ID)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Input

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
tweets_train = dataiku.Dataset("tweets_train")
train_df = tweets_train.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Machine Learning

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
def train_model_with_tfidf_stratified_kfold(df: pd.DataFrame, n_splits: int = 5, max_features: int = 5000)->tuple[pd.DataFrame, list, LogisticRegression]:
    """
    Applies a logistic regression model using TF-IDF features and evaluates it with stratified K-fold cross-validation.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset.
    n_splits (int): The number of splits for stratified K-fold cross-validation. Default is 5.
    max_features (int): The maximum number of features to consider for the TF-IDF vectorizer. Default is 5000.

    Returns:
    tuple: A tuple containing the list of accuracies, classification reports, ROC AUC scores for each fold, and the trained model.
    """

    X_combined, y, tfidf = preprocess_data(df, None)

    # Step 4: Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    reports = []
    roc_aucs = []

    for train_index, test_index in skf.split(X_combined, y):

        X_train, X_test = X_combined[train_index], X_combined[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Define the model
        model = LogisticRegression(max_iter=1000)
        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        roc_aucs.append(roc_auc)
        reports.append(report)

    metrics_df = pd.DataFrame({
        'Mean Accuracy': [np.mean(accuracies)],
        'Mean Precision': [np.mean(precisions)],
        'Mean Recall': [np.mean(recalls)],
        'Mean F1 Score': [np.mean(f1s)],
        'Mean ROC AUC': [np.mean([auc for auc in roc_aucs if auc is not None])],
        'Model Name': ['Logistic Regression'],
        'Date and Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    })

    return metrics_df, reports, model, tfidf

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
lr_metrics_df, reports, lr_model, tfidf = train_model_with_tfidf_stratified_kfold(train_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Save pickle

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
lr_artefact_name = f"lr_{date_time}"

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Pipeline for encrypted data
artefact_pickle_name = f"{lr_artefact_name}.pkl"
model_pickle_path = os.path.join(MODELS_PATH, artefact_pickle_name)

with tempfile.TemporaryDirectory() as temp_dir:

    local_file_path = os.path.join(temp_dir, artefact_pickle_name)

    with open(local_file_path, 'wb') as file:
        pickle.dump(lr_model, file)

    MODELS_DATA_FOLDER.upload_file(model_pickle_path, local_file_path)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Output

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
metrics_per_fold_df = pd.DataFrame()
metrics_per_fold_df["model"] = lr_artefact_name
metrics_per_fold_df["date_time"] = date_time

for i, report in enumerate(reports):
    if isinstance(report, dict):
        # Remove 'accuracy', 'macro avg', and 'weighted avg' from the report
        report.pop('accuracy', None)
        report.pop('macro avg', None)
        report.pop('weighted avg', None)

        report_df = pd.DataFrame.from_dict(report).transpose()
        report_df['class'] = report_df.index  # Save the key of each dictionary into a new column called "class"
        report_df['Fold'] = i + 1  # Add the fold number to the DataFrame
        metrics_per_fold_df = pd.concat([metrics_per_fold_df, report_df], ignore_index=True)
    else:
        print(f"Warning: Report for Fold {i+1} is not a dictionary and cannot be converted to a DataFrame.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
lr_metrics = dataiku.Dataset("lr_metrics")
lr_metrics.write_with_schema(lr_metrics_df)

metrics_per_fold = dataiku.Dataset("metrics_per_fold")
metrics_per_fold.write_with_schema(metrics_per_fold_df)