# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
from datetime import datetime

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Dataset tweets_encryption_train renamed to tweets_train by anne-soline.guilbert-ly@dataiku.com on 2025-03-27 11:53:12
tweets_encryption_train = dataiku.Dataset("tweets_train")
tweets_encryption_train_df = tweets_encryption_train.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def apply_and_evaluate_model_with_tfidf_stratified_kfold(df: pd.DataFrame, label_col: str = 'label', n_splits: int = 5, max_features: int = 5000):
    """
    Applies a logistic regression model using TF-IDF features and evaluates it with stratified K-fold cross-validation.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset.
    label_col (str): The name of the column containing the target labels. Default is 'label'.
    n_splits (int): The number of splits for stratified K-fold cross-validation. Default is 5.
    max_features (int): The maximum number of features to consider for the TF-IDF vectorizer. Default is 5000.

    Returns:
    tuple: A tuple containing the list of accuracies, classification reports, ROC AUC scores for each fold, and the trained model.
    """

    # Step 2: Split dataset into features and target
    # Handle date, user, and language as dummy variables
    X = pd.get_dummies(df[['date', 'user', 'language']], drop_first=True)

    # Handle numerical columns with standard scaling
    numerical_cols = ['tweet_length_chars', 'tweet_length_words', 'repetitive_letters',
                      'mention_only', 'unreadable', 'too_many_numbers']
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(df[numerical_cols].fillna(0))
    X_numerical = pd.DataFrame(X_numerical, columns=numerical_cols, index=df.index)

    # Concatenate all features
    X = pd.concat([X, X_numerical], axis=1)
    y = df[label_col]

    # Step 3: Apply TF-IDF transformation on the text column
    tfidf = TfidfVectorizer(
        min_df=0.01,  # Adjusted min_df to a lower value
        max_df=0.9,   # Adjusted max_df to a higher value
        ngram_range=(1, 1),
        stop_words=None
    )

    try:
        X_tfidf = tfidf.fit_transform(df['text'].fillna(''))
    except ValueError as e:
        print(f"Error during TF-IDF transformation: {e}")
        return None, None, None, None

    # Combine TF-IDF features with other features
    X_combined = np.hstack((X.values, X_tfidf.toarray()))

    # Step 4: Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
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
        report = classification_report(y_test, y_pred, output_dict=True)

        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except ValueError as e:
            print(f"Error during ROC AUC calculation: {e}")
            roc_auc = None

        accuracies.append(accuracy)
        reports.append(report)
        roc_aucs.append(roc_auc)

    # Feature importance plot
    feature_names = list(X.columns) + tfidf.get_feature_names_out()
    feature_importance = model.coef_[0]
    sorted_idx = np.argsort(feature_importance)

    plt.figure(figsize=(10, 8))
    feature_importance_plot = sns.barplot(
        y=np.array(feature_names)[sorted_idx],
        x=feature_importance[sorted_idx],
        orient='h'
    )
    plt.xlabel("Feature Importance")
    plt.title("Feature Importance Plot")
    plt.show()

    return np.mean(accuracies), reports, np.mean([auc for auc in roc_aucs if auc is not None]), model, feature_importance_plot

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Train and evaluate tweets_encryption_train_df

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
accuracy_encrypted, report_encrypted, roc_auc_encrypted, model_encrypted_data, feature_importance_plot_encrypted = apply_and_evaluate_model_with_tfidf_stratified_kfold(tweets_encryption_train_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if accuracy_encrypted is not None:
    encrypted_metrics_df = pd.DataFrame({
        'Metric': ['Average Accuracy', 'Average ROC AUC'],
        'Value': [accuracy_encrypted, roc_auc_encrypted]
    })
    print(encrypted_metrics_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
encrypted_metrics_per_fold_df = pd.DataFrame()

for i, report in enumerate(report_encrypted):
    if isinstance(report, dict):
        # Remove 'accuracy', 'macro avg', and 'weighted avg' from the report
        report.pop('accuracy', None)
        report.pop('macro avg', None)
        report.pop('weighted avg', None)

        report_df = pd.DataFrame.from_dict(report).transpose()
        report_df['class'] = report_df.index  # Save the key of each dictionary into a new column called "class"
        report_df['Fold'] = i + 1  # Add the fold number to the DataFrame
        encrypted_metrics_per_fold_df = pd.concat([encrypted_metrics_per_fold_df, report_df], ignore_index=True)
    else:
        print(f"Warning: Report for Fold {i+1} is not a dictionary and cannot be converted to a DataFrame.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
artefact_name_encrypted = f"lr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
encrypted_metrics_per_fold_df["model"] = artefact_name_encrypted

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Save pickle

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Pipeline for encrypted data
pipeline_local_path_encrypted = f"{artefact_name_encrypted}.pkl"
pipeline_remote_path_encrypted = f"{artefact_name_encrypted}.pkl"
remote_output_folder_encrypted = dataiku.Folder("VQ6fLov2")

with tempfile.TemporaryDirectory() as local_tmp_dir_encrypted:

    local_file_path_encrypted = os.path.join(local_tmp_dir_encrypted, pipeline_local_path_encrypted)

    with open(local_file_path_encrypted, 'wb') as file:
        pickle.dump(model_encrypted_data, file)

    remote_output_folder_encrypted.upload_file(pipeline_remote_path_encrypted, local_file_path_encrypted)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# ## Save artefacts

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Artefacts
fi_local_path_encrypted = f"{artefact_name_encrypted}_feature_importance.png"
fi_remote_path_encrypted = f"{artefact_name_encrypted}_feature_importance.png"
output_folder_encrypted = dataiku.Folder("2T1uAdOy")

with tempfile.TemporaryDirectory() as tmp_dir_name_encrypted:
    local_file_path_encrypted = os.path.join(tmp_dir_name_encrypted, fi_local_path_encrypted)
    fig = feature_importance_plot_encrypted.get_figure()
    fig.savefig(fi_local_path_encrypted)
    output_folder_encrypted.upload_file(fi_remote_path_encrypted, fi_local_path_encrypted)
    plt.close(fig)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Create output datasets

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
# Dataset encrypted_metrics renamed to lr_metrics by anne-soline.guilbert-ly@dataiku.com on 2025-03-27 12:02:44
encrypted_metrics = dataiku.Dataset("lr_metrics")
encrypted_metrics.write_with_schema(encrypted_metrics_df)

encrypted_metrics_per_fold = dataiku.Dataset("encrypted_metrics_per_fold")
encrypted_metrics_per_fold.write_with_schema(encrypted_metrics_per_fold_df)
