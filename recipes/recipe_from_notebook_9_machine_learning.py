# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Example: load a DSS dataset as a Pandas dataframe
prepared_tweets_encryption = dataiku.Dataset("prepared_tweets_encryption")
df = prepared_tweets_encryption.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def apply_and_evaluate_model_with_tfidf_stratified_kfold(df, label_col='label', n_splits=5, max_features=5000):

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
        return None, None, None

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

    return np.mean(accuracies), reports, np.mean([auc for auc in roc_aucs if auc is not None])

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Apply to prepared_tweets_encryption

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Apply the function to each DataFrame
accuracy_encrypted, report_encrypted, roc_auc_encrypted = apply_and_evaluate_model_with_tfidf_stratified_kfold(prepared_tweets_encryption)

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

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # Apply to prepared_tweets_removal

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
accuracy_removed, report_removed, roc_auc_removed = apply_and_evaluate_model_with_tfidf_stratified_kfold(prepared_tweets_removal)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
if accuracy_removed is not None:
    removed_metrics_df = pd.DataFrame({
        'Metric': ['Average Accuracy', 'Average ROC AUC'],
        'Value': [accuracy_removed, roc_auc_removed]
    })
    print(removed_metrics_df)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
removed_metrics_per_fold_df = pd.DataFrame()

for i, report in enumerate(report_removed):
    if isinstance(report, dict):
        # Remove 'accuracy', 'macro avg', and 'weighted avg' from the report
        report.pop('accuracy', None)
        report.pop('macro avg', None)
        report.pop('weighted avg', None)
        
        report_df = pd.DataFrame.from_dict(report).transpose()
        report_df['class'] = report_df.index  # Save the key of each dictionary into a new column called "class"
        report_df['Fold'] = i + 1  # Add the fold number to the DataFrame
        removed_metrics_per_fold_df = pd.concat([removed_metrics_per_fold_df, report_df], ignore_index=True)
    else:
        print(f"Warning: Report for Fold {i+1} is not a dictionary and cannot be converted to a DataFrame.")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
encrypted_metrics = dataiku.Dataset("encrypted_metrics")
encrypted_metrics.write_with_schema(encrypted_metrics_df)

encrypted_metrics_per_fold = dataiku.Dataset("encrypted_metrics_per_fold")
encrypted_metrics_per_fold.write_with_schema(encrypted_metrics_per_fold_df)

removed_metrics = dataiku.Dataset("removed_metrics")
removed_metrics.write_with_schema(removed_metrics_df)

removed_metrics_per_fold = dataiku.Dataset("removed_metrics_per_fold")
removed_metrics_per_fold.write_with_schema(removed_metrics_per_fold_df)
