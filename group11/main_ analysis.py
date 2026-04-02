# Import the necessary libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.offline as pyo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
% matplotlib inline

# Read the dataset
df = pd.read_csv('transaction_dataset.csv')
pd.set_option('display.max_columns', None)
df.head()

# Drop the columns that are not needed
df.drop(['Unnamed: 0','Index','Address'],axis =1,inplace = True)
df.head()

# Print the count of each label
print("Count of each label:\n", df['FLAG'].value_counts())
print("Number of rows with undetermined fraud status:", df['FLAG'].isnull().sum())
df.shape

# Print the description of the data
df.describe()

# Print the number of null values per feature
print("Number of null values per feature:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()

# Print the number of unique values for each feature
print("Number of unique values for feature ERC20_most_rec_token_type:", df[' ERC20_most_rec_token_type'].nunique())
print("Number of unique values for feature ERC20 most sent token type:", df[' ERC20 most sent token type'].nunique())

# Print the duplicate rows
duplicate_rows = df[df.duplicated()]
duplicate_rows
df.drop_duplicates(inplace=True)
df_copy = df.copy()
df_copy.columns = df_copy.columns.str.strip()
for col in df_copy.columns:
    if df_copy[col].isnull().sum() == 829:
        df_copy[col].fillna(df_copy[col].mean(), inplace=True)

# Output columns that still have null values
print("Columns with remaining null values:")
print(df_copy.isnull().sum()[df_copy.isnull().sum() > 0])
print()

# Check value counts for the categorical feature
categorical_feature = 'ERC20_most_rec_token_type'
print(f"Value counts for feature {categorical_feature}:\n{df_copy[categorical_feature].value_counts()}")

# Clean categorical features - replace 0 values with null as 0 has no actual meaning in categorical features
categorical_features = ['ERC20_most_rec_token_type', 'ERC20 most sent token type']
df_copy[categorical_features] = df_copy[categorical_features].replace({'0': np.nan})

# Output the processed features
print("Processed features:")
print(df_copy[categorical_features])

# Check the null percentage of the specific feature
null_percentage = df_copy['ERC20 most sent token type'].isnull().sum() / len(df_copy)
print("Null percentage of feature ERC20 most sent token type:", null_percentage)

# Create a list to store features with zero variance
to_drop = list(df_copy.var(numeric_only=True)[df_copy.var(numeric_only=True) == 0].keys())
print("List of features with zero variance:", to_drop)

# Drop features with zero variance
df_copy.drop(to_drop, inplace=True, axis=1)
print("Data description after dropping features with zero variance:")
print(df_copy.describe())

# Find features with standard deviation close to zero
low_std_features = list(df_copy.std(numeric_only=True)[df_copy.std(numeric_only=True) < 0.001].keys())
print("List of features with standard deviation close to zero:", low_std_features)
df_copy.drop(['min value sent to contract','max val sent to contract','avg value sent to contract'\
              ,'total ether sent contracts']\
             ,axis =1 , inplace = True)
# Fill missing values for categorical features using mode
df_copy = df_copy.fillna(df_copy.mode().iloc[0])

# Output columns that still have null values
remaining_nulls = df_copy.isnull().sum()[df_copy.isnull().sum() > 0]
print("Columns with remaining null values:")
print(remaining_nulls)
df_copy[['ERC20_most_rec_token_type', 'ERC20 most sent token type']].columns

# Code for processing features with more than 50 categories (unique values)
# In this process, we will use Label Encoding to process features with more than 50 categories

# Create a copy for processing
df_encoded = df_copy.copy()

# Store features that need Label Encoding
label_encoding_features = []

# Store categorical features to be converted and their mapping dictionaries
categorical_features_dict = {}

# Iterate through the features to be processed
for feature in df_copy[['ERC20_most_rec_token_type', 'ERC20 most sent token type']].columns:
    if df_copy[feature].nunique() >= 50:  # If there are more than 50 distinct categories, we will use Label Encoding
        label_encoding_features.append(feature)
        categorical_features_dict[feature] = {}  # The dictionary will store original values as keys and sequential numbers (categories) as values
        i = 1  # Sequential number (category)
        
        # Map feature values to categories using a dictionary
        for sample in df_copy[feature]:
            if sample not in categorical_features_dict[feature].keys() and sample is not np.nan:  # We replace each value (excluding 'null') with the dictionary
                categorical_features_dict[feature][sample] = i
                i += 1

        # Apply Label Encoding using the mapping dictionary in the copied data and original data
        df_encoded[feature].replace(categorical_features_dict[feature], inplace=True)

# Output features for Label Encoding and the mapping dictionaries
print("Features requiring Label Encoding:", label_encoding_features)
print("Mapping dictionaries for categorical features:", categorical_features_dict)

from sklearn.decomposition import PCA

# Perform Principal Component Analysis
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(df_encoded.drop('FLAG', axis=1))

# Create Isolation Forest model
isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)

# Train the model and predict anomalies
isolation_forest_preds = isolation_forest_model.fit_predict(X_pca)

# Identify outliers
outliers_isolation_forest = isolation_forest_preds == -1

# Remove outliers identified by Isolation Forest from the original data
df_encoded_cleaned = df_encoded[~outliers_isolation_forest]

# Output statistical information of the data after removing outliers
print("Statistical information of data after removing outliers:")
print(df_encoded_cleaned.describe())

# Assume df_copy is the DataFrame containing the data
df_pca_cleaned_isolation = df_encoded_cleaned

# Keep the 'FLAG' column and perform PCA on other features
features_to_pca = df_pca_cleaned_isolation.drop('FLAG', axis=1)

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_to_pca)

# Perform dimensionality reduction using PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
features_pca = pca.fit_transform(features_scaled)

# Convert the reduced-dimensionality data back to a DataFrame and add the 'FLAG' column
df_pca = pd.DataFrame(data=features_pca, columns=['PC{}'.format(i) for i in range(1, pca.n_components_ + 1)])
df_pca['FLAG'] = df_pca_cleaned_isolation['FLAG'].reset_index(drop=True)

# Output the dimension-reduced DataFrame
print(df_pca)

# Calculate the correlation coefficient between features
corr = df_pca.corr()

# Filter features with high correlation based on the correlation coefficient
corr_df = corr[(corr > 0.6)].dropna(axis=1, thresh=2)
corr_df = corr_df[corr_df != 1]

# Find features with correlation higher than 0.9
to_drop = corr_df[corr_df > 0.9]
high_corr_features = to_drop.values[to_drop.values > 0]

# Find the features to be dropped
to_drop_features = to_drop.unstack().sort_values(kind="quicksort")[to_drop.unstack() > 0]

print("List of features to be dropped:")
print(to_drop_features)

# Check the correlation between features and the label, sorted in descending order
correlation_with_label = df_pca.corr()['FLAG'].sort_values(ascending=False)

# Output the correlation between features and the label
print("Correlation between features and label:")
print(correlation_with_label)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# To improve readability and understandability, we can split the code into two functions
def plot_correlation_heatmap(data, title):
    corr = data.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style('white'):
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.heatmap(corr, mask=mask, annot=False, cmap='magma', center=0, linewidths=0.1, square=True)
        plt.title(title)
        plt.show()

# Plot correlation matrices for fraudulent and non-fraudulent samples separately
fraud_sample = df_encoded_cleaned[df_encoded_cleaned['FLAG'] == 1]
non_fraud_sample = df_encoded_cleaned[df_encoded_cleaned['FLAG'] == 0]

plot_correlation_heatmap(fraud_sample, 'Fraudulent Correlation')
plot_correlation_heatmap(non_fraud_sample, 'Non-Fraudulent Correlation')

# Identify some outliers from the scatter plots plotted earlier
# Next, we will check if removing these outliers improves our model results
outliers_indices = df_copy[(df_copy['Received Tnx'] > 3000) & (df_copy['FLAG'] == 1) & (df_copy['Received Tnx'] < 4000)].index

# Print the indices of the found outliers
print("Indices of the found outliers:")
print(outliers_indices)

# Plot the boxplot and stripplot for the 'Avg min between received tnx' feature
sns.set(style='darkgrid')
def plot_boxplot_and_stripplot(data, title):
    plt.subplots(figsize=(14, 8))
    boxplot = sns.boxplot(data=data)
    stripplot = sns.stripplot(data=data, marker="o", alpha=0.3, color="blue")
    boxplot.axes.set_title(title, fontsize=16)
    boxplot.set_xlabel("Conditions", fontsize=14)
    boxplot.set_ylabel("Values", fontsize=14)
    plt.show()

# Plot the boxplot and stripplot for the 'Avg min between received tnx' feature
data1 = df_copy['Avg min between received tnx'][df_copy['FLAG'] == 0]
data2 = df_copy['Avg min between received tnx'][df_copy['FLAG'] == 1]
plot_boxplot_and_stripplot(data=[data1, data2], title="Distribution of Avg min between received tnx")

# Plot the boxplot and stripplot for the 'Time Diff between first and last (Mins)' feature
data1 = df_copy['Time Diff between first and last (Mins)'][df_copy['FLAG'] == 0]
data2 = df_copy['Time Diff between first and last (Mins)'][df_copy['FLAG'] == 1]
plot_boxplot_and_stripplot(data=[data1, data2], title="Distribution of Time Diff between first and last (Mins)")

# Plot the boxplot and stripplot for the 'Avg min between received tnx' feature
data1 = df_copy['Avg min between received tnx'][df_copy['FLAG'] == 0]
data2 = df_copy['Avg min between received tnx'][df_copy['FLAG'] == 1]
plot_boxplot_and_stripplot(data=[data1, data2], title="Distribution of Avg min between received tnx")

# Find outliers in the 'Time Diff between first and last (Mins)' feature
outliers_1 = df_copy[
    (df_copy['FLAG'] == 1) & 
    (df_copy['Time Diff between first and last (Mins)'] > 600000)
]
df_copy.drop(outliers_1.index, axis=0, inplace=True)

# Find outliers in the 'Avg min between received tnx' feature and delete them
outliers_2 = df_copy[
    (df_copy['FLAG'] == 1) & 
    (df_copy['Avg min between received tnx'] > 60000)
]
df_copy.drop(outliers_2.index, axis=0, inplace=True)

# Find outliers in the 'Received Tnx' feature
outliers_3 = df_copy[
    (df_copy['FLAG'] == 1) & 
    (df_copy['Received Tnx'] > 8000)
]

# Print the found outliers
print("Outliers found in the 'Time Diff between first and last (Mins)' feature:")
print(outliers_1)
print()

print("Outliers found in the 'Received Tnx' feature:")
print(outliers_3)

# Print the indices of the outliers in the 'Avg min between received tnx' feature
print("Indices of the outliers in the 'Avg min between received tnx' feature (fraudulent):")
print(df_copy['Avg min between received tnx'][
    (df_copy['FLAG'] == 0) & 
    (df_copy['Avg min between received tnx'] > 400000)
].index)
print()

# Print the indices of the outliers in the 'Avg min between received tnx' feature (non-fraudulent)
print("Indices of the outliers in the 'Avg min between received tnx' feature (non-fraudulent):")
print(df_copy['Avg min between received tnx'][
    (df_copy['FLAG'] == 1) & 
    (df_copy['Avg min between received tnx'] > 140000)
].index)

# Calculate and process outliers for each feature
for feature in df_copy.drop('FLAG', axis=1).columns:
    # Ensure the feature is of numeric type
    if np.issubdtype(df_copy[feature].dtype, np.number):
        IQR = np.percentile(df_copy[feature], 75) - np.percentile(df_copy[feature], 25)
        lower_limit = np.percentile(df_copy[feature], 25) - 1.5 * IQR
        upper_limit = np.percentile(df_copy[feature], 75) + 1.5 * IQR
        
        # Calculate the number of outliers
        outliers_count = ((df_copy[feature] > upper_limit) | (df_copy[feature] < lower_limit)).sum()
        
        # Calculate the proportion of outliers
        outlier_proportion = outliers_count / df_copy.shape[0]
        
        # Set the threshold for the maximum proportion of outliers
        max_outlier_proportion = 0.07
        
        if outlier_proportion <= max_outlier_proportion:
            print(f"Feature '{feature}' has an outlier proportion of {outlier_proportion:.2%}, will be considered.")
# Choose further action, such as removing outliers from the dataset
            df_copy = df_copy[~((df_copy[feature] > upper_limit) | (df_copy[feature] < lower_limit))]

# Create a copy (keep the original logic)
dropping_corr = df_encoded.copy()

# Key fix: only select numeric columns for correlation calculation (exclude string columns)
# numeric_only=True forces only numeric columns to be processed, avoiding string conversion errors
corr_matrix = df_copy.corr(numeric_only=True)

# Filter columns related to FLAG (first check if FLAG exists)
if 'FLAG' not in corr_matrix.columns:
    raise ValueError("The data frame does not contain the 'FLAG' column, please check if the column name is correct!")

# Sort columns by correlation with FLAG
corr_decision = pd.DataFrame(corr_matrix['FLAG'].sort_values(ascending=False))

# Filter features with low correlation (absolute value < 0.02)
low_corr_features = corr_decision[
    (corr_decision['FLAG'] < 0.02) & (corr_decision['FLAG'] > -0.02)
].index

# Delete features with low correlation (first check if features exist in dropping_corr)
# Avoid KeyError due to feature not existing
low_corr_features = [feat for feat in low_corr_features if feat in dropping_corr.columns]
dropping_encoder = dropping_corr.drop(low_corr_features, axis=1)

# Output result verification
print(f"Number of features with low correlation with FLAG (|r| < 0.02): {len(low_corr_features)}")
print("List of low correlation features:", low_corr_features)
print("\nNumber of columns in the dataset after deleting low correlation features:", dropping_encoder.shape[1])
print("Number of columns in the dataset before deleting low correlation features:", dropping_corr.shape[1])

df_pca
# Output the count of each label
print("Count of each label:\n", df_pca['FLAG'].value_counts())

# Check if there are rows with undetermined fraud status
print("Number of rows with undetermined fraud status:", df_pca['FLAG'].isnull().sum())

# Remove rows with undetermined fraud status
df_pca_cleaned = df_encoded_cleaned.dropna(subset=['FLAG'])

# Output the count of each label after removing rows with undetermined fraud status
print("Count of each label after removing rows with undetermined fraud status:\n", df_pca_cleaned['FLAG'].value_counts())

# Set the style
sns.set(style='white')

# Prepare data
X = df_pca_cleaned.drop(['FLAG'], axis=1)
Y = df_pca_cleaned['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Feature scaling (optional, depending on whether scaling is needed)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the logistic regression model
log_model = LogisticRegression(max_iter=10000, C=1.0, penalty='l2', solver='liblinear', class_weight=None)
log_model.fit(X_train_scaled, y_train)
y_pred = log_model.predict(X_test_scaled)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()  

import xgboost as xgb   

# Set the style
sns.set(style='white')

# Prepare data
X = df_pca_cleaned.drop(['FLAG'], axis=1)
Y = df_pca_cleaned['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Create the XGBoost classifier model
xgb_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    objective='binary:logistic', random_state=42
)

# Use the model to train and predict
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Set the style
sns.set(style='white')

# Prepare data
X = df_pca_cleaned.drop(['FLAG'], axis=1)
Y = df_pca_cleaned['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Create the random forest classifier model
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42
)

# Use the model to train and predict
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style='white')

# Prepare data
X = df_pca_cleaned.drop(['FLAG'], axis=1)
Y = df_pca_cleaned['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the SVM classifier model
svm_model = SVC(C=10, gamma=1, kernel='rbf', random_state=42)

# Use the model to train and predict
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style='white')

# Prepare data
X = df_pca_cleaned.drop(['FLAG'], axis=1)
Y = df_pca_cleaned['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the naive Bayes classifier model
nb_model = GaussianNB()

# Use the model to train and predict
nb_model.fit(X_train_scaled, y_train)
y_pred = nb_model.predict(X_test_scaled)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Easy to strengthen the weight of error samples
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style
sns.set(style='white')

# Prepare data
X = dropping_corr.drop(['FLAG'], axis=1)
Y = dropping_corr['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the AdaBoost classifier model
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Use the model to train
adaboost_model.fit(X_train_scaled, y_train)

# Use the model to predict
y_pred = adaboost_model.predict(X_test_scaled)

# Output the classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

