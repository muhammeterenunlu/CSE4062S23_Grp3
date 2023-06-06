from sklearn.preprocessing import OneHotEncoder, Binarizer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Assuming data is your DataFrame
data = pd.read_csv('tickets.csv')

# Handle missing values - this is just an example,
# you might need to handle missing values differently based on the nature of your features
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns = data.columns)

# Split features and target variable
X = data_filled.drop('ISSUE_CATEGORY', axis=1)
y = data_filled['ISSUE_CATEGORY']

# Encode categorical features to binary using one-hot encoding
categorical_features = ['PRIORITY', 'URGENCY', 'COMPONENT', 'IMPACT', 'ASSIGNEE', 'ISSUE_TYPE']
one_hot_encoder = OneHotEncoder(drop='first')
one_hot_encoded = one_hot_encoder.fit_transform(X[categorical_features])

# Convert one-hot encoded data to DataFrame and add non-categorical features
X_one_hot = pd.DataFrame(one_hot_encoded.toarray(), columns = one_hot_encoder.get_feature_names_out(categorical_features))
non_categorical_features = ['Total_Assignee', 'Total_Worklog_Assginee', 'Total_Log_Hours_Assignee', 'COMMENTOR_COUNT', 'COMMENT_COUNT']
X_processed = pd.concat([X_one_hot, X[non_categorical_features].reset_index(drop=True)], axis=1)

# Binarize numerical features using a threshold - this is just an example,
# you might need to select appropriate thresholds for each feature based on the nature of your data
binarizer = Binarizer(threshold=0.0) # Change the threshold based on your understanding of the data
X_binarized = binarizer.fit_transform(X_processed[non_categorical_features])
X_processed[non_categorical_features] = X_binarized

# Now X_processed is ready to be used with the Bernoulli Naive Bayes algorithm
