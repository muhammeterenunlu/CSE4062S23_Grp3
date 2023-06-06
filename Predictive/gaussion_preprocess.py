from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Assuming data is your DataFrame
data = pd.read_csv('your_data.csv')

# Handle missing values - this is just an example,
# you might need to handle missing values differently based on the nature of your features
imputer = SimpleImputer(strategy='most_frequent')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns = data.columns)

# Split features and target variable
X = data_filled.drop('ISSUE_CATEGORY', axis=1)
y = data_filled['ISSUE_CATEGORY']

# Encode categorical features to ordinal values
categorical_features = ['PRIORITY', 'URGENCY', 'COMPONENT', 'IMPACT', 'ASSIGNEE', 'ISSUE_TYPE']
ordinal_encoder = OrdinalEncoder()
X[categorical_features] = ordinal_encoder.fit_transform(X[categorical_features])

# Standardize numerical features
numerical_features = ['Total_Assignee', 'Total_Worklog_Assginee', 'Total_Log_Hours_Assignee', 'COMMENTOR_COUNT', 'COMMENT_COUNT']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

#Use X in Naive-Bayes Gauusian
