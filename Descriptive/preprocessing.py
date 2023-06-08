import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

def prepare_data(data):
    columns_to_use = [
        'PRIORITY', # replace nan values with mode
        'URGENCY', # replace nan values with KNN imputer, convert to ordinal values
        'COMPONENT', # replace nan values with mode, aggregate category with count less than 100 into a single category named 'OTHER'
        'IMPACT', # replace nan values with 'unknown'
        'ISSUE_CATEGORY', # remove rows with 'nan' values, aggregate category with count less than 100 into a single category named 'OTHER'
        'ASSIGNEE', # aggregate category with count less than 100 into a single category named 'OTHER'
        'Total_Assignee',
        'Total_Worklog_Assginee',
        'Total_Log_Hours_Assignee', # remove less than 100 elements
        'COMMENTOR_COUNT', # remove less than 100 elements
        'COMMENT_COUNT', # remove less than 100 elements
        'ISSUE_TYPE'
    ]

    data = data[columns_to_use].copy()  # Make a copy of the data

    
    # Conduct outlier analysis for numerical columns
    numerical_columns = ['Total_Assignee', 'Total_Worklog_Assginee', 'Total_Log_Hours_Assignee', 'COMMENTOR_COUNT', 'COMMENT_COUNT']
    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # Keep only valid data
        data = data[(data[column] >= Q1 - 1.5*IQR) & (data[column] <= Q3 + 1.5*IQR)]
    
    categorical_columns = ['PRIORITY', 'URGENCY', 'COMPONENT', 'IMPACT', 'ISSUE_CATEGORY', 'ASSIGNEE', 'ISSUE_TYPE'] 

    # Convert only categorical columns to string type before encoding
    for column in categorical_columns:
        data[column] = data[column].astype(str)

    # Print the count of unique values in each column
    print("Original values in each column(Preprocessed Dataset):")
    for column in data.columns:
        print(data[column].value_counts())

    # Remove rows with 'nan' values in ISSUE_CATEGORY column
    data = data[data['ISSUE_CATEGORY'] != 'nan']

    # Replace 'nan' strings in URGENCY, IMPACT and COMPONENT columns with 'unknown'
    data['COMPONENT'] = data['COMPONENT'].replace('nan', 'unknown')
    data['IMPACT'] = data['IMPACT'].replace('nan', 'unknown')

    # Replace empty strings in COMPONENT column with mode
    mode_value = data.loc[data['COMPONENT'] != 'unknown', 'COMPONENT'].mode()[0]
    data['COMPONENT'] = data['COMPONENT'].replace('unknown', mode_value)

    # Aggregate category with count less than 100 into a single category
    for column in ['COMPONENT', 'ISSUE_CATEGORY', 'ASSIGNEE']:
        counts = data[column].value_counts()
        data[column] = data[column].apply(lambda x: 'OTHER' if counts[x] < 100 else x)
 
    # Convert 'URGENCY' and 'PRIORITY' column to ordinal values
    data['URGENCY'] = data['URGENCY'].map({'Low': 0, 'Medium': 1, 'High': 2, 'nan': np.nan})
    data['PRIORITY'] = data['PRIORITY'].map({'Trivial': 0, 'Minor': 1, 'Major': 2, 'Critical': 3, 'Blocker': 4, 'nan': np.nan})

    # Impute missing values in 'URGENCY' column using KNN imputer
    data_for_imputation = data[['URGENCY']]
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    data_imputed = imputer.fit_transform(data_for_imputation)
    data['URGENCY'] = data_imputed

    # Impute missing values in 'PRIORITY' column using mode
    mode_value_priority = data.loc[data['PRIORITY'].notna(), 'PRIORITY'].mode()[0]
    data['PRIORITY'] = data['PRIORITY'].replace(np.nan, mode_value_priority)

    # Compute the weighted mean for 'Total_Log_Hours_Assignee' and replace values with frequencies below 100
    counts = data['Total_Log_Hours_Assignee'].value_counts()
    values_to_replace = counts[counts < 100].index
    weights = counts.loc[values_to_replace].values
    values_to_replace_weighted = np.repeat(values_to_replace, weights)
    weighted_mean = np.mean(values_to_replace_weighted)
    data.loc[data['Total_Log_Hours_Assignee'].isin(values_to_replace), 'Total_Log_Hours_Assignee'] = weighted_mean

    print("Unique values in each column(Preprocessed Dataset):")
    for column in data.columns:
        print(data[column].value_counts())

    label_encoder = LabelEncoder()
    label_encoding_columns = ['COMPONENT', 'IMPACT', 'ISSUE_CATEGORY', 'ASSIGNEE', 'ISSUE_TYPE']
    for column in label_encoding_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Save the preprocessed data to a XLSX file
    data.to_excel('preprocessed_data.xlsx', index=False)

    return data