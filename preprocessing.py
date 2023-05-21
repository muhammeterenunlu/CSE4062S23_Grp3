import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def prepare_data(data):
    columns_to_use = [
        'PRIORITY',
        'URGENCY',
        'COMPONENT',
        'IMPACT',
        'ISSUE_CATEGORY',
        'POZISYON',
        'IS_BILGISI',
        'ASSIGNEE',
        'Total_Assignee',
        'Total_Worklog_Assginee',
        'Total_Log_Hours_Assignee',
        'COMMENTOR_COUNT',
        'COMMENT_COUNT',
        'ISSUE_TYPE'
    ]

    data = data[columns_to_use].copy()  # Make a copy of the data

    # Convert all columns to string type before encoding
    for column in data.columns:
        data[column] = data[column].astype(str)

    # Check for missing values
    for column in data.columns:
        if data[column].isnull().any():
            # Fill missing values with 0 or appropriate value
            data[column] = data[column].fillna('0')

    # Apply ordinal encoding
    ordinal_encoder = OrdinalEncoder()
    ordinal_columns = ['PRIORITY', 'URGENCY']
    data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])
    
    # Apply label encoding
    label_encoder = LabelEncoder()
    label_encoding_columns = ['COMPONENT', 'IMPACT', 'ISSUE_CATEGORY', 'POZISYON', 'IS_BILGISI', 'ASSIGNEE', 'ISSUE_TYPE']
    for column in label_encoding_columns:
        data[column] = label_encoder.fit_transform(data[column])

    return data