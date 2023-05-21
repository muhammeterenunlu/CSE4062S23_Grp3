
from pandas.api.types import CategoricalDtype
import pandas as pd

def prepare_data(data):
    # Define order for PRIORITY and URGENCY
    priority_order = CategoricalDtype(['Trivial', 'Minor', 'Major', 'Critical', 'Blocker'], ordered=True)
    urgency_order = CategoricalDtype(['Low', 'Medium', 'High'], ordered=True)

    # Convert PRIORITY and URGENCY to ordinal (category)
    data['PRIORITY'] = data['PRIORITY'].astype(priority_order)
    data['URGENCY'] = data['URGENCY'].astype(urgency_order)

    # Apply one hot encoding
    data = pd.get_dummies(data, columns=['PRIORITY', 'URGENCY'])

    # Define your ordinal columns
    ordinal_boolean_columns = [
        'PRIORITY_Trivial',
        'PRIORITY_Minor',
        'PRIORITY_Major',
        'PRIORITY_Critical',
        'PRIORITY_Blocker',
        'URGENCY_Low',
        'URGENCY_Medium',
        'URGENCY_High',
    ]

    for column in ordinal_boolean_columns:
        # Convert boolean to int
        if data[column].dtype == 'bool':
            data[column] = data[column].astype(int)
        
        # Check for missing values
        if data[column].isnull().any():
            # Fill missing values with 0 or appropriate value
            data[column] = data[column].fillna(0)

    return data