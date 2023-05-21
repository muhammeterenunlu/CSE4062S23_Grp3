
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

    return data