import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import os

def correlation_matrix_analysis(data):
    # Define order for PRIORITY and URGENCY
    priority_order = CategoricalDtype(['Trivial', 'Minor', 'Major', 'Critical', 'Blocker'], ordered=True)
    urgency_order = CategoricalDtype(['Low', 'Medium', 'High'], ordered=True)

    # Convert PRIORITY and URGENCY to ordinal
    data['PRIORITY'] = data['PRIORITY'].astype(priority_order)
    data['URGENCY'] = data['URGENCY'].astype(urgency_order)

    # Apply one hot encoding
    data = pd.get_dummies(data, columns=['PRIORITY', 'URGENCY'])

    # Select only the columns you're interested in
    selected_columns = [
        'PRIORITY_Trivial',
        'PRIORITY_Minor',
        'PRIORITY_Major',
        'PRIORITY_Critical',
        'PRIORITY_Blocker',
        'URGENCY_Low',
        'URGENCY_Medium',
        'URGENCY_High',
        'Total_Assignee',
        'Total_Worklog_Assginee',
        'Total_Log_Hours_Assignee',
        'COMMENTOR_COUNT',
        'COMMENT_COUNT',
    ]

    # Compute the correlation matrix for the selected columns
    corr = data[selected_columns].corr(numeric_only=False)

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.title("Correlation Matrix - Selected Columns Only")
    plt.savefig(os.path.join("figures", "correlation_matrix_selected.png"), dpi=300)
    
    # Print the first 10 rows of the transformed dataset
    print("\nFirst 10 rows of the transformed dataset:")
    print(data[selected_columns].head(10))

    plt.close()

    return data