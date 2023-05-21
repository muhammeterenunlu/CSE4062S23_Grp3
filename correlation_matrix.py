import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import os

def correlation_matrix_analysis(data):

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
    plt.close()

    return data