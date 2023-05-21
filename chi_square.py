import pandas as pd
import numpy as np
from scipy import stats

def chi_square_analysis(data):

    # Define your nominal columns
    nominal_columns = [
        'COMPONENT',
        'IMPACT',
        'POZISYON',
        'IS_BILGISI',
        'ASSIGNEE',
        'ISSUE_TYPE',
    ]

    # Your class attribute
    class_attribute = 'ISSUE_CATEGORY'

    for column in nominal_columns:
        print(f"\nAnalyzing relationship between {class_attribute} and {column}")

        # Construct a contingency table
        contingency_table = pd.crosstab(data[class_attribute], data[column])

        # Conduct Chi-Square test for this contingency table
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

        print("Chi-Square Test Results:")
        print(f"Chi-Square statistic: {chi2}")
        print(f"P-value: {p_value}")
        
        # Check the significance of p-value
        if p_value < 0.05:
            print(f"There is a significant relationship between {class_attribute} and {column}.")
        else:
            print(f"There is no significant relationship between {class_attribute} and {column}.")
