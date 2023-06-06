import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def anova_kruskallwallis_analysis(data):

    # Define your numerical and ordinal columns
    numerical_and_ordinal_columns = [
        'PRIORITY',
        'URGENCY',
        'Total_Assignee',
        'Total_Worklog_Assginee',
        'Total_Log_Hours_Assignee',
        'COMMENTOR_COUNT',
        'COMMENT_COUNT',
    ]

    # Your class attribute
    class_attribute = 'ISSUE_CATEGORY'

    # Convert necessary columns back to numeric type before ANOVA and Kruskal-Wallis tests
    for column in numerical_and_ordinal_columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    for column in numerical_and_ordinal_columns:
        print(f"\nAnalyzing relationship between {class_attribute} and {column}")

        # Conduct ANOVA test for this column and the class attribute
        model = ols(f'{column} ~ C({class_attribute})', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print("ANOVA Test Results:")
        print(anova_table)

        # Conduct Kruskal-Wallis test for this column and the class attribute
        groups = data.groupby(class_attribute)[column].apply(list)
        h_statistic, p_value = stats.kruskal(*groups)

        print("\nKruskal-Wallis Test Results:")
        print(f"H-statistic: {h_statistic}")
        print(f"P-value: {p_value}")

        # Check the significance of p-value
        p_value_anova = anova_table["PR(>F)"][0]
        if p_value_anova < 0.05:
            print(f"There is a significant relationship between {class_attribute} and {column} according to ANOVA.")
        else:
            print(f"There is no significant relationship between {class_attribute} and {column} according to ANOVA.")
