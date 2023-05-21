import os
import pandas as pd
from correlation_matrix import correlation_matrix_analysis

def main():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load data
    data = pd.read_excel("data.xlsx", engine="openpyxl", header=0, sheet_name=2)

    # Apply the correlation matrix analysis
    data = correlation_matrix_analysis(data)

# Call the function
main()
