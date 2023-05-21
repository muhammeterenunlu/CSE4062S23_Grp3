import os
import pandas as pd
from preprocessing import prepare_data
from correlation_matrix import correlation_matrix_analysis
from chi_square import chi_square_analysis
from anova_kruskallwallis import anova_kruskallwallis_analysis

def main():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load original data
    data = pd.read_excel("data.xlsx", engine="openpyxl", header=0, sheet_name=3)

    # Print the columns of the original data
    print("Original data")
    print(data.columns)
    # Print the first 10 rows of the transformed dataset
    print("\nFirst 10 rows of the transformed dataset:")
    print(data.head(10))

    # Prepare data
    data = prepare_data(data)

    # Print the columns of the preprocessed data
    print("\nPreprocessed data")
    print(data.columns)
    # Print the first 10 rows of the transformed dataset
    print("\nFirst 10 rows of the transformed dataset:")
    print(data.head(10))

    # Apply the correlation matrix analysis
    data = correlation_matrix_analysis(data)

    print("Relationship between Class Attribute(Nominal) with Nominal values")
    # Apply the chi-square analysis
    chi_square_analysis(data)

    print("\nRelationship between Class Attribute(Nominal) with Ordinal and Numeric values")
    # Apply the ANOVA and Kruskal-Wallis analysis
    anova_kruskallwallis_analysis(data)

# Call the function
main()
