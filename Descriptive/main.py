import os
import pandas as pd
from preprocessing import prepare_data
from kmeans import kmeans_clustering

def main():
    # Ensure figures directory exists
    if not os.path.exists("Descriptive/figures"):
        os.makedirs("Descriptive/figures")

    os.chdir("c:/Users/user/Desktop/CSE4062S23_Grp3/Descriptive/")

    # Load original data
    data = pd.read_excel("data.xlsx", engine="openpyxl", header=0, sheet_name=3)

    # Print the columns of the original data
    print("Original data")
    print(data.columns)
    # Print the first 10 rows of the transformed dataset
    print("\nFirst 10 rows of the transformed dataset:")
    print(data.head(10))
    # How many rows and columns are there in the original data?
    print("\nNumber of rows and columns in the original data:", data.shape)

    # Prepare data
    data = prepare_data(data)
    # Print the columns of the preprocessed data
    print("\nPreprocessed data")
    print(data.columns)
    # Print the first 10 rows of the transformed dataset
    print("\nFirst 10 rows of the transformed dataset:")
    print(data.head(10))
    # How many rows and columns are there in the preprocessed data?
    print("\nNumber of rows and columns in the preprocessed data:", data.shape)

    # Call the kmeans_clustering function
    kmeans_clustering()

# Call the function
main()
