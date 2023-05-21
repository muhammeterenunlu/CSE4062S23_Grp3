import os
import pandas as pd
from preprocessing import prepare_data
from correlation_matrix import correlation_matrix_analysis
from chi_square import chi_square_analysis
from anova_kruskallwallis import anova_kruskallwallis_analysis
from decision_tree_using_gain_ratio import decision_tree_classification_gain, evaluate_model_gain_ratio
from decision_tree_using_gini_index import decision_tree_classification_gini, evaluate_model_gini_index

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

    # Decision tree using gain ratio classification
    y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain = decision_tree_classification_gain(data)

    # Evaluate model
    results_gain = evaluate_model_gain_ratio(y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)
    # Calculate error rates
    error_rate_train_gain = 1 - results_gain['train']['accuracy']
    error_rate_test_gain = 1 - results_gain['test']['accuracy']

    print("Decision Tree using Gain Ratio")
    print("------------------------------")
    print("Training set results:")
    print("Accuracy:", results_gain['train']['accuracy'])
    print("Recall:", results_gain['train']['recall'])
    print("Precision:", results_gain['train']['precision'])
    print("F1 Score:", results_gain['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_gain['test']['accuracy'])
    print("Recall:", results_gain['test']['recall'])
    print("Precision:", results_gain['test']['precision'])
    print("F1 Score:", results_gain['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_gain)
    print("Error Rate (Test):", error_rate_test_gain)

    # Decision tree using Gini index classification
    y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini = decision_tree_classification_gini(data)

    # Evaluate model
    results_gini = evaluate_model_gini_index(y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini)
    # Calculate error rates
    error_rate_train_gini = 1 - results_gini['train']['accuracy']
    error_rate_test_gini = 1 - results_gini['test']['accuracy']

    print("\nDecision Tree using Gini Index")
    print("--------------------------------")
    print("Training set results:")
    print("Accuracy:", results_gini['train']['accuracy'])
    print("Recall:", results_gini['train']['recall'])
    print("Precision:", results_gini['train']['precision'])
    print("F1 Score:", results_gini['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_gini['test']['accuracy'])
    print("Recall:", results_gini['test']['recall'])
    print("Precision:", results_gini['test']['precision'])
    print("F1 Score:", results_gini['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_gini)
    print("Error Rate (Test):", error_rate_test_gini)
    
# Call the function
main()
