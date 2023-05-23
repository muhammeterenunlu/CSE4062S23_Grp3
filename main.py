import os
import pandas as pd
from preprocessing import prepare_data
from chi_square import chi_square_analysis
#from anova_kruskallwallis import anova_kruskallwallis_analysis
from decision_tree_using_gain_ratio import decision_tree_classification_gain, evaluate_model_gain_ratio
from decision_tree_using_gini_index import decision_tree_classification_gini, evaluate_model_gini_index
from decision_tree_using_gradient_boosting import decision_tree_classification_gradient_boosting, evaluate_model_gradient_boosting
from ann_using_1_hl_adam import ann_1_hidden_layer_classification_adam, evaluate_ann_1_hidden_layer_adam
from ann_using_1_hl_sgd import ann_1_hidden_layer_classification_sgd, evaluate_ann_1_hidden_layer_sgd
from ann_using_1_hl_rmsprop import ann_1_hidden_layer_classification_rmsprop, evaluate_ann_1_hidden_layer_rmsprop

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
    # How many rows and columns are there in the original data?
    print("\nNumber of rows and columns in the original data:", data.shape)

    print("Relationship between Class Attribute(Nominal) with Nominal values")
    # Apply the chi-square analysis
    chi_square_analysis(data)

    #print("\nRelationship between Class Attribute(Nominal) with Ordinal and Numeric values")
    # Apply the ANOVA and Kruskal-Wallis analysis
    #anova_kruskallwallis_analysis(data)

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

    # Decision tree using Information Gain Gain Ratio classification
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

    # Decision tree using Gini Index classification
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

    
    # Decision tree using Gradient Boosting classification
    y_train_gb, y_train_pred_gb, y_test_gb, y_pred_gb = decision_tree_classification_gradient_boosting(data)

    # Evaluate model
    results_gb = evaluate_model_gradient_boosting(y_train_gb, y_train_pred_gb, y_test_gb, y_pred_gb)
    # Calculate error rates
    error_rate_train_gb = 1 - results_gb['train']['accuracy']
    error_rate_test_gb = 1 - results_gb['test']['accuracy']

    print("\nDecision Tree using Gradient Boosting")
    print("-----------------")
    print("Training set results:")
    print("Accuracy:", results_gb['train']['accuracy'])
    print("Recall:", results_gb['train']['recall'])
    print("Precision:", results_gb['train']['precision'])
    print("F1 Score:", results_gb['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_gb['test']['accuracy'])
    print("Recall:", results_gb['test']['recall'])
    print("Precision:", results_gb['test']['precision'])
    print("F1 Score:", results_gb['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_gb)
    print("Error Rate (Test):", error_rate_test_gb)
    

    # ANN with 1 hidden layer classification (ADAM)
    y_train_ann_1_adam, y_train_pred_ann_1_adam, y_test_ann_1_adam, y_pred_ann_1_adam = ann_1_hidden_layer_classification_adam(data)

    # Evaluate model
    results_ann_1_adam = evaluate_ann_1_hidden_layer_adam(y_train_ann_1_adam, y_train_pred_ann_1_adam, y_test_ann_1_adam, y_pred_ann_1_adam)
    # Calculate error rates
    error_rate_train_ann1_adam = 1 - results_ann_1_adam['train']['accuracy']
    error_rate_test_ann1_adam = 1 - results_ann_1_adam['test']['accuracy']

    print("\nANN with 1 Hidden Layer (ADAM Optimizer)")
    print("-------------------------")
    print("Training set results:")
    print("Accuracy:", results_ann_1_adam['train']['accuracy'])
    print("Recall:", results_ann_1_adam['train']['recall'])
    print("Precision:", results_ann_1_adam['train']['precision'])
    print("F1 Score:", results_ann_1_adam['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_ann_1_adam['test']['accuracy'])
    print("Recall:", results_ann_1_adam['test']['recall'])
    print("Precision:", results_ann_1_adam['test']['precision'])
    print("F1 Score:", results_ann_1_adam['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_ann1_adam)
    print("Error Rate (Test):", error_rate_test_ann1_adam)

    # ANN with 1 hidden layer classification (SGD)
    y_train_ann_1_sgd, y_train_pred_ann_1_sgd, y_test_ann_1_sgd, y_pred_ann_1_sgd = ann_1_hidden_layer_classification_sgd(data)

    # Evaluate model
    results_ann_1_sgd = evaluate_ann_1_hidden_layer_sgd(y_train_ann_1_sgd, y_train_pred_ann_1_sgd, y_test_ann_1_sgd, y_pred_ann_1_sgd)

    # Calculate error rates
    error_rate_train_ann1_sgd = 1 - results_ann_1_sgd['train']['accuracy']
    error_rate_test_ann1_sgd = 1 - results_ann_1_sgd['test']['accuracy']

    print("\nANN with 1 Hidden Layer (SGD Optimizer)")
    print("-----------------------------------------")
    print("Training set results:")
    print("Accuracy:", results_ann_1_sgd['train']['accuracy'])
    print("Recall:", results_ann_1_sgd['train']['recall'])
    print("Precision:", results_ann_1_sgd['train']['precision'])
    print("F1 Score:", results_ann_1_sgd['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_ann_1_sgd['test']['accuracy'])
    print("Recall:", results_ann_1_sgd['test']['recall'])
    print("Precision:", results_ann_1_sgd['test']['precision'])
    print("F1 Score:", results_ann_1_sgd['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_ann1_sgd)
    print("Error Rate (Test):", error_rate_test_ann1_sgd)

    # ANN with 1 hidden layer classification (RMSprop)
    y_train_ann_1_rmsprop, y_train_pred_ann_1_rmsprop, y_test_ann_1_rmsprop, y_pred_ann_1_rmsprop = ann_1_hidden_layer_classification_rmsprop(data)

    # Evaluate model
    results_ann_1_rmsprop = evaluate_ann_1_hidden_layer_rmsprop(y_train_ann_1_rmsprop, y_train_pred_ann_1_rmsprop, y_test_ann_1_rmsprop, y_pred_ann_1_rmsprop)

    # Calculate error rates
    error_rate_train_ann1_rmsprop = 1 - results_ann_1_rmsprop['train']['accuracy']
    error_rate_test_ann1_rmsprop = 1 - results_ann_1_rmsprop['test']['accuracy']

    print("\nANN with 1 Hidden Layer (RMSprop Optimizer)")
    print("---------------------------------------------")
    print("Training set results:")
    print("Accuracy:", results_ann_1_rmsprop['train']['accuracy'])
    print("Recall:", results_ann_1_rmsprop['train']['recall'])
    print("Precision:", results_ann_1_rmsprop['train']['precision'])
    print("F1 Score:", results_ann_1_rmsprop['train']['f1_score'])

    print("\nTest set results:")
    print("Accuracy:", results_ann_1_rmsprop['test']['accuracy'])
    print("Recall:", results_ann_1_rmsprop['test']['recall'])
    print("Precision:", results_ann_1_rmsprop['test']['precision'])
    print("F1 Score:", results_ann_1_rmsprop['test']['f1_score'])

    print("\nError Rate (Training):", error_rate_train_ann1_rmsprop)
    print("Error Rate (Test):", error_rate_test_ann1_rmsprop)

# Call the function
main()
