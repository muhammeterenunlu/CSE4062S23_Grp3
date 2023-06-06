import os
import pandas as pd
from preprocessing import prepare_data
from chi_square import chi_square_analysis
from decision_tree_using_info_gain import decision_tree_classification_info_gain, evaluate_model_info_gain
from decision_tree_using_gini_index import decision_tree_classification_gini, evaluate_model_gini_index
from decision_tree_using_gradient_boosting import decision_tree_classification_gradient_boosting, evaluate_model_gradient_boosting
from ann_using_1_hl_adam import ann_1_hidden_layer_classification_adam, evaluate_ann_1_hidden_layer_adam
from ann_using_1_hl_sgd import ann_1_hidden_layer_classification_sgd, evaluate_ann_1_hidden_layer_sgd
from ann_using_1_hl_rmsprop import ann_1_hidden_layer_classification_rmsprop, evaluate_ann_1_hidden_layer_rmsprop
from linear_svm import linear_svm_classification, evaluate_linear_svm
from naive_bayes_bernoulli import naive_bayes_classification_bernoulli, evaluate_model_bernoulli_naive_bayes
from naive_bayes_gaussian import naive_bayes_classification_gaussian, evaluate_model_gaussian_naive_bayes
from naive_bayes_multinomial import naive_bayes_classification_multinomial, evaluate_model_multinomial_naive_bayes
from results import print_decision_tree_info_gain, print_decision_tree_gini_index, print_decision_tree_gradient_boosting, print_ann_1_hidden_layer_adam, print_ann_1_hidden_layer_sgd, print_ann_1_hidden_layer_rmsprop, print_linear_svm, print_bernoulli_naive_bayes, print_gaussian_naive_bayes, print_multinomial_naive_bayes


def main():
    # Ensure figures directory exists
    #if not os.path.exists("figures"):
    #    os.makedirs("figures")

    os.chdir("c:/Users/user/Desktop/CSE4062S23_Grp3/Predictive/")

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

    # Naive Bayes bernoulli classification
    y_train_nb_bernoulli, y_train_pred_nb_bernoulli, y_test_nb_bernoulli, y_pred_nb_bernoulli = naive_bayes_classification_bernoulli(data)

    # Evaluate model
    results_nb_bernoulli = evaluate_model_bernoulli_naive_bayes(y_train_nb_bernoulli, y_train_pred_nb_bernoulli, y_test_nb_bernoulli, y_pred_nb_bernoulli)
    # Calculate error rates
    error_rate_train_nb_bernoulli = 1 - results_nb_bernoulli['train']['accuracy']
    error_rate_test_nb_bernoulli = 1 - results_nb_bernoulli['test']['accuracy']

    print_bernoulli_naive_bayes(results_nb_bernoulli, error_rate_train_nb_bernoulli, error_rate_test_nb_bernoulli, y_train_nb_bernoulli, y_train_pred_nb_bernoulli, y_test_nb_bernoulli, y_pred_nb_bernoulli)

    # Naive Bayes gaussian classification
    y_train_nb_gaussian, y_train_pred_nb_gaussian, y_test_nb_gaussian, y_pred_nb_gaussian = naive_bayes_classification_gaussian(data)

    # Evaluate model
    results_nb_gaussian = evaluate_model_gaussian_naive_bayes(y_train_nb_gaussian, y_train_pred_nb_gaussian, y_test_nb_gaussian, y_pred_nb_gaussian)
    # Calculate error rates
    error_rate_train_nb_gaussian = 1 - results_nb_gaussian['train']['accuracy']
    error_rate_test_nb_gaussian = 1 - results_nb_gaussian['test']['accuracy']

    print_gaussian_naive_bayes(results_nb_gaussian, error_rate_train_nb_gaussian, error_rate_test_nb_gaussian, y_train_nb_gaussian, y_train_pred_nb_gaussian, y_test_nb_gaussian, y_pred_nb_gaussian)

    # Naive Bayes multinomial classification
    y_train_nb_multinomial, y_train_pred_nb_multinomial, y_test_nb_multinomial, y_pred_nb_multinomial = naive_bayes_classification_multinomial(data)

    # Evaluate model
    results_nb_multinomial = evaluate_model_multinomial_naive_bayes(y_train_nb_multinomial, y_train_pred_nb_multinomial, y_test_nb_multinomial, y_pred_nb_multinomial)
    # Calculate error rates
    error_rate_train_nb_multinomial = 1 - results_nb_multinomial['train']['accuracy']
    error_rate_test_nb_multinomial = 1 - results_nb_multinomial['test']['accuracy']

    print_multinomial_naive_bayes(results_nb_multinomial, error_rate_train_nb_multinomial, error_rate_test_nb_multinomial, y_train_nb_multinomial, y_train_pred_nb_multinomial, y_test_nb_multinomial, y_pred_nb_multinomial)

    # Decision tree using Information Gain classification
    y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain = decision_tree_classification_info_gain(data)

    # Evaluate model
    results_gain = evaluate_model_info_gain(y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)
    # Calculate error rates
    error_rate_train_gain = 1 - results_gain['train']['accuracy']
    error_rate_test_gain = 1 - results_gain['test']['accuracy']

    print_decision_tree_info_gain(results_gain, error_rate_train_gain, error_rate_test_gain, y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)

    # Decision tree using Gini Index classification
    y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini = decision_tree_classification_gini(data)

    # Evaluate model
    results_gini = evaluate_model_gini_index(y_train_gini, y_train_pred_gini, y_test_gini, y_pred_gini)
    # Calculate error rates
    error_rate_train_gini = 1 - results_gini['train']['accuracy']
    error_rate_test_gini = 1 - results_gini['test']['accuracy']
    (results_gini, error_rate_train_gini, error_rate_test_gini)
    print_decision_tree_gini_index(results_gain, error_rate_train_gain, error_rate_test_gain, y_train_gain, y_train_pred_gain, y_test_gain, y_pred_gain)
   
    # Decision tree using Gradient Boosting classification
    y_train_gb, y_train_pred_gb, y_test_gb, y_pred_gb = decision_tree_classification_gradient_boosting(data)

    # Evaluate model
    results_gb = evaluate_model_gradient_boosting(y_train_gb, y_train_pred_gb, y_test_gb, y_pred_gb)
    # Calculate error rates
    error_rate_train_gb = 1 - results_gb['train']['accuracy']
    error_rate_test_gb = 1 - results_gb['test']['accuracy']

    print_decision_tree_gradient_boosting(results_gb, error_rate_train_gb, error_rate_test_gb, y_train_gb, y_train_pred_gb, y_test_gb, y_pred_gb)

    # ANN with 1 hidden layer classification (ADAM)
    y_train_ann_1_adam, y_train_pred_ann_1_adam, y_test_ann_1_adam, y_pred_ann_1_adam = ann_1_hidden_layer_classification_adam(data)

    # Evaluate model
    results_ann_1_adam = evaluate_ann_1_hidden_layer_adam(y_train_ann_1_adam, y_train_pred_ann_1_adam, y_test_ann_1_adam, y_pred_ann_1_adam)
    # Calculate error rates
    error_rate_train_ann1_adam = 1 - results_ann_1_adam['train']['accuracy']
    error_rate_test_ann1_adam = 1 - results_ann_1_adam['test']['accuracy']

    print_ann_1_hidden_layer_adam(results_ann_1_adam, error_rate_train_ann1_adam, error_rate_test_ann1_adam, y_train_ann_1_adam, y_train_pred_ann_1_adam, y_test_ann_1_adam, y_pred_ann_1_adam)

    # ANN with 1 hidden layer classification (SGD)
    y_train_ann_1_sgd, y_train_pred_ann_1_sgd, y_test_ann_1_sgd, y_pred_ann_1_sgd = ann_1_hidden_layer_classification_sgd(data)

    # Evaluate model
    results_ann_1_sgd = evaluate_ann_1_hidden_layer_sgd(y_train_ann_1_sgd, y_train_pred_ann_1_sgd, y_test_ann_1_sgd, y_pred_ann_1_sgd)

    # Calculate error rates
    error_rate_train_ann1_sgd = 1 - results_ann_1_sgd['train']['accuracy']
    error_rate_test_ann1_sgd = 1 - results_ann_1_sgd['test']['accuracy']

    print_ann_1_hidden_layer_sgd(results_ann_1_sgd, error_rate_train_ann1_sgd, error_rate_test_ann1_sgd, y_train_ann_1_sgd, y_train_pred_ann_1_sgd, y_test_ann_1_sgd, y_pred_ann_1_sgd)

     # ANN with 1 hidden layer classification (RMSprop)
    y_train_ann_1_rmsprop, y_train_pred_ann_1_rmsprop, y_test_ann_1_rmsprop, y_pred_ann_1_rmsprop = ann_1_hidden_layer_classification_rmsprop(data)

    # Evaluate model
    results_ann_1_rmsprop = evaluate_ann_1_hidden_layer_rmsprop(y_train_ann_1_rmsprop, y_train_pred_ann_1_rmsprop, y_test_ann_1_rmsprop, y_pred_ann_1_rmsprop)

    # Calculate error rates
    error_rate_train_ann1_rmsprop = 1 - results_ann_1_rmsprop['train']['accuracy']
    error_rate_test_ann1_rmsprop = 1 - results_ann_1_rmsprop['test']['accuracy']

    print_ann_1_hidden_layer_rmsprop(results_ann_1_rmsprop, error_rate_train_ann1_rmsprop, error_rate_test_ann1_rmsprop, y_train_ann_1_rmsprop, y_train_pred_ann_1_rmsprop, y_test_ann_1_rmsprop, y_pred_ann_1_rmsprop)

    # Linear SVM classification
    y_train_svm, y_train_pred_svm, y_test_svm, y_pred_svm = linear_svm_classification(data)

    # Evaluate model
    results_svm = evaluate_linear_svm(y_train_svm, y_train_pred_svm, y_test_svm, y_pred_svm)
    # Calculate error rates
    error_rate_train_svm = 1 - results_svm['train']['accuracy']
    error_rate_test_svm = 1 - results_svm['test']['accuracy']

    print_linear_svm(results_svm, error_rate_train_svm, error_rate_test_svm, y_train_svm, y_train_pred_svm, y_test_svm, y_pred_svm)


# Call the function
main()
