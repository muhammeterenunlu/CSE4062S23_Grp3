def print_decision_tree_info_gain(results_gain, error_rate_train_gain, error_rate_test_gain):
    print("Decision Tree using Information Gain")
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

def print_decision_tree_gini_index(results_gini, error_rate_train_gini, error_rate_test_gini):
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

def print_decision_tree_gradient_boosting(results_gb, error_rate_train_gb, error_rate_test_gb):
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

def print_ann_1_hidden_layer_adam(results_ann_1_adam, error_rate_train_ann1_adam, error_rate_test_ann1_adam):
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

def print_ann_1_hidden_layer_sgd(results_ann_1_sgd, error_rate_train_ann1_sgd, error_rate_test_ann1_sgd):
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


def print_ann_1_hidden_layer_rmsprop(results_ann_1_rmsprop, error_rate_train_ann1_rmsprop, error_rate_test_ann1_rmsprop):
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
