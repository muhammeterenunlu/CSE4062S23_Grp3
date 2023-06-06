import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_excel(r"C:\Users\ASUS\OneDrive - marun.edu.tr\Marmara\4. Sınıf 2. Dönem\Data Science\Proje\data.xlsx", engine="openpyxl", header=0, sheet_name=3)

def svm(data):
    # Prepare the dataset
    X = data.drop('ISSUE_CATEGORY', axis=1)
    y = data['ISSUE_CATEGORY']

    # Split data into training and test sets using holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(X_train.dtypes)
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Fit the model to the training data
    svm_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test, y_pred)

    # Create a table to store the evaluation metrics
    evaluation_table = pd.DataFrame(columns=['#', 'Experiment', 'Accuracy', 'F1-macro', 'F1-micro', 'AUC', 'SSE'])
    evaluation_table.loc[0] = [1, 'SVM', accuracy, f1_macro, None, auc, None]

    # Print the evaluation metrics
    print(evaluation_table)

    # Perform stratified 10-fold cross-validation
    cv_scores = cross_val_score(svm_model, X, y, cv=10, scoring='accuracy')

# Call the svm function with your dataset
svm(data)
