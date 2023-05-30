from sklearn.ensemble import RandomForestClassifier

def random_forest_classification_gini(data):
    # Prepare the dataset
    X = data.drop('ISSUE_CATEGORY', axis=1)
    y = data['ISSUE_CATEGORY']

    # Split data into training and test sets using holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create the Random Forest classifier with Gini index criterion
    forest = RandomForestClassifier(criterion="gini", random_state=42)
    
    # Implement AdaBoost
    clf = AdaBoostClassifier(estimator=forest, n_estimators=20, random_state=42)

    # Train the model using cross-validation on the training set
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_pred = cross_val_predict(clf, X_train, y_train, cv=kf)

    # Train the model on the full training set
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    return y_train, y_train_pred, y_test, y_pred


def evaluate_model_gini_index(y_train, y_train_pred, y_test, y_pred):
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    recall_train = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)
    recall_test = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    precision_train = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
    precision_test = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_pred, average='weighted')

    results = {
        'train': {
            'accuracy': acc_train,
            'recall': recall_train,
            'precision': precision_train,
            'f1_score': f1_train
        },
        'test': {
            'accuracy': acc_test,
            'recall': recall_test,
            'precision': precision_test,
            'f1_score': f1_test
        }
    }

    return results
