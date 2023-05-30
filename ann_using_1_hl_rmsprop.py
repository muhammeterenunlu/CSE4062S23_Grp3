# Import necessary libraries
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.wrappers.scikit_learn import KerasClassifier

# Function to build the ANN model with 1 hidden layer
def build_ann_model(input_dim):
    model = tf.keras.Sequential([
        # Add a dense (fully connected) hidden layer with 32 neurons and ReLU activation
        tf.keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        # Add an output layer with num_classes neurons and softmax activation for multi-class classification
        tf.keras.layers.Dense(32, activation='softmax')
    ])

    # Compile the model using the Adam optimizer and sparse categorical crossentropy loss function
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to perform classification using the ANN model
def ann_1_hidden_layer_classification_rmsprop(data):
    # Prepare the dataset
    X = data.drop('ISSUE_CATEGORY', axis=1)
    y = data['ISSUE_CATEGORY']

    # Split data into training and test sets using the holdout method
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the input features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a function that returns a compiled model
    def create_compiled_model():
        model = build_ann_model(input_dim=X_train.shape[1])
        return model

    # Wrap the model using KerasClassifier
    keras_clf = KerasClassifier(build_fn=create_compiled_model, epochs=100, batch_size=32, verbose=0)

    # Perform cross-validation on the training set
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    y_train_pred = cross_val_predict(keras_clf, X_train_scaled, y_train, cv=kf)

    # Train the model on the full training set
    keras_clf.fit(X_train_scaled, y_train)

    # Make predictions on the training and test sets
    y_train_pred = keras_clf.predict(X_train_scaled)
    y_pred = keras_clf.predict(X_test_scaled)

    return y_train, y_train_pred, y_test, y_pred

# Function to evaluate the performance of the ANN model
def evaluate_ann_1_hidden_layer_rmsprop(y_train, y_train_pred, y_test, y_pred):
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_pred)
    recall_train = recall_score(y_train, y_train_pred, average='weighted', zero_division=1)
    recall_test = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    precision_train = precision_score(y_train, y_train_pred, average='weighted', zero_division=1)
    precision_test = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    f1_train = f1_score(y_train, y_train_pred, average='weighted')
    f1_test = f1_score(y_test, y_pred, average='weighted')

    # Store the results in a dictionary
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