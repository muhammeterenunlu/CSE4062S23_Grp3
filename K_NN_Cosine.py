import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.label_encoder = None

    def preprocess_data(self, X):
        # One-hot encode categorical data
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        self.label_encoder = encoder.categories_[0]
        return X_encoded.toarray()

    def fit(self, X, y):
        self.X_train = self.preprocess_data(X)
        self.y_train = y

    def Cos_distance(self, x1, x2):
        return (np.dot(x1,x2))/(np.sqrt(np.sum((x1) ** 2))*np.sqrt(np.sum((x2))

    def predict(self, X):
        X_encoded = self.preprocess_data(X)
        y_pred = []

        for x in X_encoded:
            distances = pairwise_distances(self.X_train, [x], metric=self.Cos_distance)
            nearest_indices = np.argsort(distances.flatten())[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted_label_index = np.argmax(counts)
            predicted_label = self.label_encoder[predicted_label_index]
            y_pred.append(predicted_label)

        return np.array(y_pred)


# Read input dataset from Excel file
df = pd.read_excel('input_data.xlsx')

# Separate features (X) and labels (y)
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values

# Example test dataset
X_test = pd.read_csv('test_data.csv')

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
