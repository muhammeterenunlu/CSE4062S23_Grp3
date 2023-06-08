import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import os

def dbscan_clustering():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load your data
    data = pd.read_excel("preprocessed_data.xlsx")

    # Compute the nearest neighbors
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    neighbors = nearest_neighbors.fit(data)
    distances, indices = neighbors.kneighbors(data)

    # Sort the distance and plot it
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.figure(figsize=(10,5))
    plt.plot(distances)
    plt.title('K-distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('Epsilon')
    plt.savefig(os.path.join('figures', 'k_distance_graph.png'))  # Save the figure
    plt.close()

    # We can see epsilon value from the knee point in the K-distance graph
    epsilon = 0.3  # Suppose knee point is at 0.3

    # Initialize DBSCAN with the computed epsilon
    db = DBSCAN(eps=epsilon, min_samples=5)

    # Fit and predict clusters
    pred_y = db.fit_predict(data)

    # If we have a 2D data, we can visualize it
    if data.shape[1] == 2:
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=pred_y, cmap='viridis')
        plt.title('Clusters of data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig(os.path.join('figures', 'dbscan_clusters.png'))  # Save the figure
        plt.close()

