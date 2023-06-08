import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

def kmeans_clustering():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load your data
    data = pd.read_excel("preprocessed_data.xlsx")

    # Use PCA to reduce dimensionality
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Create a list to store the WCSS values for the elbow method
    wcss = []
    silhouette = []

    # Change the number of clusters from 1 to 10 in a loop
    for i in range(2, 11): # silhouette_score requires at least 2 clusters
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_pca)
        wcss.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data_pca, kmeans.labels_))

    # Plot the graph for the elbow method
    plt.plot(range(2, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.savefig(os.path.join('figures', 'elbow_method.png')) # Save the figure
    plt.close()

    # Plot the graph for the silhouette scores
    plt.plot(range(2, 11), silhouette)
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join('figures', 'silhouette_scores.png')) # Save the figure
    plt.close()

    # Determine the optimal number of clusters
    optimal_cluster_number = 3
    kmeans = KMeans(n_clusters=optimal_cluster_number, init='k-means++', max_iter=300, n_init=10, random_state=0)
    print("Number of clusters for kmean(calculated by elbow method & silhouette score): " + str(optimal_cluster_number) + "\n")
    pred_y = kmeans.fit_predict(data_pca)

    # Create a scatter plot of the first two principal components, colored by cluster labels
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=pred_y, cmap='viridis')
    plt.title('Clusters of data')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(os.path.join('figures', 'clusters.png')) # Save the figure
    plt.close()
