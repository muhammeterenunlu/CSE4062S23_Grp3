import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def hierarchical_clustering():
    # Ensure figures directory exists
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Load your data
    data = pd.read_excel("preprocessed_data.xlsx")

    # Use PCA to reduce dimensionality
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Generate the linkage matrix
    linked = linkage(data_pca, 'ward')

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.savefig(os.path.join('figures', 'dendrogram.png')) # Save the figure
    plt.close()
