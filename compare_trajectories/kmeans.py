from sklearn.cluster import KMeans
import logging
import os
from compare_trajectories.utils import save_data_numpy

path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def apply_kmeans(data, n_clusters=30, centroids_filename = "centroids.npy", save_data = True):
    kmeans = KMeans(n_clusters=n_clusters)
    logging.info("Training Kmeans")
    kmeans.fit(data)
    data_predictions = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    if save_data:
        save_data_numpy(
            cluster_centers,
            os.path.join(path, f"data/kmeans"),
            centroids_filename
        )
    return data_predictions
