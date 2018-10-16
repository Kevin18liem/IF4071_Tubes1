from __init__ import read_data
from kmeans import kmeans
from agglomerative import Agglomerative

def kmeans_clustering(filename, tolerance, start_centroids):
    sample = read_data(filename)
    kmeans_instance = kmeans(sample, start_centroids, tolerance=0.25)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centroids = kmeans_instance.get_centroids()
    print ("Cluster Result: \n", clusters)
    print ("Centroids Result: \n",centroids)

def agglomerative_clustering(filename, n_clusters, linkage, affinity):
    sample = read_data(filename)
    agglomerative_instance = Agglomerative(n_clusters, linkage, affinity)
    agglomerative_instance.fit(sample)
    clusters = agglomerative_instance.get_clusters()
    print ("Cluster Result: \n", clusters)

# KMeans Clustering
start_centroids = [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]
filename = "dataset/iris_without_label.data"
kmeans_clustering(filename, 0.25, start_centroids)

# Agglomerative Clustering
filename = "dataset/iris_without_label.data"
agglomerative_clustering(filename, 3, "single", "euclidean")