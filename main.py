from __init__ import read_data
from kmeans import kmeans
from dbscan import dbscan
from kmedoids import kmedoids
from agglomerative import Agglomerative

def kmeans_clustering(filename, tolerance, start_centroids):
    sample = read_data(filename)
    kmeans_instance = kmeans(sample, start_centroids, tolerance=0.25)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centroids = kmeans_instance.get_centroids()
    print ("Cluster Result: \n", clusters)
    print ("Centroids Result: \n",centroids)

def dbscan_clustering(filename, epsilon, min_pts):
	sample = read_data(filename)
	dbscan_instance = dbscan(sample, epsilon, min_pts)
	dbscan_instance.process()
	clusters = dbscan_instance.get_clusters()
	outliers = dbscan_instance.get_outliers()
	print("Clusters :\n", clusters)
	print("Outliers :\n", outliers)

def agglomerative_clustering(filename, n_clusters, linkage, affinity):
    sample = read_data(filename)
    agglomerative_instance = Agglomerative(n_clusters, linkage, affinity)
    agglomerative_instance.fit(sample)
    clusters = agglomerative_instance.get_clusters()
    print(linkage, affinity)
    print("Cluster Result: \n", clusters)

def kmedoids_clustering(filename, tolerance, start_medoids):
    sample = read_data(filename)
    kmedoids_instance = kmedoids(sample,start_medoids,tolerance=0)
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
    print ("Cluster Result: \n", clusters)
    print ("Centroids Result: \n",medoids)

# KMeans Clustering
start_centroids = [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]
start_medoids = [0,2,7]
filename = "./dataset/iris_without_label.data"
kmeans_clustering(filename, 0.25, start_centroids)
kmedoids_clustering(filename,0,start_medoids)
dbscan_clustering(filename, 2.2, 50)
agglomerative_clustering(filename, 3, "single", "euclidean")
agglomerative_clustering(filename, 3, "single", "manhattan")
agglomerative_clustering(filename, 3, "complete", "euclidean")
agglomerative_clustering(filename, 3, "complete", "manhattan")
agglomerative_clustering(filename, 3, "average", "euclidean")
agglomerative_clustering(filename, 3, "average", "manhattan")
agglomerative_clustering(filename, 3, "average_group", "euclidean")
agglomerative_clustering(filename, 3, "average_group", "euclidean")