from kmeans import kmeans
from dbscan import dbscan
from __init__ import read_data

def kmeans_clustering(filename, tolerance, start_centroids):
    sample = read_data(filename)
    kmeans_instance = kmeans(sample, start_centroids, tolerance=0.25)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centroids = kmeans_instance.get_centroids()
    # print ("Cluster Result: \n", clusters)
    # print ("Centroids Result: \n",centroids)

def dbscan_clustering(filename, epsilon, min_pts):
	sample = read_data(filename)
	dbscan_instance = dbscan(sample, epsilon, min_pts)
	dbscan_instance.process()
	clusters = dbscan_instance.get_clusters()
	outliers = dbscan_instance.get_outliers()
	print("Clusters :\n", clusters)
	print("Outliers :\n", outliers)

# KMeans Clustering
start_centroids = [[5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2]]
filename = "./dataset/iris_without_label.data"
kmeans_clustering(filename, 0.25, start_centroids)
dbscan_clustering(filename, 2.2, 50)
