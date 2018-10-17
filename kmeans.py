import numpy
import matrix
from matrix import distance_matrix, type_matrix


class kmeans:
    def __init__(self,data,initial_centroids, tolerance, **kwargs):
        self.__data = numpy.matrix(data)
        self.__clusters = []
        self.__centroids = numpy.matrix(initial_centroids)
        self.__tolerance = tolerance
        self.__matrix = kwargs.get('matrix', distance_matrix(type_matrix.EUCLIDEAN_SQUARE))
        
    def process(self):
        if (len(self.__data[0])) != len(self.__centroids[0]):
            raise NameError('Dimension of the input data and dimension of the initial cluster centers must be equal.')

        maximum_change = float('inf')
        stop_condition = self.__tolerance
        while maximum_change > stop_condition:
            self.__clusters = self.__update_clusters()
            update_centroids = self.__update_centroids()
            if len(self.__centroids) != len(update_centroids):
                maximum_change = float('inf')
            else:
                changes = self.__matrix(self.__centroids, update_centroids)
                maximum_change = numpy.max(changes)
            self.__centroids = update_centroids.tolist()
    def get_clusters(self):
        return self.__clusters
    def get_centroids(self):
        if isinstance(self.__centroids, list):
            return self.__centroids 
        return self.__centroids.tolist()
    def __update_clusters(self):
        clusters = [[] for _ in range(len(self.__centroids))]
        dataset_diff = numpy.zeros((len(clusters), len(self.__data)))
        for index_centroid in range(len(self.__centroids)):
            dataset_diff[index_centroid] = self.__matrix(self.__data, self.__centroids[index_centroid])

        optimum_indexes = numpy.argmin(dataset_diff, axis=0)
        for index_point in range(len(optimum_indexes)):
            index_cluster = optimum_indexes[index_point]
            clusters[index_cluster].append(index_point)
        clusters = [cluster for cluster in clusters if len(cluster)>0]
        return clusters
    def __update_centroids(self):
        dimension = self.__data.shape[1]
        centroids = numpy.zeros((len(self.__clusters),dimension))

        for index in range(len(self.__clusters)):
            cluster_points = self.__data[self.__clusters[index], :]
            centroids[index] = cluster_points.mean(axis = 0)
        return numpy.matrix(centroids)
