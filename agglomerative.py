import numpy
from linkagematrix import LinkageDistanceMatrix

class Agglomerative:
    def __init__(self, n_clusters = 2, linkage = "single", affinity = "euclidean"):
        self.data = numpy.array([])
        self.clusters = []
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity
        self.distance_matrix = numpy.array([])
        
    def fit(self, data):
        self.data = numpy.matrix(data)
        for index in range(len(data)):
            self.clusters.append([index])
        while (len(self.clusters) > self.n_clusters):
            self.distance_matrix = LinkageDistanceMatrix(self.data, self.clusters, self.linkage, self.affinity)
            self.clusters = self.update_clusters()

    def get_clusters(self):
        return self.clusters

    def update_clusters(self):
        clusters = []
        minindexi = 0
        minindexj = 0
        mindistance = float('inf')
        for indexi in range(len(self.clusters)):
            for indexj in range(indexi, len(self.clusters)):
                if (indexi != indexj and self.distance_matrix.matrix[indexi, indexj] < mindistance):
                    minindexi = indexi
                    minindexj = indexj
                    mindistance = self.distance_matrix.matrix[indexi, indexj]
        iterator = -1
        for cluster in self.clusters:
            iterator += 1
            if (iterator == minindexi):
                clusters.append(self.clusters[minindexi] + self.clusters[minindexj])
            elif (iterator != minindexj):
                clusters.append(self.clusters[iterator])
        return clusters
