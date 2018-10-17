import numpy
import math

class LinkageDistanceMatrix:
    def __init__(self, data, clusters, linkage = "single", affinity = "euclidean"):
        self.data = data
        self.clusters = clusters
        self.linkage = linkage
        self.affinity = affinity
        self.matrix = self.create_distance_matrix()
    
    def create_distance_matrix(self):
        if (self.linkage == "complete"):
            return self.create_complete_distance_matrix()
        elif (self.linkage == "average"):
            return self.create_average_distance_matrix()
        elif (self.linkage == "average_group"):
            return self.create_average_group_distance_matrix()
        else:
            return self.create_single_distance_matrix()
    
    def create_single_distance_matrix(self):
        matrix = numpy.zeros((len(self.clusters), len(self.clusters)))
        for matrix1 in range(len(self.clusters)):
            for matrix2 in range(matrix1, len(self.clusters)):
                distance = float('inf')
                for index1 in self.clusters[matrix1]:
                    for index2 in self.clusters[matrix2]:
                        pairdistance = 0
                        if (self.affinity == "manhattan"):
                            for feature in range(self.data[0].size):
                                pairdistance = pairdistance + abs(self.data[index2, feature] - self.data[index1, feature])
                        else:
                            for feature in range(self.data[0].size):
                                pairdistance = pairdistance + (self.data[index2, feature] - self.data[index1, feature]) ** 2
                            pairdistance = math.sqrt(pairdistance)
                        distance = min(distance, pairdistance)
                matrix[matrix1, matrix2] = distance
                matrix[matrix2, matrix1] = distance
        return matrix
    
    def create_complete_distance_matrix(self):
        matrix = numpy.zeros((len(self.clusters), len(self.clusters)))
        # TODO
        return matrix

    def create_average_distance_matrix(self):
        matrix = numpy.zeros((len(self.clusters), len(self.clusters)))
        # TODO
        return matrix

    def create_average_group_distance_matrix(self):
        matrix = numpy.zeros((len(self.clusters), len(self.clusters)))
        # TODO
        return matrix