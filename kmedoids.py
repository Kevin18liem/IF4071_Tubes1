import matrix
import random
import numpy
from matrix import distance_matrix, type_matrix

class kmedoids:

    def __init__(self, data, initial_medoids_index, tolerance, **kwargs):
        self.__data = data
        self.__clusters = []
        self.index_medoids = initial_medoids_index
        self.__tolerance = tolerance
        self.__matrix = kwargs.get('matrix', distance_matrix(type_matrix.EUCLIDEAN))
        # self.distance_calculator = self.__create_distance_calculator()
        self.medoids_matrix_checker = numpy.zeros((len(initial_medoids_index), len(data)))
        for i in range(len(initial_medoids_index)):
            self.medoids_matrix_checker[i][initial_medoids_index[i]] = 1
    def process(self):

        diff = float('inf')
        stop_conditon = self.__tolerance
        counter = 0
        while (True):
            # Calculate Old Absolute Error
            self.__clusters = self.__update_clusters()
            old_absolute_error = self.calculate_absolute_error(self.__clusters, self.index_medoids)
            # Calculate New Absolute Error
            # temp_index_medoids = self.index_medoids
            temp_index_medoids = self.__update_medoids()
            new_temporary_clusters = self.__update_clusters()
            new_absolute_error = self.calculate_absolute_error(new_temporary_clusters, temp_index_medoids)
            # Calculate Difference
            diff_error =  new_absolute_error - old_absolute_error
            if diff_error < 0:
                self.index_medoids = temp_index_medoids
            else:
                break
    def get_clusters(self):

        return self.__clusters

    def get_medoids(self):
        
        return self.index_medoids

    def __update_clusters(self):
        clusters = [[self.index_medoids[i]] for i in range(len(self.index_medoids))]
        for index_point in range(len(self.__data)):
            if index_point in self.index_medoids:
                continue
            
            index_optim = -1
            dist_optim = float('Inf')

            for index in range(len(self.index_medoids)):
                dist = self.__matrix(numpy.matrix(self.__data[index_point]), numpy.matrix(self.__data[self.index_medoids[index]]))
                if dist < dist_optim:
                    index_optim = index
                    dist_optim = dist
            clusters[index_optim].append(index_point)
        return clusters

    def __update_medoids(self):
        medoid_indexes = self.index_medoids
        random_index_to_change = random.randint(0,len(self.index_medoids)-1)
        random_value = random.randint(0,len(self.__data)-1)
        while self.medoids_matrix_checker[random_index_to_change][random_value] == 1:
            random_index_to_change = random.randint(0,len(self.index_medoids)-1)
            random_value = random.randint(0,len(self.__data)-1)
        medoid_indexes[random_index_to_change] = random_value
        self.medoids_matrix_checker[random_index_to_change][random_value] = 1
        return medoid_indexes
    def calculate_absolute_error(self, clusters, index_medoids):
        medoids = []
        for i in index_medoids:
            medoids.append(self.__data[i])
        sum = 0
        for i in range(len(index_medoids)):
            for j in range(len(clusters[i])):
                data = self.__data[clusters[i][j]]
                sum += numpy.sum(numpy.absolute(numpy.array(data) - numpy.array(medoids[i])),axis=0)
        return (sum)