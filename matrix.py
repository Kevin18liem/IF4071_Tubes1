import numpy
from enum import IntEnum

class type_matrix(IntEnum):
    
    EUCLIDEAN = 0
    EUCLIDEAN_SQUARE = 1
    MANHATTAN = 2

class distance_matrix:
    def __init__(self, matrix_type, **kwargs):
        self.__type = matrix_type
        self.__args = kwargs
        self.__func = self.__args.get('func', None)
        self.__calculator = self.__create_calculator_distance()

    def __call__(self, object1, object2):
        return self.__calculator(object1, object2)
    def __create_calculator_distance(self):

        if self.__type == type_matrix.EUCLIDEAN:
                return euclidean_distance_numpy
        elif self.__type == type_matrix.EUCLIDEAN_SQUARE:
                return euclidean_distance_square_numpy
        elif self.__type == type_matrix.MANHATTAN:
                return manhattan_distance_numpy
    def get_distance_type(self):
        return (self.__type)

def euclidean_distance_numpy(object1, object2):
    return numpy.sum(numpy.sqrt(numpy.square(object1 - object2)), axis=1).T
def euclidean_distance_square_numpy(object1, object2):
    return numpy.sum(numpy.square(object1 - object2), axis=1).T
def manhattan_distance_numpy(object1, object2):
    return numpy.sum(numpy.absolute(object1 - object2), axis=1).T