import math
import collections

class dbscan:
	def __init__(self, data, epsilon, min_pts, distance_type=1):
		self.data = data
		self.clusters = []
		self.clusters_id = []
		self.epsilon = epsilon
		self.min_pts = min_pts
		self.neighborhood_list = [[] for _ in self.data]
		self.distance_type = distance_type
		self.core_points = []
		self.outlier = []

	def compare(self, cluster1, cluster2):
		first_element = cluster1[0]

		if first_element in cluster2:
			return True
		else:
			return False

	def process(self):
		for i in range(0, len(self.data)):
			for j in range(0, len(self.data)):
				if i!=j and self.calculate_distance(self.data[i], self.data[j]) <= self.epsilon:
					self.neighborhood_list[i].append(j)

		for i in range(0, len(self.neighborhood_list)):
			if len(self.neighborhood_list[i]) + 1 >= self.min_pts:
				self.core_points.append(i)
			else:
				self.outlier.append(i)

		for i in range(0, len(self.core_points)):
			found = False
			current_cluster = self.neighborhood_list[i]
			current_cluster.append(i)

			for cluster in self.clusters:
				if self.compare(cluster, current_cluster):
					found = True
					break

			if not found:
				self.clusters.append(current_cluster)

	def calculate_distance(self, instance1, instance2):
		EUCLIDEAN = 1
		EUCLIDEAN_SQUARE = 2
		MANHATTAN = 3

		diff = 0

		if self.distance_type == EUCLIDEAN :
			for val1, val2 in zip(instance1, instance2):
				diff += (val1 - val2)**2
			diff = math.sqrt(diff)
		elif self.distance_type == EUCLIDEAN_SQUARE :
			for val1, val2 in zip(instance1, instance2):
				diff += (val1 - val2)**2
		else:
			for val1, val2 in zip(instance1, instance2):
				diff += math.fabs(val1 - val2)
		
		return diff

	def get_clusters(self):
		return self.clusters

	def get_outliers(self):
		return self.outlier

