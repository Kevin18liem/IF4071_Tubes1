from __future__ import division
import numpy as np

class Multi_Layer_NN:

	def __init__(self, data, num_input_node, hidden_node, 
		num_output_node, num_batch, 
		learning_rate_const, tolerance_const, **kwargs):
		
		self.instance = data
		self.input_node_size = num_input_node
		self.w_hidden_node = hidden_node		
		self.output_node_size = num_output_node
		# Gradient Descent Parameters
		self.batch_size = num_batch
		self.learning_rate = learning_rate_const
		self.tolerant = tolerance_const
		self.momentum = kwargs.get('momentum', 0)
		self.epochs = kwargs.get('epochs', 10)

	def train(self, instance_target):
		batch_iteration = int(np.ceil(self.instance.shape[0] / self.batch_size))
		old_loss = -np.inf

		for step in range(epochs):
			loss = 0
			for i in range(batch_iteration):
				start_index = i * self.batch_size
				end_index = i * self.batch_size + self.batch_size

	def feed_forward(self, w_output):
		s = list()
		o = list()
		# Feed Forward Hidden Node
		s_temp = self.instance.dot(self.w_hidden_node[0].T)
		o_temp = self.sigmoid(s_temp)
		s.append(s_temp)
		o.append(o_temp)
		iteration = len(self.w_hidden_node)
		for i in range(1,iteration):
			s_temp = o[i-1].dot(self.w_hidden_node[i].T)
			o_temp = self.sigmoid(s_temp)
			o.append(o_temp)
		# Feed Forward Output Node
		s_out = o[-1].dot(w_output.T)
		o_out = self.sigmoid(s_out)
		return s, o, s_out, o_out

	def sigmoid(self, X):
	    output = 1 / (1 + np.exp(-X))
	    return np.matrix(output)

	def back_propagation(self, instance_target, o, o_out, w_output):
		d = list()
		d_temp = np.multiply(np.multiply(o_out, 1-o_out), instance_target-o_out)
		d.insert(0, d_temp)
		d_temp = np.multiply(np.multiply(o[-1], 1-o[-1]), (w_output.T.dot(d[0].T)).T)
		d.insert(0, d_temp)
		iteration = len(self.w_hidden_node)
		for i in range(iteration-1,0,-1):
			print('o[i-1]', o[i-1], '1-o[i-1]', 1-o[i-1])
			d_temp = np.multiply(np.multiply(o[i-1], 1-o[i-1]), (self.w_hidden_node[i].T.dot(d[0].T)).T)
			d.insert(0, d_temp)
		return d

	def gradient_descent(self, instance_target, w_output):
		_,o,_,o_out = feed_forward(instance, self.w_hidden_node, w_output)
		
		# Back-Propagation		
		vector_of_one = np.ones((o_out.shape[0],1))

		# Calculate delta target and output
		delta_t_o = np.matrix(np.diag(np.dot(o_out, (vector_of_one-o_out).T))).T
		delta_t_o = np.matrix(np.diag(delta_t_o.dot((instance_target - o_out).T))).T

		# Calculate delta hidden layer

	def sigmoid_output_to_derivative(output):
		return np.multiply(output, (1-output))

    # def concat_ones_vector(X):
    # 	ones_vector = np.ones((X.shape[0], 1))
    # 	return np.concatenate((ones_vector, X), axis=1)	
# def sigmoid(X):
#     output = 1 / (1 + np.exp(-X))
#     return np.matrix(output)

# Main
np.random.seed(0)

W1 = np.random.randn(4, 4) / np.sqrt(4)
print('w1', W1)
W2 = np.random.randn(3, 4) / np.sqrt(3)
print('w2', W2)
w_output = np.random.randn(1, 3)
print('wout', w_output)

instance = np.matrix([[1,2,3,4],[4,5,6,7]])
print('instance ', instance)
instance_target = np.matrix([[2],[5]])
print('instance_target', instance_target)

w_hidden_node = list()
# o = list()
# d = list()

w_hidden_node.append(W1)
# b1 = instance.dot(w_hidden_node[0].T)
# c1 = np.matrix(1 / (1 + np.exp(-b1)))
# o.append(c1)
# print('output layer1 ', c1)

w_hidden_node.append(W2)
# b2 = c1.dot(w_hidden_node[1].T)
# c2 = np.matrix(1 / (1 + np.exp(-b2)))
# o.append(c2)
# print('output layer2 ', c2)

# b3 = c2.dot(w_output.T)
# c3 = np.matrix(1 / (1 + np.exp(-b3)))
# o_out = c3
# print('output layerout ', c3)

# d3 = np.multiply(np.multiply(c3, 1-c3), instance_target-o_out)
# d.insert(0, d3)
# print('delta layerout ', d3)

# d2 = np.multiply(np.multiply(c2, 1-c2), (w_output.T.dot(d[0].T)).T)
# d.insert(0, d2)
# print('delta layer2', d2)

# d1 = np.multiply(np.multiply(c1, 1-c1), (W2.T.dot(d[0].T)).T)
# d.insert(0, d1)
# print('delta layer1', d1)

# Main
multiLayerNN = Multi_Layer_NN(instance, 4, w_hidden_node, 1, 2, 0.5, 0.5, momentum = 0.001, epochs = 2)
s, o, s_out, o_out = multiLayerNN.feed_forward(w_output)
print('Feed forward ', s, o, s_out, o_out)
print('o', o)
d = multiLayerNN.back_propagation(instance_target, o, o_out, w_output)
print('Back propagation ', d)