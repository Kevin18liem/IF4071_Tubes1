from __future__ import division
import numpy as np

class Multi_Layer_NN:

	def __init__(self, data, num_input_node, num_hidden_node, 
		num_output_node, num_batch, 
		learning_rate_const, tolerance_const, **kwargs):
		
		self.instance = data
		self.input_node_size = num_input_node
		self.hidden_node = hidden_node		
		self.output_node_size = num_output_node
		# Gradient Descent Parameters
		self.batch_size = num_batch
		self.learning_rate = learning_rate_const
		self.tolerant = tolerant_const
		self.momentum = kwargs.get('momentum', 0)
		self.epochs = kwargs.get('epochs', 10)

	def train(self,instance, instance_target):
		batch_iteration = int(np.ceil(instance.shape[0] / self.batch_size))
		old_loss = -np.inf

		for step in range(epochs):
			loss = 0
			for i in range(batch_iteration):
				start_index = i * self.batch_size
				end_index = i * self.batch_size + self.batch_size

	def feed_forward(instance,w_hidden_node, w_output):
		s = list()
		o = list()
		# Feed Forward Hidden Node
		s_temp = instance.dot(w_hidden_node[0].T)
		o_temp = self.sigmoid(s_temp)
		s.append(s_temp)
		o.append(o_temp)
		iteration = len(w_hidden_node)
		for i in range(1,iteration):
			s_temp = o[i-1].dot(w_hidden_node[i].T)
			s.append(s_temp)
			o_temp = self.sigmoid(s_temp)
			o.append(o_temp)
		# Feed Forward Output Node
		s_out = o[-1].dot(w_output.T)
		o_out = self.sigmoid(s_out)
		return s,o,s_out,o_out

	def sigmoid(self,X):
	    output = 1 / (1 + np.exp(-X))
	    return np.matrix(output)

	def gradient_descent(instance, target, w_hidden_node, w_output):
		_,o,_,o_out = feed_forward(instance, w_hidden_node, w_output)
		
		# Back-Propagation		
		vector_of_one = np.ones((o_out.shape[0],1))

		# Calculate delta target and output
		delta_t_o = np.matrix(np.diag(np.dot(o_out, (vector_of_one-o_out).T))).T
		delta_t_o = np.matrix(np.diag(delta_t_o.dot((target - o_out).T))).T

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
# np.random.seed(0)
# W1 = np.random.randn(4, 4) / np.sqrt(4)
# print (W1.T)
# W2 = np.random.randn(3,4) / np.sqrt(3)
# print (W2.T)

# a = np.matrix([[1,2,3,4],[4,5,6,7]])
# w_hidden_node = list()
# w_hidden_node.append(W1)
# b = a.dot(w_hidden_node[0].T)
# print(b)
# print(b.item((0,0)))

# w_hidden_node.append(W2)
# print(w_hidden_node.shape)
# b = a.dot(w_hidden_node[1].T)
# print (sigmoid(b))
# vector_of_one = np.ones((2,1))
# o_out = np.matrix([[2],[3]])
# target = np.matrix([[3],[5]])
# print (vector_of_one-o_out)
# delta_t_o = np.matrix(np.diag(np.dot(o_out, (vector_of_one-o_out).T))).T
# print (delta_t_o)
# delta_t_o = np.matrix(np.diag(delta_t_o.dot((target - o_out).T))).T
# print(delta_t_o)
