import numpy as np

class MultiLayerNN:

    def __init__(self, data, hidden_node, output_node,
        num_batch, learning_rate_const, tolerance_const,
        **kwargs):
        
        self.instance = data
        self.w_hidden_node = hidden_node
        self.w_output_node = output_node
        # Gradient Descent Parameters
        self.batch_size = num_batch
        self.learning_rate = learning_rate_const
        self.tolerance = tolerance_const
        self.momentum = kwargs.get('momentum', 0)
        self.epochs = kwargs.get('epochs', 10)

    def train(self, instance_target):
        instance_target_t = np.array([instance_target]).T
        batch_iteration = int(np.ceil(self.instance.shape[0] / self.batch_size))
        old_loss = -np.inf
        for step in range(self.epochs):
            loss = 0
            for i in range(batch_iteration):
                start_index = i * self.batch_size
                end_index = i * self.batch_size + self.batch_size
                if end_index > len(instance_target_t):
                    end_index = len(instance_target_t)
                self.gradient_descent(self.instance[start_index:end_index:1], instance_target_t[start_index:end_index:1])

    def feed_forward(self, instance):
        s = list()
        o = list()
        # Feed Forward Hidden Node
        s_temp = instance.dot(self.w_hidden_node[0].T)
        o_temp = self.sigmoid(s_temp)
        s.append(s_temp)
        o.append(o_temp)
        iteration = len(self.w_hidden_node)
        for i in range(1,iteration):
            s_temp = o[i-1].dot(self.w_hidden_node[i].T)
            o_temp = self.sigmoid(s_temp)
            o.append(o_temp)
        # Feed Forward Output Node
        s_out = o[-1].dot(self.w_output_node.T)
        o_out = self.sigmoid(s_out)
        return s, o, s_out, o_out

    def sigmoid(self, X):
        output = 1 / (1 + np.exp(-X))
        return np.matrix(output)

    def back_propagation(self, instance_target, o, o_out):
        d = list()
        # Back Propagation Output Node
        d_temp = np.multiply(np.multiply(o_out, 1-o_out), instance_target.T-o_out)
        d.insert(0, d_temp)
        # Back Propagation Hidden Node
        d_temp = np.multiply(np.multiply(o[-1], 1-o[-1]), (self.w_output_node.T.dot(d[0].T)).T)
        d.insert(0, d_temp)
        iteration = len(self.w_hidden_node)
        for i in range(iteration-1, 0, -1):
            d_temp = np.multiply(np.multiply(o[i-1], 1-o[i-1]), (self.w_hidden_node[i].T.dot(d[0].T)).T)
            d.insert(0, d_temp)
        return d

    def update_weight(self, instance, o, d):
        # Update Weight Output Node
        self.w_output_node[0] = self.w_output_node[0] + self.momentum * self.w_output_node[0] + self.learning_rate * d[-1].T.dot(o[-1])
        # Update Weight Hidden Node
        iteration = len(self.w_hidden_node)
        for i in range(iteration-1, 0, -1):
            self.w_hidden_node[i] = self.w_hidden_node[i] + self.momentum * self.w_hidden_node[i] + self.learning_rate * d[i].T.dot(o[i-1])
        self.w_hidden_node[0] = self.w_hidden_node[0] + self.momentum * self.w_hidden_node[0] + self.learning_rate * d[0].T.dot(instance)

    def gradient_descent(self, instance, instance_target):
        # Feed Forward
        _,o,_,o_out = self.feed_forward(instance)
        # Back Propagation      
        d = self.back_propagation(instance_target, o, o_out)
        # Update Weight
        self.update_weight(instance, o, d)

# Main
instance = np.matrix([[1,2,3,4],[4,5,6,7]])
print('instance ', instance)
instance_target = [2, 5]
print('instance_target', instance_target)

np.random.seed(0)
W1 = np.random.randn(4, 4) / np.sqrt(4)
W2 = np.random.randn(3, 4) / np.sqrt(3)
w_output_node = np.random.randn(1, 3)
w_hidden_node = list()
w_hidden_node.append(W1)
w_hidden_node.append(W2)
print('w ', w_hidden_node)
print('wo ', w_output_node)

multiLayerNN = MultiLayerNN(instance, w_hidden_node, w_output_node, 1, 0.5, 0.5, momentum = 0.001, epochs = 2)
multiLayerNN.train(instance_target)
print('w ', multiLayerNN.w_hidden_node)
print('wo ',multiLayerNN. w_output_node)