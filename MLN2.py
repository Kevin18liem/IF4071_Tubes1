# Backpropagation class 

class BackPropagation: 
n_inputs = 1 
n_hiddens = [1] 
n_outputs = 1 
l_rate = 0.5 
n_epoch = 20 
batch_size = 1 
momentum = 1 
network = list() 

def __init__(self, n_inputs, n_hiddens, l_rate, n_epoch, batch_size, momentum): 
self.n_inputs = n_inputs 
self.n_hiddens = n_hiddens 
self.l_rate = l_rate 
self.n_epoch = n_epoch 
self.batch_size = batch_size 
self.momentum = momentum 
self.network = list() 

hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hiddens[0])] 
self.network.append(hidden_layer) 

for i in range(1, min(len(n_hiddens), 10)): 
hidden_layer = [{'weights':[random() for j in range(self.n_hiddens[i-1] + 1)]} for i in range(self.n_hiddens[i])] 
self.network.append(hidden_layer) 

output_layer = [{'weights':[random() for i in range(self.n_hiddens[-1] + 1)]} for i in range(self.n_outputs)] 
self.network.append(output_layer) 

def activate(self, weights, inputs): 
activation = weights[-1] 
for i in range(len(weights)-1): 
activation += weights[i] * inputs[i] 
return activation 

def transfer(self, activation): 
return 1.0 / (1.0 + np.exp(-activation)) 

def forward_propagate(self, network, init_input): 
inputs = init_input 
for layer in network: 
new_inputs = [] 
for neuron in layer: 
activation = self.activate(neuron['weights'], inputs) 
neuron['output'] = self.transfer(activation) 
new_inputs.append(neuron['output']) 
inputs = new_inputs 
return inputs 

def transfer_derivative(self, output): 
return output * (1.0 - output) 

def backward_propagate_error(self, network, expected): 
for i in reversed(range(len(network))): 
layer = network[i] 
errors = list() 
if i != len(network)-1: 
for j in range(len(layer)): 
error = 0.0 
for neuron in network[i + 1]: 
error += (neuron['weights'][j] * neuron['delta']) 
errors.append(error) 
else: 
for j in range(len(layer)): 
neuron = layer[j] 
a = 1 
errors.append(expected[j] - neuron['output']) 
for j in range(len(layer)): 
neuron = layer[j] 
neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output']) 

def initialize_delta_weight(self, network): 
delta_weight = [] 
for layer in network: 
delta_weight.append([]) 
for neuron in layer: 
delta_weight[-1].append(np.zeros(len(neuron['weights']))) 
return delta_weight 

def accumulate_delta_weight(self, network, init_input, l_rate, momentum, delta_weight): 
for i in range(len(network)): 
inputs = init_input[:-1] 
if i != 0: 
inputs = [neuron['output'] for neuron in network[i - 1]] 
for j in range(len(network[i])): 
for k in range(len(inputs)): 
delta_weight[i][j][k] += (l_rate * network[i][j]['delta'] * inputs[k] + momentum * network[i][j]['weights'][k]) 
delta_weight[i][j][-1] += (l_rate * network[i][j]['delta'] + momentum * network[i][j]['weights'][-1]) 

def update_weight(self, network, delta_weight): 
for i in range(len(network)): 
for j in range(len(network[i])): 
for k in range(len(network[i][j]['weights'])): 
network[i][j]['weights'][k] += delta_weight[i][j][k] 

def fit(self, init_inputs_x, init_inputs_y, validation_data = None): 
init_inputs = list() 
for i in range(0,len(init_inputs_x)): 
init_inputs.append(np.append(init_inputs_x[i],init_inputs_y[i])) 
for epoch in range(self.n_epoch): 
sum_error = 0 
error_calculated = 0 
batch_member_calculated = 0 
delta_weight = self.initialize_delta_weight(self.network) 
for init_data in init_inputs: 
outputs = self.forward_propagate(self.network, init_data) 
expected = [0 for i in range(self.n_outputs)] 
expected[0] = init_data[-1] 
sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))]) 
error_calculated += 1 
self.backward_propagate_error(self.network, expected) 

self.accumulate_delta_weight(self.network, init_data, self.l_rate, self.momentum, delta_weight) 
batch_member_calculated += 1 

if batch_member_calculated == self.batch_size: 
self.update_weight(self.network, delta_weight) 
batch_member_calculated = 0 
delta_weight = self.initialize_delta_weight(self.network) 
if validation_data != None: 
pred = self.predict(validation_data[0]) 
val_score = accuracy_score(pred, validation_data[1]) 
print('>epoch=%d, error=%.4f, val_score=%.2f, outputs=%.2f' % (epoch, sum_error/error_calculated, val_score, outputs[0])) 
else: 
print('>epoch=%d, error=%.4f' % (epoch, sum_error/error_calculated)) 
tempnetwork = list() 
for layer in self.network: 
templayer = list() 
for neuron in layer: 
templayer.append({'weights' : neuron['weights']}) 
tempnetwork.append(templayer) 
self.network = tempnetwork 

def predict(self, init_inputs): 
labels = np.zeros(len(init_inputs)) 
for i in range(0,len(init_inputs)): 
labels[i] = self.forward_propagate(self.network, init_inputs[i])[0] 
return labels