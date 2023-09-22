import numpy as np

def sigmoidActivate(np_arr):
    return 1 / (1 + np.exp(np_arr))

def sigmoidActivateGradient(np_arr):
    return sigmoidActivate(np_arr) * (1 - sigmoidActivate(np_arr))

class Network:
    def __init__(self, structure):
        self.structure = structure
        self.size = len(structure)
        self.bias = [np.random.normal(0, 1, size=(size)) for size in structure[1:]]
        self.weights = [np.random.normal(0, 1/np.sqrt(structure[layer]), size=(structure[layer + 1], structure[layer])) for layer in range(self.size - 1)]

    def activate(self, inputs):
        for layer in range(self.size - 1):
            inputs = sigmoidActivate(np.dot(self.weights[layer], inputs) + self.bias[layer])
        return inputs
    
    def learn(self, training_data, learn_rate, lmbda):
        weight_alterations = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights]
        bias_alterations = [np.zeros(bias_matrix.shape) for bias_matrix in self.bias]
        training_data_size = len(training_data)

        for training_pair in training_data:
            inputs = training_pair[0]
            expected_output = training_pair[1]

            inputs_history = [inputs]
            current_layer_inputs = inputs
            for layer in range(self.size - 1):
                current_layer_output = np.dot(self.weights[layer], current_layer_inputs) + self.bias[layer]
                inputs_history.append(current_layer_output)
                current_layer_inputs = sigmoidActivate(current_layer_output)

            current_layer_cost_node_gradient = np.array(2 * (current_layer_inputs - expected_output))
            for layer in range(self.size - 1):
                bias_alterations[-(layer + 1)] -= current_layer_cost_node_gradient
                weight_alterations[-(layer + 1)] -= np.dot(current_layer_cost_node_gradient.reshape((self.structure[-(layer+1)], 1)), inputs_history[-(layer + 2)].reshape((1, self.structure[-(layer+2)])))
                current_layer_cost_node_gradient = np.array(np.dot(self.weights[-(layer + 1)].T, current_layer_cost_node_gradient) * sigmoidActivateGradient(inputs_history[-(layer + 2)]))

        for layer in range(self.size - 1):
            weight_alterations[layer] *= learn_rate / training_data_size
            bias_alterations[layer] *= learn_rate / training_data_size
            self.weights[layer] *= 1 - lmbda * learn_rate / training_data_size 
            self.weights[layer] += weight_alterations[layer]
            self.bias[layer] += bias_alterations[layer]