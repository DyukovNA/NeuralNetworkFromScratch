import numpy as np


def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return total, sigmoid(total)

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias)

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x)[1])    # 0.9990889488055994


def mse_loss(out_true, out_pred):
    if len(out_true) != len(out_pred):
        raise Exception('Arrays of different lengths')
    return ((out_true - out_pred) ** 2).mean()


class Network:
    def __init__(self, num_of_nodes):
        weights = [np.random.normal()] * num_of_nodes
        bias = 0

        self.neurons = []
        for i in range(num_of_nodes):
            self.neurons.append(Neuron(weights, bias))

        self.out = Neuron(weights, bias)


    def feedforward(self, input):
        # Hidden layer
        out_neurons = []
        for i in range(len(self.neurons)):
            out_neurons.append(self.neurons[i].feedforward(input)[1])

        # Output
        out = self.out.feedforward(np.array(out_neurons)[1])

        return out

    def train(self, data, true_outs, epochs, learn_rate):
        for epoch in range(epochs):
            for x, y_true in zip(data, true_outs):
                out_neurons = []
                to_out = []
                for i in range(len(self.neurons)):
                    out_neurons.append(self.neurons[i].feedforward(x))
                    to_out.append(self.neurons[i].feedforward(x)[1])

                out = self.out.feedforward(np.array(to_out))
                y_pred = out[1]

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_w = []
                for neuron in out_neurons:
                    d_ypred_d_w.append(neuron[0] * deriv_sigmoid(neuron[1]))

                d_ypred_d_b_out = deriv_sigmoid(out[0])

                # Neuron o1
                for i in range(len(self.out.weights)):
                    self.out.weights[i] -= learn_rate * d_L_d_ypred * d_ypred_d_w[i]
                self.out.bias -= learn_rate * d_L_d_ypred * d_ypred_d_b_out

                for i in range(len(self.neurons)):
                    sum = out_neurons[i][0]
                    d_h1_d_w1 = x[0] * deriv_sigmoid(sum)
                    d_h1_d_w2 = x[1] * deriv_sigmoid(sum)
                    d_h1_d_b = deriv_sigmoid(sum)
                    d_ypred_d_h = self.out.weights[i] * deriv_sigmoid(out[0])
                    self.neurons[i].weights[0] -= learn_rate * d_L_d_ypred * d_ypred_d_h * d_h1_d_w1
                    self.neurons[i].weights[1] -= learn_rate * d_L_d_ypred * d_ypred_d_h * d_h1_d_w2
                    self.neurons[i].bias -= learn_rate * d_L_d_ypred * d_ypred_d_h * d_h1_d_b
        if epoch % 10 == 0:
            y_preds = np.apply_along_axis(self.feedforward, 1, data)
            loss = mse_loss(true_outs, y_preds)
            print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = Network(2)
network.train(data, all_y_trues, 10000, 0.1)

emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)[1]) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)[1]) # 0.039 - M
