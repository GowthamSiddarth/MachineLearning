import numpy as np
import pandas as pd


def sigmoid(x): return 1 / (1 + np.exp(-x))


def normalize(x): return (x - np.mean(x)) / np.std(x)


def encode_labels(y):
    z = np.zeros((10, y.shape[0]))
    z[y.values.reshape(y.shape[0]), range(y.shape[0])] = 1
    return z


def predict(output_layer_activations):
    return np.exp(output_layer_activations) / np.sum(np.exp(output_layer_activations), axis=0, keepdims=True)


def get_data(path, instances, normalization=True):
    train_data = pd.read_csv(path, skiprows=1)
    train_X, train_Y = train_data.iloc[:instances, 1:], train_data.iloc[:instances, :1]

    if normalization:
        train_X = normalize(train_X)
    train_Y = encode_labels(train_Y)

    return train_X.T, train_Y


def cost_function(y_predictions, y_actual, weights, regularization_factor):
    m, decay = y_predictions.shape[1], 0
    loss = -1 / m * np.sum(
        np.dot(np.log(y_predictions), y_actual.T) + np.dot(np.log(1 - y_predictions), 1 - y_actual.T))
    print("loss = " + str(loss))
    for weight_matrix in weights.values():
        decay += np.sum(np.square(weight_matrix))
    decay = (regularization_factor / (2 * m)) * decay
    print("decay = " + str(decay))
    return loss + decay


def forward_propagation(x, weights, bias):
    num_of_layers, prev_layer_activation, activations = len(weights.keys()), x, {0: x}
    for current_layer in range(1, num_of_layers + 1):
        current_layer_total = np.dot(weights[current_layer], prev_layer_activation) + bias[current_layer]
        current_layer_activation = sigmoid(current_layer_total)
        activations[current_layer] = current_layer_activation
        prev_layer_activation = current_layer_activation

    return activations


def backward_propagation(activations, y, weights, regularization_factor):
    output_layer_predictions = predict(activations[len(activations) - 1])
    deltas, weights_derivatives, bias_derivatives = {len(activations): output_layer_predictions - y}, {}, {}
    for layer in list(range(len(activations) - 1, 0, -1)):
        deltas[layer] = np.multiply(np.dot(weights[layer].T, deltas[layer + 1]),
                                    np.multiply(activations[layer - 1], 1 - activations[layer - 1]))
        weights_derivatives[layer] = np.dot(deltas[layer + 1], activations[layer - 1].T) + regularization_factor * \
                                                                                           weights[
                                                                                               layer]
        bias_derivatives[layer] = np.sum(deltas[layer + 1], axis=1).reshape((deltas[layer + 1].shape[0], 1))

    return weights_derivatives, bias_derivatives


def update_weights_and_bias(weights, weights_derivatives, bias, bias_derivatives, learning_rate):
    for layer in weights.keys():
        weights[layer] = weights[layer] - learning_rate * weights_derivatives[layer]
        bias[layer] = bias[layer] - learning_rate * bias_derivatives[layer]

    return weights, bias


def train_network(x, y, weights, bias, learning_rate, regularization_factor, iterations):
    cost_history = []
    for iteration in range(1, iterations + 1):
        print("Iteration: " + str(iteration))
        activations = forward_propagation(x, weights, bias)
        for layer in activations.keys():
            print("Activations " + str(layer))
            print(activations[layer])
        weights_derivatives, bias_derivatives = backward_propagation(activations, y, weights, regularization_factor)
        weights, bias = update_weights_and_bias(weights, weights_derivatives, bias, bias_derivatives, learning_rate)

        cost = cost_function(activations[len(activations) - 1], y, weights, regularization_factor)
        cost_history.append(cost)

        if 0 == iteration % 100:
            print("Cost at iteration " + str(iteration) + ": " + str(cost))
    return weights, bias, cost_history


def initialize_network(num_of_features, hidden_layers_nodes, num_of_outputs):
    input_layer = num_of_features
    weights, bias, prev_layer_nodes = {}, {}, input_layer
    for index in range(len(hidden_layers_nodes)):
        current_layer_nodes = hidden_layers_nodes[index]
        current_layer_weights = np.random.rand(current_layer_nodes, prev_layer_nodes)
        current_layer_bias = np.random.rand(current_layer_nodes, 1)
        weights[index + 1], bias[index + 1] = current_layer_weights, current_layer_bias
        prev_layer_nodes = current_layer_nodes

    output_layer_weights = np.random.rand(num_of_outputs, prev_layer_nodes)
    output_layer_bias = np.random.rand(num_of_outputs, 1)
    weights[len(hidden_layers_nodes) + 1], bias[len(hidden_layers_nodes) + 1] = output_layer_weights, output_layer_bias
    return weights, bias


if __name__ == "__main__":
    print("Starting...")
    train_data_path = "../../data/hand-written-digit-recognition-train.csv"
    instances = 10000

    train_X, train_Y = get_data(train_data_path, instances)
    print(train_X.shape)
    print(train_Y.shape)
    print(np.sum(train_Y))

    m = train_X.shape[0]
    weights, bias = initialize_network(m, [5, 3], 10)
    print("WEIGHTS")
    for layer, weight_matrix in weights.items():
        print("Layer " + str(layer) + ": " + str(weight_matrix.shape))

    print("BIAS")
    for layer, bias_matrix in bias.items():
        print("Layer " + str(layer) + ": " + str(bias_matrix.shape))

    learning_rate, regularization_factor, iterations = 0.1, 0.01, 2000
    weights, bias, cost_history = train_network(train_X, train_Y, weights, bias, learning_rate,
                                                regularization_factor, iterations)
