import numpy as np
import pandas as pd


def sigmoid(x): return 1 / (1 + np.exp(-x))


def normalize(x): return (x - np.mean(x)) / np.std(x)


def cost_function(y_pred, y_actual, weights, regularization_factor):
    m, decay = y_pred.shape[0], 0
    loss = -1 / m * np.dot(y_actual.T, y_pred) + np.dot(1 - y_actual.T, 1 - y_pred)
    for weight_matrix in weights.values():
        decay += np.sum(np.square(weight_matrix))
    decay = regularization_factor / (2 * m) * decay
    return loss + decay


def forward_propogation(x, weights, bias):
    num_of_layers, prev_layer_activation, activations = len(weights.keys()), x, {}
    for current_layer in range(1, num_of_layers + 1):
        current_layer_total = np.dot(weights[current_layer], prev_layer_activation) + bias[current_layer]
        current_layer_activation = sigmoid(current_layer_total)
        activations[current_layer] = current_layer_activation
        prev_layer_activation = current_layer_activation

    return activations


def get_data(path, instances, normalization=True):
    train_data = pd.read_csv(path)
    train_X, train_Y = train_data.iloc[:instances, 1:], train_data.iloc[:instances, :1]

    if normalization:
        train_X = normalize(train_X)

    return train_X.T, train_Y


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

    m = train_X.shape[0]
    weights, bias = initialize_network(m, [5, 3], 10)
    print("WEIGHTS")
    for layer, weight_matrix in weights.items():
        print("Layer " + str(layer) + ": " + str(weight_matrix.shape))

    print("BIAS")
    for layer, bias_matrix in bias.items():
        print("Layer " + str(layer) + ": " + str(bias_matrix.shape))

    activations = forward_propogation(train_X, weights, bias)
    print("ACTIVATIONS")
    for layer, activation_matrix in activations.items():
        print("Layer " + str(layer) + ": " + str(activation_matrix.shape))
