import numpy as np
import pandas as pd


def sigmoid(x): return 1 / (1 + np.exp(-x))


def normalize(x): return (x - np.mean(x)) / np.std(x)


def forward_propogation(x, weights):
    num_of_layers, prev_layer_activation, activations = len(weights.keys()), x, {}
    for current_layer in range(1, num_of_layers + 1):
        current_layer_total = np.dot(weights[current_layer], prev_layer_activation)
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
    input_layer, output_layer = num_of_features, num_of_outputs
    weights, prev_layer_nodes = {}, input_layer
    for index in range(len(hidden_layers_nodes)):
        current_layer_nodes = hidden_layers_nodes[index]
        current_layer_weights = np.random.rand(current_layer_nodes, prev_layer_nodes)
        weights[index + 1] = current_layer_weights
        prev_layer_nodes = current_layer_nodes

    output_layer_weights = np.random.rand(output_layer, prev_layer_nodes)
    weights[len(hidden_layers_nodes) + 1] = output_layer_weights
    return weights


if __name__ == "__main__":
    print("Starting...")
    train_data_path = "../../data/hand-written-digit-recognition-train.csv"
    instances = 10000

    train_X, train_Y = get_data(train_data_path, instances)
    print(train_X.shape)
    print(train_Y.shape)

    m = train_X.shape[0]
    weights = initialize_network(m, [5, 3], 10)
    print("WEIGHTS")
    for layer, weight_matrix in weights.items():
        print("Layer " + str(layer) + ": " + str(weight_matrix.shape))

    activations = forward_propogation(train_X, weights)
    print("ACTIVATIONS")
    for layer, activation_matrix in activations.items():
        print("Layer " + str(layer) + ": " + str(activation_matrix.shape))
