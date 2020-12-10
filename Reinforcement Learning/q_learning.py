import sys
from environment import MountainCar
import numpy as np
from scipy.sparse import coo_matrix
import random


class denseLayer:

    def __init__(self, num_feats, num_neurons, weight_initalization, activation):

        if weight_initalization == 1:
            self.weights = np.random.uniform(low=-.1, high=.1, size=(num_feats + 1, num_neurons))
        elif weight_initalization == 2:
            self.weights = np.zeros([num_feats + 1, num_neurons])
        else:
            raise ('weight_initalization need to be set to random or zero')

        self.activation = activation
        self.inputs = None
        self.outputs = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]

    def update_weights(self, delta, learning_rate):
        self.weights -= (learning_rate * np.outer(self.inputs, delta))

    def forward_pass(self, inputs):

        if inputs.ndim == 1:
            inputs = inputs[None, :]

        inputs_with_bias = np.ones((inputs.shape[0], inputs.shape[1] + 1))
        inputs_with_bias[:, 1:] = inputs
        yhat = np.dot(inputs_with_bias, self.weights)

        if self.activation == 'sigmoid':
            outputs = self.sigmoid(yhat)
        elif self.activation == 'softmax':
            outputs = self.softmax(yhat)
        else:
            # linear
            outputs = yhat

        self.inputs = inputs_with_bias
        self.outputs = outputs

        return outputs


def state_features(state, state_space):
    data = list(state.values())
    col = list(state.keys())
    feature_matrix = coo_matrix((data, ([0]*len(col), col)), shape=(1, state_space))
    return feature_matrix.toarray()

def main(args):
    mode = sys.argv[1]
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])

    car = MountainCar(mode=mode)#, fixed=1)
    current_state = car.reset()

    input_layer = denseLayer(num_feats=car.state_space, num_neurons=3,
                             weight_initalization=2, activation='linear')
    return_list = []
    for i in range(episodes):
        total_rewards = 0

        for j in range(max_iterations):
            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2])
                next_state, reward, end = car.step(action)
            else:
                y_hat = input_layer.forward_pass(state_features(current_state, car.state_space))
                action = np.argmax(y_hat)

                next_state, reward, end = car.step(action)
                target = reward + gamma * input_layer.forward_pass(state_features(next_state, car.state_space))
                delta = y_hat - target
                input_layer.update_weights(delta, learning_rate)

            total_rewards += reward
            current_state = next_state
            if end:
                break

        return_list.append(total_rewards)

    with open(returns_out, 'w') as f:
        for line in return_list:
            print(str(line), file=f)


    with open(weight_out, 'w') as f:
        rows, cols = input_layer.weights.shape
        for i in range(rows):
            if i==0:
                print(str(input_layer.weights[0,0]), file=f)
            else:
                for j in range(cols):
                    print(str(input_layer.weights[i, j]), file=f)

if __name__ == "__main__":
    main(sys.argv)
