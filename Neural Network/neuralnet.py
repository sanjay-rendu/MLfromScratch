import numpy as np
import csv
from scipy.sparse import coo_matrix
import sys

def create_features(file_loc):
    with open(file_loc) as csvfile:
        inputs = []
        labels = []
        csvreader = csv.reader(csvfile)
        for line in csvreader:
            target = float(line[0])
            # add 1 for intercept
            features = [float(x) for x in line[1:]]
            labels.append(target)
            inputs.append(features)
    return np.array(inputs), np.array(labels)

class denseLayer:
    
    def __init__(self, num_feats, num_neurons, weight_initalization, activation):
        
        if weight_initalization == 1:
            self.weights = np.random.uniform(low=-.1, high=.1,size=(num_feats+1, num_neurons))
        elif weight_initalization == 2:
            self.weights = np.zeros([num_feats+1, num_neurons])
        else:
            raise('weight_initalization need to be set to random or zero')
            
        self.activation = activation
        self.inputs = None
        self.outputs = None
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1)[:,np.newaxis]
    
    def update_weights(self, delta, learning_rate):
        self.weights -= (learning_rate * np.outer(self.inputs, delta))
        
    def forward_pass(self, inputs):
        
        if inputs.ndim ==1:
            inputs = inputs[None,:]
        
        inputs_with_bias = np.ones((inputs.shape[0],inputs.shape[1]+1))
        inputs_with_bias[:,1:] = inputs
        yhat = np.dot(inputs_with_bias, self.weights)

        if self.activation=='sigmoid':
            outputs = self.sigmoid(yhat)
        elif self.activation=='softmax':
            outputs = self.softmax(yhat)
        else:
            raise('activation need to be set to sigmoid or softmax')
            
        self.inputs = inputs_with_bias
        self.outputs = outputs
        
        return outputs


def error_rate(y,yhat):
    error_rate = sum(y != yhat) / len(y)
    return error_rate

if __name__ == '__main__':
    train_loc = sys.argv[1]
    test_loc = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    train_inputs, train_labels = create_features(train_loc)
    test_inputs, test_labels = create_features(test_loc)

    input_layer = denseLayer(num_feats=train_inputs.shape[1], num_neurons=hidden_units,
                             weight_initalization=init_flag, activation='sigmoid')
    hidden_layer = denseLayer(num_feats=hidden_units, num_neurons=10, weight_initalization=init_flag,
                              activation='softmax')

    N = len(train_labels)
    labels_matrix = coo_matrix(([1] * N, (list(range(0, N)), train_labels)), shape=(N, 10))
    labels_matrix = labels_matrix.toarray()

    test_labels_matrix = coo_matrix(([1] * len(test_labels), (list(range(0, len(test_labels))), test_labels)),
                                    shape=(len(test_labels), 10))
    test_labels_matrix = test_labels_matrix.toarray()

    metrics_result = ''
    for epoch in range(0, num_epoch):
        # SGD
        for i in range(0, N):
            # forward pass
            fwd_pass_input_layer = input_layer.forward_pass(train_inputs[i])
            fwd_pass_hidden_layer = hidden_layer.forward_pass(fwd_pass_input_layer)

            # back propagation
            delta_hidden = fwd_pass_hidden_layer - labels_matrix[i]
            delta_input = np.dot(delta_hidden, hidden_layer.weights[1:, :].T)*fwd_pass_input_layer*(1-fwd_pass_input_layer)
            hidden_layer.update_weights(delta_hidden, learning_rate)
            input_layer.update_weights(delta_input, learning_rate)

        # cross entropy - train
        p_train = hidden_layer.forward_pass(input_layer.forward_pass(train_inputs))
        train_cross_entropy = 0
        for i in range(0, len(p_train)):
            train_cross_entropy += (labels_matrix[i]*np.log(p_train[i])).sum()

        train_cross_entropy = train_cross_entropy/len(p_train)
        # cross entropy - test
        p_test = hidden_layer.forward_pass(input_layer.forward_pass(test_inputs))
        test_cross_entropy = 0
        for i in range(0, len(p_test)):
            test_cross_entropy += (test_labels_matrix[i] * np.log(p_test[i])).sum()

        test_cross_entropy = test_cross_entropy / len(p_test)

        metrics_result += """epoch={} crossentropy(train): {}\nepoch={} crossentropy(validation): {}\n""".format(
            epoch, -1*train_cross_entropy, epoch, -1*test_cross_entropy)


    train_pred = np.argmax(hidden_layer.forward_pass(input_layer.forward_pass(train_inputs)),axis=1)
    test_pred = np.argmax(hidden_layer.forward_pass(input_layer.forward_pass(test_inputs)),axis=1)

    with open(train_out, 'w') as f:
        for line in train_pred:
            print(line, file=f)

    with open(test_out, 'w') as f:
        for line in test_pred:
            print(line, file=f)

    train_err = error_rate(train_labels,train_pred)
    test_err = error_rate(test_labels, test_pred)
    with open(metrics_out, 'w') as f:
        print(metrics_result + """error(train): {}\nerror(validation): {}""".format(train_err, test_err), file=f)


