from scipy.sparse import coo_matrix
import csv
import numpy as np
import sys

def logistic(x):
    return 1/(1+np.exp(-x))

def update_theta(theta, target, features, alpha, N):
    col = [int(x.split(":")[0])for x in features]
    feature_matrix = coo_matrix(([1]*len(col), ([0]*len(col), col)), shape=(1, len(theta)))
    y_hat = logistic(feature_matrix.dot(theta))
    new_theta = theta - ((alpha*(1/N)*(y_hat-target))*feature_matrix.toarray())
    return new_theta[0]

def tran_lr(data_loc, theta, num_epochs, alpha=0.1):
    for i in range(0,num_epochs):
        N = sum(1 for line in open(data_loc))
        with open(data_loc) as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter="\t")
            for line in tsvreader:
                target = int(line[0])
                features = line[1:]
                theta = update_theta(theta, target, features, alpha, N)
    return theta

def predict_lr(data_loc, theta):
    output = []
    with open(data_loc) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            features = line[1:]
            col = [int(x.split(":")[0])for x in features]
            feature_matrix = coo_matrix(([1]*len(col), ([0]*len(col), col)), shape=(1, len(theta)))
            y_hat = logistic(feature_matrix.dot(theta))
            output += [1 if y_hat >= 0.5 else 0]
    return output

def error_rate(data_loc, predicted):
    actual = []
    with open(data_loc) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            target = int(line[0])
            actual += [target]
            
    error_rate = sum(np.array(actual) != np.array(predicted)) / len(predicted)
    return error_rate

if __name__ == '__main__':
    train_loc = sys.argv[1]
    val_loc = sys.argv[2]
    test_loc = sys.argv[3]
    dict_loc = sys.argv[4]

    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    alpha = 0.1
    num_words = sum(1 for line in open(dict_loc))
    theta = [0] * num_words

    trained_weights = tran_lr(train_loc, theta, num_epoch)
    predicted_train = predict_lr(train_loc, trained_weights)
    predicted_test = predict_lr(test_loc, trained_weights)
    train_err = error_rate(train_loc, predicted_train)
    test_err = error_rate(test_loc, predicted_test)

    with open(train_out, 'w') as f:
        for line in predicted_train:
            print(line, file=f)

    with open(test_out, 'w') as f:
        for line in predicted_test:
            print(line, file=f)

    with open(metrics_out, 'w') as f:
        print("""error(train): {}\nerror(test): {}""".format(train_err, test_err), file=f)

    