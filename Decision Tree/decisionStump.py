import sys
import numpy as np
from scipy import stats


class decision_stump:

    def __init__(self, train_data, attr_index):
        self.predictions = {}
        self.attr_index = int(attr_index)
        self.train_data = train_data
        self.left_val, self.right_val = np.unique(train_data[:, self.attr_index])

    def train(self):
        train_data = self.train_data

        left_vote = stats.mode(train_data[:, -1][train_data[:, self.attr_index] == self.left_val]).mode[0]
        right_vote = stats.mode(train_data[:, -1][train_data[:, self.attr_index] == self.right_val]).mode[0]
        self.predictions = {self.left_val: left_vote, self.right_val: right_vote}

    def predict(self, data):

        if self.predictions == {}:
            print("Train the model first!")
        else:
            predict_data = data[:, self.attr_index]
            predict_data = np.where(predict_data == self.left_val, self.predictions[self.left_val],
                                    self.predictions[self.right_val])
            return predict_data

if __name__ == '__main__':

    train_loc = sys.argv[1]
    test_loc = sys.argv[2]
    attr_index = sys.argv[3]

    pred_train_loc = sys.argv[4]
    pred_test_loc = sys.argv[5]

    metrics_loc = sys.argv[6]

    train_df = np.genfromtxt(fname=train_loc, delimiter="\t", skip_header=1, dtype="|U10")
    test_df = np.genfromtxt(fname=test_loc, delimiter="\t", skip_header=1, dtype="|U10")

    stump = decision_stump(train_df, attr_index)
    stump.train()
    train_predictions = stump.predict(train_df)
    test_predictions = stump.predict(test_df)

    np.savetxt(pred_train_loc,train_predictions, fmt="%s")
    np.savetxt(pred_test_loc, test_predictions, fmt="%s")

    y = train_df[:,-1]
    y_hat = train_predictions
    error_train = sum(y != y_hat) / train_df.shape[0]

    y = test_df[:,-1]
    y_hat = test_predictions
    error_test = sum(y != y_hat) / train_df.shape[0]

    with open(metrics_loc, 'w') as f:
        print("""error(train): {}\nerror(test): {}""".format(error_train, error_test), file=f)

