import sys
import numpy as np
from scipy import stats
import io
from contextlib import redirect_stdout
from scipy.maxentropy.basemodel import cro

def label_counts(data):
    unique, frequency = np.unique(data[:, -1], return_counts=True)
    return list(zip(frequency, unique))


def print_counts(zip_list):
    if len(zip_list) == 2:
        print("[{} {}/{} {}]".format(zip_list[0][0], zip_list[0][1], zip_list[1][0], zip_list[1][1]))
    else:
        print("[{} {}]".format(zip_list[0][0], zip_list[0][1]))


def mutual_info(y, x):
    y_0 = y[np.where(y == 0)].shape[0]
    y_1 = y[np.where(y == 1)].shape[0]
    y_all = y.shape[0]

    h_y = stats.entropy([y_0 / y_all, y_1 / y_all], base=2)

    sum_y_x = 0
    for i in np.unique(x):
        # print(i)
        p_i = x[np.where(x == i)].shape[0] / x.shape[0]

        y_given_x = y[np.where(x == i)]
        total = y_given_x.shape[0]

        zeros = y_given_x[np.where(y_given_x == 0)].shape[0]
        ones = y_given_x[np.where(y_given_x == 1)].shape[0]

        h_y_given_x = stats.entropy([zeros / total, ones / total], base=2)
        sum_y_x += p_i * h_y_given_x

    return h_y - sum_y_x


def split_criteria(train_df):
    y = train_df[:, -1]
    y = np.where(y == np.unique(y)[0], 0, 1)
    mutual_info_dict = {}
    for i in range(0, train_df.shape[1] - 1):
        x = train_df[:, i]
        x = np.where(x == np.unique(x)[0], 0, 1)
        mutual_info_dict[i] = mutual_info(y, x)
    split_index = max(mutual_info_dict, key=mutual_info_dict.get)
    return split_index, mutual_info_dict[split_index]


def splitter(data, split_index):
    left_val, right_val = np.unique(data[:, split_index])

    left_data = data[np.where(data[:, split_index] == left_val)]
    right_data = data[np.where(data[:, split_index] == right_val)]

    return {left_val: left_data, right_val: right_data}


class DecisionNode:

    def __init__(self, split_attr, left_branch, right_branch, left_val, right_val):
        self.split_attr = split_attr
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.left_val = left_val
        self.right_val = right_val


class Leaf:
    def __init__(self, data):
        self.prediction = stats.mode(data[:, -1]).mode[0]


def build_tree(data, max_depth, column_headers, depth=0):
    if depth == 0:
        print_counts(label_counts(data))

    if depth < max_depth:
        depth = depth + 1

        # find criteria to split
        split_index, mi = split_criteria(data)

        if mi > 0:
            split_dict = splitter(data, split_index)

            print(*["|" * depth], end=' ')
            print(column_headers[split_index] + " = {}: ".format(list(split_dict.keys())[0]), end='')
            left_val = list(split_dict.keys())[0]
            print(print_counts(label_counts(split_dict[left_val])))
            left_branch = build_tree(split_dict[left_val], max_depth, column_headers, depth)

            print(*["|" * depth], end=' ')
            print(column_headers[split_index] + " = {}: ".format(list(split_dict.keys())[1]), end='')
            print(print_counts(label_counts(split_dict[list(split_dict.keys())[1]])))
            right_val = list(split_dict.keys())[1]
            right_branch = build_tree(split_dict[right_val], max_depth, column_headers, depth)

        else:
            return Leaf(data)

        return DecisionNode(split_index, left_branch, right_branch, left_val, right_val)
    else:
        return Leaf(data)


def predict(model, row):
    if isinstance(model, Leaf):
        output = model.prediction
        return np.array(output,dtype="<U10")
    else:
        attr_index = model.split_attr
        if row[attr_index] == model.left_val:
            return predict(model.left_branch, row)
        else:
            return predict(model.right_branch, row)

if __name__ == '__main__':

    train_loc = sys.argv[1]
    test_loc = sys.argv[2]
    max_depth = int(sys.argv[3])

    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    train_df = np.genfromtxt(fname=train_loc, delimiter="\t", skip_header=1, dtype="|U10")
    test_df = np.genfromtxt(fname=test_loc, delimiter="\t", skip_header=1, dtype="|U10")

    with open(train_loc, 'r') as f:
        column_headers = str.split(f.readline(), "\t")

    with io.StringIO() as buf, redirect_stdout(buf):
        trained_tree = build_tree(train_df, max_depth, column_headers)
        tree_print = buf.getvalue()

    print(tree_print.replace("None\n", "").rstrip())

    train_predictions = np.apply_along_axis(lambda x: predict(trained_tree, x), 1, train_df)
    test_predictions = np.apply_along_axis(lambda x: predict(trained_tree, x), 1, test_df)

    np.savetxt(train_out,train_predictions, fmt="%s")
    np.savetxt(test_out, test_predictions, fmt="%s")

    y = train_df[:,-1]
    y_hat = train_predictions
    error_train = sum(y != y_hat) / train_df.shape[0]

    y = test_df[:,-1]
    y_hat = test_predictions
    error_test = sum(y != y_hat) / test_df.shape[0]

    with open(metrics_out, 'w') as f:
        print("""error(train): {}\nerror(test): {}""".format(error_train, error_test), file=f)
