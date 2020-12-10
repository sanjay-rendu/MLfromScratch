import numpy as np
import sys
from collections import OrderedDict

def idx_dict(file_loc):
    idx_map = OrderedDict()
    i = 0
    with open(file_loc) as file:
        for line in file.read().splitlines():
            idx_map[line] = i
            i += 1
    return idx_map


def create_features(file_loc, word_map, tag_map):
    with open(file_loc) as file:
        sentences = []
        label_seq = []
        for line in file.read().splitlines():
            wt = line.split(" ")
            words = [word_map[x.split("_")[0]] for x in wt]
            tags = [tag_map[x.split("_")[1]] for x in wt]

            sentences.append(words)
            label_seq.append(tags)

    return sentences, label_seq


def calc_alpha_beta(A, B, pi, val_sen):
    alpha_mat = np.empty((len(val_sen), len(A)))  # tag, word
    for i in range(len(val_sen)):
        if i == 0:
            alpha = pi.T * B[:, val_sen[i]]
        else:
            alpha = B[:, val_sen[i]] * np.dot(A.T, alpha_mat[i - 1])
        alpha_mat[i] = alpha

    beta_mat = np.empty((len(val_sen), len(A)))
    for i in range(len(val_sen) - 1, -1, -1):
        if i == (len(val_sen) - 1):
            beta = np.ones((1, len(A)))
        else:
            beta = B[:, val_sen[i]] * np.dot(A, beta_mat[i + 1])

        beta_mat[i] = beta

        loglike = np.log(alpha_mat[-1].sum())

    return alpha_mat, beta_mat, loglike

if __name__ == '__main__':
    val_loc = sys.argv[1]
    idx_word_loc = sys.argv[2]
    idx_tag_loc = sys.argv[3]

    pi_out = sys.argv[4]
    B_out = sys.argv[5]
    A_out = sys.argv[6]

    prediction = sys.argv[7]
    metrics = sys.argv[8]

    tag_map = idx_dict(idx_tag_loc)
    word_map = idx_dict(idx_word_loc)
    val_sen, val_lab_seq = create_features(val_loc, word_map, tag_map)

    with open(pi_out) as file:
        pi = np.loadtxt(file)

    with open(B_out) as file:
        B = np.loadtxt(file)

    with open(A_out) as file:
        A = np.loadtxt(file)

    likelihood = []
    all_preds = []
    with open(prediction, 'w') as f:

        for sentence in val_sen:
            alpha, beta, loglike = calc_alpha_beta(A, B, pi, sentence)
            alpha_beta = alpha * beta
            pred_index = alpha_beta.argmax(axis=1)
            preds = []
            for idx in range(len(sentence)):
                preds.append("{}_{}".format(list(word_map.keys())[sentence[idx]],
                                            list(tag_map.keys())[pred_index[idx]]))

            print(" ".join(preds), file=f)
            all_preds += list(pred_index)
            likelihood += [loglike]

    check = (np.array([item for sublist in val_lab_seq for item in sublist]) == np.array(all_preds))

    with open(metrics, 'w') as f:
        print("Average Log-Likelihood: {}".format(sum(likelihood)/len(likelihood)), file=f)
        print("Accuracy: {}".format(sum(check)/len(check)), file=f)
