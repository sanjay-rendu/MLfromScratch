import numpy as np
import sys

def idx_dict(file_loc):
    idx_map = {}
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

if __name__ == '__main__':
    train_loc = sys.argv[1]
    idx_word_loc = sys.argv[2]
    idx_tag_loc = sys.argv[3]

    pi_out = sys.argv[4]
    B_out = sys.argv[5]
    A_out = sys.argv[6]


    tag_map = idx_dict(idx_tag_loc)
    word_map = idx_dict(idx_word_loc)

    sentences, label_seq = create_features(train_loc, word_map, tag_map)
    #val_sen, val_lab_seq = create_features(val_loc, word_map, tag_map)

    A = np.ones((len(tag_map), len(tag_map)))
    B = np.ones((len(tag_map), len(word_map)))
    pi = np.ones((len(tag_map), 1))

    for i in range(len(label_seq)):
        labels = label_seq[i]
        words = sentences[i]
        for j in range(0, len(labels)):
            label = labels[j]
            word = words[j]
            if j == 0:
                pi[label] += 1
            B[label, word] += 1
            if j > 0:
                A[labels[j-1], labels[j]] += 1

    A = A / A.sum(axis=1)[:, np.newaxis]
    B = B / B.sum(axis=1)[:, np.newaxis]
    pi = pi / pi.sum()

    np.savetxt(pi_out, pi)
    np.savetxt(A_out, A)
    np.savetxt(B_out, B)
