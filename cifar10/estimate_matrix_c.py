import os.path as osp
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import confusion_matrix

from utils import read_list, write_list, pickle


def write_matrix(mat, file_path):
    content = [' '.join(map(str, r)) for r in mat]
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))


def parse(file_path):
    lines = read_list(file_path)
    lines = map(str.split, lines)
    files, labels = zip(*lines)
    labels = map(int, labels)
    return files, labels


def compute_matrix_c(clean_labels, noisy_labels):
    cm = confusion_matrix(clean_labels, noisy_labels)
    cm -= np.diag(np.diag(cm))
    cm = cm * 1.0 / cm.sum(axis=1, keepdims=True)
    cm = cm.T
    n = len(cm)
    uni = 1.0 / (n - 1)
    matrix_c = np.zeros((n, n))
    for j in xrange(n):
        f = sorted(zip(cm[:, j].ravel(), xrange(n)), reverse=True)
        best_lik, best_i = -np.inf, -1
        for i in xrange(n + 1):
            c = np.asarray([f[k][0] for k in xrange(i)] + [0] * (n - i))
            if sum(c) > 0:
                c /= sum(c)
            lik = sum([f[k][0] * np.log(c[k]) for k in xrange(i)] +
                      [f[k][0] * np.log(uni) for k in xrange(i, n)])
            if lik >= best_lik:
                best_lik, best_i = lik, i
            if i < n and f[i][0] == 0:
                break
        for k in xrange(best_i):
            matrix_c[f[k][1], j] = f[k][0]
    return matrix_c / matrix_c.sum(axis=0)


def get_noise_types(clean_labels, noisy_labels, matrix_c):
    noise_types = []
    uni = 1.0 / (len(matrix_c) - 1)
    for i, j in zip(clean_labels, noisy_labels):
        if i == j:
            noise_types.append(0)
        elif matrix_c[j, i] < uni:
            noise_types.append(1)
        else:
            noise_types.append(2)
    return noise_types


def make_data(files, noise_types, data_root):
    # noise types training and val
    merged = zip(files, noise_types)
    np.random.shuffle(merged)
    training = ['{} {}'.format(f, t) for f, t in merged[:8000]]
    test = ['{} {}'.format(f, t) for f, t in merged[8000:]]
    write_list(training, osp.join(data_root, 'ntype_train.txt'))
    write_list(test, osp.join(data_root, 'ntype_test.txt'))
    # noise types of mixed training images
    dic = defaultdict(lambda: -1)
    dic.update(dict(zip(files, noise_types)))
    files = read_list(osp.join(data_root, 'mixed_train_images.txt'))
    files = [f.split()[0] for f in files]
    noise_types = [dic[f] for f in files]
    write_list(noise_types, osp.join(data_root, 'mixed_train_label_ntype.txt'))


def main(args):
    files, clean_labels = parse(osp.join(args.data_root, 'clean_train.txt'))
    files, noisy_labels = parse(osp.join(args.data_root, 'noisy_train.txt'))
    matrix_c = compute_matrix_c(clean_labels, noisy_labels)
    write_matrix(matrix_c, osp.join(args.data_root, 'matrix_c.txt'))
    pickle(matrix_c, osp.join(args.data_root, 'matrix_c.pkl'))
    noise_types = get_noise_types(clean_labels, noisy_labels, matrix_c)
    make_data(files, noise_types, args.data_root)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Estimate the confusing matrix C in our method")
    parser.add_argument('data_root',
        help="Root directory containing clean_train.txt and noisy_train.txt")
    args = parser.parse_args()
    main(args)