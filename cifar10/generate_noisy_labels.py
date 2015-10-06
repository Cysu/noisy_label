import os.path as osp
import numpy as np
from argparse import ArgumentParser

from utils import read_list, write_list, pickle


def write_matrix(mat, file_path):
    content = [' '.join(map(str, r)) for r in mat]
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))


def generate_matrix_q(noise_level, n=10):
    q = np.random.rand(n, n).astype(np.float32)
    for j in xrange(n):
        q[j, j] = 1.0 - noise_level
        s = sum(q[:, j]) - q[j, j]
        for i in xrange(n):
            if i != j:
                q[i, j] = q[i, j] * noise_level / s
        q[:, j] /= sum(q[:, j])
    return q


def parse(file_path):
    lines = read_list(file_path)
    lines = map(str.split, lines)
    files, labels = zip(*lines)
    labels = map(int, labels)
    return (files, labels)


def corrupt(labels, q):
    n = len(q)
    cdf = np.cumsum(q, axis=0)
    cdf[n - 1, :] = 1.0
    noisy_labels = []
    for y in labels:
        r = np.random.rand()
        for k in xrange(n):
            if r <= cdf[k, y]:
                noisy_labels.append(k)
                break
    assert len(noisy_labels) == len(labels)
    return noisy_labels


def write_file_label_list(files, labels, file_path):
    content = ['{} {}'.format(f, l) for f, l in zip(files, labels)]
    write_list(content, file_path)


def main(args):
    q = generate_matrix_q(args.level)
    write_matrix(q, osp.join(args.data_root, 'matrix_q.txt'))
    pickle(q, osp.join(args.data_root, 'matrix_q.pkl'))
    files, labels = parse(osp.join(args.data_root, 'train.txt'))
    noisy_labels = corrupt(labels, q)
    write_file_label_list(files[:10000], labels[:10000],
        osp.join(args.data_root, 'clean_train.txt'))
    write_file_label_list(files[:10000], noisy_labels[:10000],
        osp.join(args.data_root, 'noisy_train.txt'))

    noisy_as_clean_labels = labels[:10000] + noisy_labels[10000:]
    noisy_as_none_labels = labels[:10000] + [-1] * 40000
    clean_as_none_labels = [-1] * 10000 + noisy_labels[10000:]
    merged = zip(files, noisy_as_clean_labels, noisy_as_none_labels, clean_as_none_labels)
    np.random.shuffle(merged)
    files, nacl, nanl, canl = zip(*merged)
    write_file_label_list(files, nacl,
                          osp.join(args.data_root, 'mixed_train.txt'))
    write_list([f + ' -1' for f in files],
               osp.join(args.data_root, 'mixed_train_images.txt'))
    write_list(nanl, osp.join(args.data_root, 'mixed_train_label_clean.txt'))
    write_list(canl, osp.join(args.data_root, 'mixed_train_label_noisy.txt'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Generate noisy labels of given level")
    parser.add_argument('data_root',
        help="Root directory containing train.txt")
    parser.add_argument('--level', type=float, default=0.5)
    args = parser.parse_args()
    main(args)