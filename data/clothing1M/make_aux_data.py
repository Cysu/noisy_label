import os
import sys
import numpy as np
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

from utils import read_kv, read_list, write_list, pickle

if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe

def compute_matrix_c(clean_labels, noisy_labels):
    cm = confusion_matrix(clean_labels, noisy_labels)
    cm -= np.diag(np.diag(cm))
    cm = cm * 1.0 / cm.sum(axis=1, keepdims=True)
    cm = cm.T
    L = len(cm)
    alpha = 1.0 / (L - 1)
    C = np.zeros((L, L))
    for j in xrange(L):
        f = cm[:, j].ravel()
        f = zip(f, xrange(L))
        f.sort(reverse=True)
        best_lik = -np.inf
        best_i = -1
        for i in xrange(L + 1):
            c = np.zeros((L,))
            for k in xrange(0, i):
                c[k] = f[k][0]
            if c.sum() > 0:
                c /= c.sum()
            lik = 0
            for k in xrange(0, i):
                lik += f[k][0] * np.log(c[k])
            for k in xrange(i, L):
                lik += f[k][0] * np.log(alpha)
            if lik >= best_lik:
                best_lik = lik
                best_i = i
            if i < L and f[i][0] == 0:
                break
        for k in xrange(0, best_i):
            C[f[k][1], j] = f[k][0]
    return C / C.sum(axis=0)


def save_to_blobproto(data, output_file):
    shape = (1,) * (4 - data.ndim) + data.shape
    data = data.reshape(shape)
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.num, blob.channels, blob.height, blob.width = data.shape
    blob.data.extend(list(data.ravel().astype(float)))
    with open(output_file, 'wb') as f:
        f.write(blob.SerializeToString())


def make_aux_clean(data_root, output_dir):
    label_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    def _make(token):
        keys = read_list(os.path.join(data_root,
            'clean_{}_key_list.txt'.format(token)))
        lines = [k + ' ' + label_kv[k] for k in keys]
        np.random.shuffle(lines)
        output_file = os.path.join(output_dir, 'clean_{}.txt'.format(token))
        write_list(lines, output_file)
    _make('train')
    _make('val')
    _make('test')


def make_aux_ntype(data_root, output_dir):
    clean_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    noisy_kv = dict(zip(*read_kv(os.path.join(data_root, 'noisy_label_kv.txt'))))
    train_keys = set(read_list(os.path.join(data_root, 'clean_train_key_list.txt')))
    val_keys = set(read_list(os.path.join(data_root, 'clean_val_key_list.txt')))
    test_keys = set(read_list(os.path.join(data_root, 'clean_test_key_list.txt')))
    noisy_keys = set(noisy_kv.keys())
    # compute and save matrix C
    keys = (train_keys | val_keys) & noisy_keys
    clean_labels = np.asarray([int(clean_kv[k]) for k in keys])
    noisy_labels = np.asarray([int(noisy_kv[k]) for k in keys])
    C = compute_matrix_c(clean_labels, noisy_labels)
    save_to_blobproto(C, os.path.join(output_dir, 'matrix_c.binaryproto'))
    # make noise type (ntype)
    def _make(keys, token):
        clean_labels = np.asarray([int(clean_kv[k]) for k in keys])
        noisy_labels = np.asarray([int(noisy_kv[k]) for k in keys])
        lines = []
        alpha = 1.0 / (C.shape[0] - 1)
        for key, y, y_tilde in zip(keys, clean_labels, noisy_labels):
            if y == y_tilde:
                lines.append(key + ' 0')
            elif alpha >= C[y_tilde][y]:
                lines.append(key + ' 1')
            else:
                lines.append(key + ' 2')
        np.random.shuffle(lines)
        output_file = os.path.join(output_dir, 'ntype_{}.txt'.format(token))
        write_list(lines, output_file)
    _make(train_keys & noisy_keys, 'train')
    _make(val_keys & noisy_keys, 'val')
    _make(test_keys & noisy_keys, 'test')


def make_aux_mixed(data_root, output_dir, upsample_ratio=1.0):
    ntype_kv = dict(zip(*read_kv(os.path.join(output_dir, 'ntype_train.txt'))))
    clean_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    noisy_kv = dict(zip(*read_kv(os.path.join(data_root, 'noisy_label_kv.txt'))))
    clean_keys = read_list(os.path.join(data_root, 'clean_train_key_list.txt'))
    noisy_keys = read_list(os.path.join(data_root, 'noisy_train_key_list.txt'))
    # upsampling clean keys to ratio * #noisy_keys
    clean_keys = np.random.choice(clean_keys, len(noisy_keys) * upsample_ratio)
    # mix clean and noisy data
    keys = list(clean_keys) + list(noisy_keys)
    np.random.shuffle(keys)
    clean, noisy, ntype = [], [], []
    for k in keys:
        if k in clean_kv:
            clean.append(clean_kv[k])
            noisy.append('-1')
        else:
            clean.append('-1')
            noisy.append(noisy_kv[k])
        if k in ntype_kv:
            ntype.append(ntype_kv[k])
        else:
            ntype.append('-1')
    keys = [k + ' -1' for k in keys]
    write_list(keys, os.path.join(output_dir, 'mixed_train_images.txt'))
    write_list(clean, os.path.join(output_dir, 'mixed_train_label_clean.txt'))
    write_list(noisy, os.path.join(output_dir, 'mixed_train_label_noisy.txt'))
    write_list(ntype, os.path.join(output_dir, 'mixed_train_label_ntype.txt'))


def main(args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    make_aux_clean(args.data_root, args.output_dir)
    make_aux_ntype(args.data_root, args.output_dir)
    make_aux_mixed(args.data_root, args.output_dir)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Generate auxiliary files used for the clothing1M dataset")
    parser.add_argument('data_root',
        help="Directory path to the clothing1M dataset, containing images/ and "
             "other meta files.")
    parser.add_argument('output_dir', help="Output directory")
    args = parser.parse_args()
    main(args)