import os.path as osp
import numpy as np
from argparse import ArgumentParser
from scipy.misc import imsave

from utils import unpickle, write_list, mkdir_if_missing


def make_data(data, labels, output_dir, prefix):
    image_dir = prefix + '_images/'
    mkdir_if_missing(osp.join(output_dir, image_dir))
    file_label_list = []
    num = len(data)
    for i in xrange(num):
        img = np.rollaxis(data[i, :].reshape((3, 32, 32)), 0, 3)
        filename = '{:05d}.jpg'.format(i)
        imsave(osp.join(output_dir, image_dir, filename), img)
        file_label_list.append('{} {}'.format(
            osp.join(image_dir, filename), int(labels[i])))
    write_list(file_label_list, osp.join(output_dir, prefix + '.txt'))


def main(args):
    mkdir_if_missing(args.output_dir)
    # training data
    data = []
    labels = []
    for i in xrange(1, 6):
        dic = unpickle(osp.join(args.data_root, 'data_batch_{}'.format(i)))
        data.append(dic['data'])
        labels = np.r_[labels, dic['labels']]
    data = np.vstack(data)
    make_data(data, labels, args.output_dir, 'train')
    # test data
    dic = unpickle(osp.join(args.data_root, 'test_batch'))
    make_data(dic['data'], dic['labels'], args.output_dir, 'test')


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Convert raw CIFAR-10 dataset to standard format")
    parser.add_argument('data_root')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    main(args)