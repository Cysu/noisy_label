import sys
import numpy as np
from argparse import ArgumentParser
from ast import literal_eval


if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe

from utils import unpickle


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source', help="Pickled data file")
    parser.add_argument('output', help="Output file path")
    parser.add_argument('--shape', help="(num, channels, height, width)")
    args = parser.parse_args()

    data = unpickle(args.source)
    if args.shape is not None:
        data = data.reshape(literal_eval(args.shape))
    else:
        shape = data.shape
        shape = (1,) * (4 - len(shape)) + shape
        data = data.reshape(shape)
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.num, blob.channels, blob.height, blob.width = data.shape
    blob.data.extend(list(data.ravel().astype(float)))
    with open(args.output, 'wb') as f:
        f.write(blob.SerializeToString())
