import os.path as osp
import sys
from argparse import ArgumentParser


if 'external/caffe/python' not in sys.path:
    sys.path.insert(0, 'external/caffe/python')
import caffe


def main(args):
    src_net = caffe.Net(args.source_model, args.source_weights, caffe.TRAIN)
    if not osp.isfile(args.target_weights):
        tgt_net = caffe.Net(args.target_model, caffe.TRAIN)
    else:
        tgt_net = caffe.Net(args.target_model, args.target_weights, caffe.TRAIN)

    src_names = src_net.params.keys()
    tgt_names = tgt_net.params.keys()
    if args.layers_dict is not None:
        src2tgt = eval(args.layers_dict)
        same = set(src_names) & set(tgt_names)
        for name in same:
            if name in src2tgt and src2tgt[name] != name:
                raise ValueError("Conflict in layers dict {} vs. {}".format(
                    name, src2tgt[name]))
            src2tgt[name] = name
    else:
        # Copy the parameters of same layers by default.
        src2tgt = {name: name for name in (set(src_names) & set(tgt_names))}

    for src_name, tgt_name in src2tgt.iteritems():
        if src_name not in src_names:
            raise ValueError("Source net doesn't have layer " + src_name)
        if tgt_name not in tgt_names:
            raise ValueError("Target net doesn't have layer " + tgt_name)
        print "Copy layer", src_name, "=>", tgt_name
        for p, q in zip(tgt_net.params[tgt_name], src_net.params[src_name]):
            p.data[...] = q.data

    tgt_net.save(args.target_weights)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="Copy parameters from one net to the other where the "
                    "source and target layer names can be different.")
    parser.add_argument('--source-model', required=True,
        help="Model definition of the source net")
    parser.add_argument('--target-model', required=True,
        help="Model definition of the target net")
    parser.add_argument('--source-weights', required=True,
        help="Weights caffemodel of the source net")
    parser.add_argument('--target-weights', required=True,
        help="Weights caffemodel of the target net")
    parser.add_argument('--layers-dict',
        help="Python code that returns a dict like "
             "{'src_layer_1': 'tgt_layer_1', 'src_layer_2': 'tgt_layer_2'}")
    args = parser.parse_args()
    main(args)