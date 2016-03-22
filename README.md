# CVPR15 Noisy Label Project

The repository contains the code of our CVPR15 paper *Learning from Massive Noisy Labeled Data for Image Classification* ([paper link](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)).

## Installation

1.  Clone this repository

        # Make sure to clone with --recursive to get the modified Caffe
        git clone --recursive https://github.com/Cysu/noisy_label.git

2.  Build the Caffe

        cd external/caffe
        # Now follow the Caffe installation instructions here:
        #   http://caffe.berkeleyvision.org/installation.html

        # If you're experienced with Caffe and have all of the requirements installed
        # and your Makefile.config in place, then simply do:
        make -j8 && make py

        cd -

3.  Setup an experiment directory. You can either create a new one under external/, or make a link to another existing directory.

        mkdir -p external/exp

    or

        ln -s /path/to/your/exp/directory external/exp

## CIFAR-10 Experiments

1.  Download the CIFAR-10 data (python version).

        scripts/cifar10/download_cifar10.sh

2.  Synthesize label noise and prepare LMDBs. Will corrupt the labels of 40k randomly selected training images, while leaving other 10k image labels unchanged.

        scripts/cifar10/make_db.sh 0.3

    The parameter 0.3 controls the level of label noise. Can be any number between [0, 1].

3.  Run a series of experiments

        # Train a CIFAR10-quick model using only the 10k clean labeled images
        scripts/cifar10/train_clean.sh

        # Baseline:
        # Treat 40k noisy labels as ground truth and finetune from the previous model
        scripts/cifar10/train_noisy_gt_ft_clean.sh

        # Our method
        scripts/cifar10/train_ntype.sh
        scripts/cifar10/init_noisy_label_loss.sh
        scripts/cifar10/train_noisy_label_loss.sh

We provide the training logs in `logs/cifar10/` for reference.

## Clothing1M Experiments

Clothing1M is the dataset we proposed in our paper.

1.  Download the dataset. Please contact *xiaotong[at]ee[dot]cuhk.edu.hk* to get the download link. Untar the images and unzip the annotations under `external/exp/datasets/clothing1M`. The directory structure should be

        external/exp/datasets/clothing1M/
        ├── category_names_chn.txt
        ├── category_names_eng.txt
        ├── clean_label_kv.txt
        ├── clean_test_key_list.txt
        ├── clean_train_key_list.txt
        ├── clean_val_key_list.txt
        ├── images
        │   ├── 0
        │   ├── ⋮
        │   └── 9
        ├── noisy_label_kv.txt
        ├── noisy_train_key_list.txt
        ├── README.md
        └── venn.png

2.  Make the LMDBs and compute the matrix C to be used.

        scripts/clothing1M/make_db.sh

3.  Run experiments for our method

        # Download the ImageNet pretrained CaffeNet
        wget -P external/exp/snapshots/ http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel

        # Train the clothing prediction CNN using only the clean labeled images
        scripts/clothing1M/train_clean.sh

        # Train the noise type prediction CNN
        scripts/clothing1M/train_ntype.sh

        # Train the whole net using noisy labeled data
        scripts/clothing1M/init_noisy_label_loss.sh
        scripts/clothing1M/train_noisy_label_loss.sh

We provide the training logs in `logs/clothing1M/` for reference.

## Reference

    @inproceedings{xiao2015learning,
      title={Learning from Massive Noisy Labeled Data for Image Classification},
      author={Xiao, Tong and Xia, Tian and Yang, Yi and Huang, Chang and Wang, Xiaogang},
      booktitle={CVPR},
      year={2015}
    }