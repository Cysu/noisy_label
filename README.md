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

        mkdir external/exp

    or

        ln -s /path/to/your/exp/directory external/exp

    Then

        mkdir external/exp/datasets
        mkdir external/exp/db
        mkdir external/exp/models

## CIFAR-10 Experiments

1.  Download the CIFAR-10 data (python version).

        scripts/download_cifar10.sh

2.  Synthesize label noise and prepare LMDBs. Will corrupt the labels of 40k randomly selected training images, while leaving other 10k image labels unchanged.

        scripts/make_data_cifar10.sh 0.3

    The parameter 0.3 controls the level of label noise. Can be any number between [0, 1].

3.  Run a series of experiments

        # Train a CIFAR10-quick model using only the 10k clean labeled images
        scripts/train_cifar10_clean.sh

        # Treat 40k noisy labels as ground truth and finetune from the previous model
        scripts/train_cifar10_noisy_gt_ft_clean.sh

        # Our method
        scripts/train_cifar10_ntype.sh
        scripts/init_cifar10_noisy_label_loss.sh
        scripts/train_cifar10_noisy_label_loss.sh

We provide the training logs in logs/ just for your reference.

## Reference

    @inproceedings{xiao2015learning,
      title={Learning from Massive Noisy Labeled Data for Image Classification},
      author={Xiao, Tong and Xia, Tian and Yang, Yi and Huang, Chang and Wang, Xiaogang},
      booktitle={CVPR},
      year={2015}
    }