# NEXT Sparse Event Identification

This repository contains some networks to do signal/background separation.  The initial push will contain networks based on a Residual architecture in both dense and sparse implementations.


## Dependencies
 - larcv3 (for IO)
 - pytorch (for training)
 - horovod (for distributed training)
 - tensorboardX (for logging of training data)
 - SparseConvNet (for sparse networks)

Eventually, I want to add PointNet, PointNet++, and DGCNN (Edgeconvs for graph networks)


## Running

Assuming you have these input files:
 - dataset for training: next_new_classification_train.h5
 - dataset for training: next_new_classification_test.h5
 
To train the network, run:

`python bin/resnet3d.py train -f next_new_classification_train.h5`

To also evaluate the performances on test set, add `--aux-file next_new_classification_test.h5`. Loss and accuracy on the test set are calculated every 10 iterations on the train set.

By default, the output is saved in `./log`. You can overwrite the output directory by adding:

`-ld /path/to/log/dir/`

By default, the network is trained for 5000 iterations. You can overwrite this with:

`-i number_of_iterations`

The model is automatically saved every 100 iterations.

To list all the available options, run 

`python bin/resnet3d.py --help` or `python bin/resnet3d.py train --help`


## Input files

You can convert files from NEXT HDF5 to larcv HDF5 using the python script [to_larcv3.py](to_larcv3.py). 
Open the script and modify the `top_level_path` appopriately. For example, this will be the path to the directory that contains all the Tl208_NEW_v1_03_01_nexus_v5_03_04_cut*.NN_points_10000.root.h5 MC files.

The conversion script will produce three files: next_new_classification_train.h5, next_new_classification_test.h5, next_new_classification_val.h5.

The number of signal and background events in this dataset is not the same, and this requires to apply weights to signal and background to balance the loss. If you are using this dataset, you can just add the flag `-bl`, which will weight signal events with 1.60 and background events with 0.62. You can also specify your own weights by adding flags `--weight-sig 1.60 --weight-bkg 0.62`. 


## Analyze the output

The output in the log directory contains both checkpoints (snaphot of the trained model taken every 100 iterations), and tensorboard files. If you have tensorboard installed, you can quickly look at the results by doing:

`tensorboard --logdir /path/to/log/dir/ [--host localhost]`


## Distributed training

To be written...

## Inference

To be written...



