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
 - dataset for testing: next_new_classification_test.h5
 
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

### IO Test

If you want to only study how long it takes for IO operations, without actually training the network, run:

`python bin/resnet3d.py iotest -f next_new_classification_train.h5`


## Input files

You can convert files from NEXT HDF5 to larcv HDF5 using the python script [to_larcv3.py](to_larcv3.py). 
Open the script and modify the `top_level_path` appropriately. For example, this will be the path to the directory that contains all the Tl208_NEW_v1_03_01_nexus_v5_03_04_cut*.NN_points_10000.root.h5 MC files.

The conversion script will produce three files: next_new_classification_train.h5, next_new_classification_test.h5, next_new_classification_val.h5.

The number of signal and background events in this dataset is not the same, and this requires to apply weights to signal and background to balance the loss. If you are using this dataset, you can just add the flag `-bl`, which will weight signal events with 1.60 and background events with 0.62. You can also specify your own weights by adding flags `--weight-sig 1.60 --weight-bkg 0.62`. 

On some machines the MC files are already available here:

| Machine         | Path to files  |
| ----------------|-------------|
| Summit (NPH133) | `/gpfs/alpine/proj-shared/nph133/nextnew/nextnew_Tl208_larcv/` |
| gpu1next        | `/home/deltutto/next_data_larcv/`    |


## Analyze the output

The output in the log directory contains both checkpoints (snaphot of the trained model taken every 100 iterations), and tensorboard files. If you have tensorboard installed, you can look at the results by doing:

`tensorboard --logdir /path/to/log/dir/ [--host localhost]`


## Distributed training

To run in distributed mode, just add the `-d` flag.

For example, on Summit, to run over 3 nodes with 6 GPUs each:

`jsrun -n18 -g1 -a1 -c7 -r6 python bin/resnet3d.py train -f next_new_classification_train.h5 --aux-file next_new_classification_test.h5 -i 10 -mb 1152 -bl -d`

In this way, the 1152 images specified in the batch size are read by the last rank and then distributed to all the other ranks. In the end, every rank will have 64 images per iteration.

## Inference

To run in inference mode, use `inference` instead of `train`, set the minibatch size to 1 and the number of iterations to the total number of entry in the inferece file. In this case I am using NEXT data run 6826, which contains 4175 entries. I am also specifing an output file, here called `inference_run6826_Enorm.txt`, that contains the results of the network classification.

`python bin/resnet3d.py inference -f run6826_full_dataset_cor_dv_larcv.h5 --producer voxels_E_norm -mb 1 -i 4175 -ld /gpfs/alpine/scratch/deltutto/nph133/next/log_next_mpiio_n10_r6_mb30720_bpl2_voxels_E_norm -out inference_run6826_Enorm.txt`
 
The output file will look like:
```
event,label,energy
0,1,1.0
1,1,1.0
2,0,1.0
3,0,1.0
```
where the first column shows the event number, the second the resul of the classification (1 for signal and 0 for background) and the third the total event energy (here equal to 1 because the energy was normlized).
