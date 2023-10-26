#!/bin/bash -l
#PBS -l select=3:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=256
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-09-08
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

run_id=simclr-supervised_ID_mb${GLOBAL_BATCH_SIZE}-3e-4

WEIGHTS="/home/cadams/Polaris/NEXT_SparseEventID/output/repr_mb10240-lr1e-2-adam-benchmark-dev-blur/checkpoints/epoch\=95-step\=4950.ckpt"
# WEIGHTS="/home/cadams/Polaris/NEXT_SparseEventID/output/repr_mb8192-lr1e-3-smallerAug/checkpoints/epoch\=55-step\=3800.ckpt"
echo $WEIGHTS

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name supervised_eventID \
mode=train \
mode.weights_location=${WEIGHTS} \
mode.optimizer.lr_schedule.peak_learning_rate=0.0003 \
run.distributed=True \
run.id=${run_id} \
framework.oversubscribe=1 \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.length=150

# data.image_key=lr_hits \
